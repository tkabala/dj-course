import { defineConfig, devices } from '@playwright/test'
import { STORAGE_STATE } from './tests/e2e/utils/auth'

// When PLAYWRIGHT_BASE_URL points at an already-running app (staging, a
// preview deploy, a manually started dev server), the local dev server must
// not be started - and must not be waited for on a port nothing will bind.
const externalBaseURL = process.env.PLAYWRIGHT_BASE_URL
const baseURL = externalBaseURL || 'http://localhost:3000'

export default defineConfig({
  testDir: './tests/e2e',
  globalSetup: './tests/e2e/global-setup.ts',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  // Progress on the terminal, plus a report to open afterwards. `open: 'never'`
  // keeps a local run from hijacking the browser on the first failure.
  reporter: [['list'], ['html', { open: 'never' }]],
  // The mock API sleeps 2s per submit and the dev server compiles routes on
  // demand, so the 30s default is uncomfortably tight for the full wizard.
  timeout: 60 * 1000,
  use: {
    baseURL,
    // Written by globalSetup - every test starts already signed in.
    storageState: STORAGE_STATE,
    // Not `on-first-retry`: locally retries are off, which would mean never
    // recording the trace of the one run that failed.
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  webServer: externalBaseURL
    ? undefined
    : {
        command: 'npm run dev',
        url: baseURL,
        reuseExistingServer: !process.env.CI,
        timeout: 120 * 1000,
      },
})
