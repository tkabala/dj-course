import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  retries: 1,
  use: {
    baseURL: 'https://tms-order-wizard-rwqs.bolt.host/',
    browserName: 'chromium',
  },
});
