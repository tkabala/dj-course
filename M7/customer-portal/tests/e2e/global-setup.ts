import { chromium, expect, type FullConfig } from '@playwright/test'
import { STORAGE_STATE, seedAuthState } from './utils/auth'
import { waitForHydration } from './utils/hydration'

const WARMUP_ROUTES = [
  '/dashboard',
  '/dashboard/transportation/new',
  '/dashboard/requests/transportation',
]

/**
 * Runs once before the workers start, and does two things:
 *
 * 1. Signs in and saves the storage state to STORAGE_STATE, which the config
 *    hands to every test context via `use.storageState`. Tests therefore start
 *    already authenticated, with no per-test login navigation.
 * 2. Warms the routes the suite touches. Nuxt's Vite dev server compiles each
 *    route's chunk on first visit, which triggers a full-page reload
 *    mid-navigation; if that first visit lands inside a test, the reload can
 *    race the client-side auth state and bounce back to /login. Visiting the
 *    routes here means the actual test runs hit already-compiled chunks.
 */
export default async function globalSetup(config: FullConfig) {
  const baseURL = config.projects[0]?.use?.baseURL
  if (!baseURL) throw new Error('globalSetup: no baseURL configured')

  const browser = await chromium.launch()
  const context = await browser.newContext({ baseURL })
  const page = await context.newPage()

  try {
    await seedAuthState(page, baseURL)

    for (const route of WARMUP_ROUTES) {
      // Warming a cold dev server races its own compilation: the reload Vite
      // issues once a route's chunk is ready aborts the very navigation that
      // triggered it (net::ERR_ABORTED). Retrying is the point of this loop -
      // by the time a route settles, its chunk is built and tests are safe.
      await expect(async () => {
        await page.goto(route)
        await waitForHydration(page)
      }).toPass({ timeout: 90_000 })

      // The auth middleware runs client-side, after hydration. Bouncing to
      // /login here means the seeded state did not take - fail loudly now
      // rather than letting every test fail on an unrelated assertion.
      if (new URL(page.url()).pathname === '/login') {
        throw new Error(
          `globalSetup: seeded auth state was rejected - ${route} redirected to /login. ` +
            'The keys written by seedAuthState() have probably drifted from stores/auth.ts.',
        )
      }
    }

    await context.storageState({ path: STORAGE_STATE })
  } finally {
    await browser.close()
  }
}
