import { test as base, expect } from '@playwright/test'
import { NewTransportationRequestPage } from './utils/new-transportation-request-page'
import { TransportationRequestsListingPage } from './utils/transportation-requests-listing-page'

interface Fixtures {
  requestForm: NewTransportationRequestPage
  requestsListing: TransportationRequestsListingPage
  failOnUncaughtErrors: void
}

/**
 * `test` with the page objects wired in, plus a guard that fails a test when the
 * app throws in the browser. Without it a broken component can still let every
 * assertion pass - Vue swallows the error, the UI just quietly stops updating -
 * which is exactly the class of bug an E2E suite is supposed to catch.
 *
 * Only uncaught exceptions count; the feature logs plenty of console noise on
 * purpose, and failing on that would make the suite useless.
 */
export const test = base.extend<Fixtures>({
  failOnUncaughtErrors: [
    async ({ page }, use) => {
      const errors: string[] = []
      page.on('pageerror', (error) => errors.push(error.message))

      await use()

      expect(errors, 'uncaught exceptions thrown by the page').toEqual([])
    },
    { auto: true },
  ],

  requestForm: async ({ page }, use) => {
    await use(new NewTransportationRequestPage(page))
  },

  requestsListing: async ({ page }, use) => {
    await use(new TransportationRequestsListingPage(page))
  },
})

export { expect } from '@playwright/test'
