import { expect, type Locator, type Page } from '@playwright/test'
import { waitForHydration } from './hydration'

/**
 * Page object for the Transportation Requests listing
 * (features/transportation/transportation-requests-listing).
 *
 * The listing is fed by an in-memory mock array that `submitTransportationRequest`
 * mutates, so a submitted request is only visible for as long as the page is not
 * reloaded. Navigate here in-app (or via `goto()` before submitting) - a
 * `page.goto()` after a submit would wipe the very row under test.
 */
export class TransportationRequestsListingPage {
  readonly url = '/dashboard/requests/transportation'
  readonly rows: Locator

  constructor(private readonly page: Page) {
    this.rows = page.locator('tbody tr')
  }

  async goto() {
    await this.page.goto(this.url)
    await waitForHydration(this.page)
    await this.waitForRows()
  }

  /** The table renders a loading state first; wait it out before counting rows. */
  async waitForRows() {
    await expect(this.rows.first()).toBeVisible()
  }

  row(text: string): Locator {
    return this.rows.filter({ hasText: text })
  }
}
