import { expect, type Locator, type Page } from '@playwright/test'
import { waitForHydration } from './hydration'
import type { CargoDetails, LocationDetails, Priority, TransportationRequestData } from './request-data'

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** Step titles in wizard order - also the `<h2>` of the step card. */
export const STEP_TITLES = [
  'Service Type',
  'Pickup Information',
  'Delivery Information',
  'Cargo Information',
  'Special Instructions',
  'Review & Submit',
] as const

export type StepTitle = (typeof STEP_TITLES)[number]

/** Labels of the progress timeline above the form - shorter than the step titles. */
export const TIMELINE_STEPS = [
  'Service Type',
  'Pickup',
  'Delivery',
  'Cargo',
  'Instructions',
  'Review',
] as const

export type TimelineStep = (typeof TIMELINE_STEPS)[number]

/**
 * Page object for the "New Transportation Request" wizard
 * (features/transportation/submit-transportation-request/SubmitTransportationRequestPage.vue).
 *
 * Fields are located by their label, which the source associates via for/id.
 * Only one step is mounted at a time (v-if on store.currentStep), so labels that
 * repeat across steps - "City *", "Country *", "Contact Person *" - are
 * unambiguous at any given moment. Everything is nevertheless scoped to the step
 * card, the timeline or the modal, so that a locator can never silently match the
 * progress timeline (which repeats step names) or the review summary (which
 * repeats every value entered earlier).
 */
export class NewTransportationRequestPage {
  readonly url = '/dashboard/transportation/new'

  /** The single `.card` on the page: the currently mounted step. */
  readonly stepCard: Locator
  readonly timeline: Locator
  readonly successModal: Locator
  readonly nextButton: Locator
  readonly backButton: Locator
  readonly submitButton: Locator

  constructor(private readonly page: Page) {
    this.stepCard = page.locator('.card')
    this.timeline = page.getByTestId('wizard-timeline')
    this.successModal = page.getByRole('dialog')
    this.nextButton = page.getByRole('button', { name: 'Next' })
    this.backButton = page.getByRole('button', { name: 'Back' })
    this.submitButton = page.getByRole('button', { name: /Submit Request|Submitting/ })
  }

  async goto() {
    await this.page.goto(this.url)
    await waitForHydration(this.page)
    await expect(this.page.getByRole('heading', { name: 'New Transportation Request' })).toBeVisible()
  }

  /** The step card, scoped by its heading so a stale step can never satisfy an assertion. */
  stepPanel(title: StepTitle): Locator {
    return this.page.locator('.card', {
      has: this.page.getByRole('heading', { name: title, exact: true }),
    })
  }

  async expectOnStep(title: StepTitle) {
    await expect(this.stepPanel(title)).toBeVisible()
  }

  // Step 1 - Service Type
  async selectServiceType(name: string) {
    await this.stepCard.getByText(name, { exact: true }).click()
  }

  /**
   * The radio itself is `sr-only` and covered by the label's content, so it
   * can't be clicked directly - but it does take its accessible name from the
   * wrapping <label>, which makes it the right thing to assert `toBeChecked()`
   * against. Asserting on the option's visible text instead proves nothing:
   * every option renders regardless of what is selected.
   */
  serviceTypeRadio(name: string): Locator {
    return this.stepCard.getByRole('radio', { name: new RegExp(escapeRegExp(name)) })
  }

  // Step 2 - Pickup Information
  async fillPickup(details: LocationDetails & { pickupDate: string }) {
    await this.stepCard.getByLabel('Pickup Address *').fill(details.street)
    await this.stepCard.getByLabel('City *').fill(details.city)
    await this.stepCard.getByLabel('Country *').selectOption(details.country)
    await this.stepCard.getByLabel('Contact Person *').fill(details.contactPerson)
    await this.stepCard.getByLabel('Contact Phone *').fill(details.contactPhone)
    await this.stepCard.getByLabel('Preferred Pickup Date *').fill(details.pickupDate)
  }

  // Step 3 - Delivery Information
  async fillDelivery(details: LocationDetails) {
    await this.stepCard.getByLabel('Delivery Address *').fill(details.street)
    await this.stepCard.getByLabel('City *').fill(details.city)
    await this.stepCard.getByLabel('Country *').selectOption(details.country)
    await this.stepCard.getByLabel('Contact Person *').fill(details.contactPerson)
    await this.stepCard.getByLabel('Contact Phone *').fill(details.contactPhone)
  }

  // Step 4 - Cargo Information
  async fillCargo(details: CargoDetails) {
    await this.stepCard.getByLabel('Cargo Description *').fill(details.description)
    await this.stepCard.getByLabel('Weight (kg) *').fill(String(details.weight))
  }

  // Step 5 - Special Instructions
  async fillSpecialInstructions(text: string) {
    await this.stepCard.getByLabel('Special Instructions').fill(text)
  }

  async selectPriority(name: Priority) {
    await this.stepCard.getByText(name, { exact: true }).click()
  }

  priorityRadio(name: Priority): Locator {
    return this.stepCard.getByRole('radio', { name: new RegExp(`^${escapeRegExp(name)}$`) })
  }

  // Navigation
  async goNext() {
    await this.nextButton.click()
  }

  async goBack() {
    await this.backButton.click()
  }

  async submit() {
    await this.submitButton.click()
  }

  /** A step marker in the progress timeline; clicking it jumps to that step. */
  timelineStep(name: TimelineStep): Locator {
    return this.timeline.getByText(name, { exact: true })
  }

  // Step 6 - Review & Submit
  reviewPanel(): Locator {
    return this.stepPanel('Review & Submit')
  }

  /** The "Edit" shortcut of one review section, told apart by its aria-label. */
  editSection(title: StepTitle): Locator {
    return this.reviewPanel().getByRole('button', { name: `Edit ${title}` })
  }

  /** Fills steps 1-5 and leaves the wizard on the review step. */
  async completeAllSteps(data: TransportationRequestData) {
    await this.selectServiceType(data.serviceType)
    await this.goNext()
    await this.fillPickup(data.pickup)
    await this.goNext()
    await this.fillDelivery(data.delivery)
    await this.goNext()
    await this.fillCargo(data.cargo)
    await this.goNext()
    await this.fillSpecialInstructions(data.specialInstructions)
    await this.selectPriority(data.priority)
    await this.goNext()
    await this.expectOnStep('Review & Submit')
  }

  /** Dismisses the success modal, which resets the form and lands on the listing. */
  async createAnotherRequest() {
    await this.successModal.getByRole('button', { name: 'Create Another Request' }).click()
  }
}
