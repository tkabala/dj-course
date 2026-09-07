import { expect, test } from './fixtures'
import { aTransportationRequest } from './utils/request-data'

test.describe('New Transportation Request', () => {
  test('submits a request and adds it to the transportation requests listing', async ({
    page,
    requestForm,
    requestsListing,
  }) => {
    const request = aTransportationRequest()

    // Count first: the listing is backed by an in-memory mock array, so "one
    // more row than before" is the only assertion that survives seeded data
    // changing.
    await requestsListing.goto()
    const rowsBefore = await requestsListing.rows.count()

    await test.step('open the wizard from the listing', async () => {
      await page.getByRole('button', { name: 'New Transportation Request' }).click()
      await page.waitForURL(requestForm.url)
      await requestForm.expectOnStep('Service Type')
    })

    await test.step('step 1 - service type', async () => {
      await expect(requestForm.nextButton).toBeDisabled()
      await requestForm.selectServiceType(request.serviceType)
      await expect(requestForm.serviceTypeRadio(request.serviceType)).toBeChecked()
      await expect(requestForm.nextButton).toBeEnabled()
      await requestForm.goNext()
    })

    await test.step('step 2 - pickup information', async () => {
      await requestForm.expectOnStep('Pickup Information')
      await requestForm.fillPickup(request.pickup)
      await expect(requestForm.nextButton).toBeEnabled()
      await requestForm.goNext()
    })

    await test.step('step 3 - delivery information', async () => {
      await requestForm.expectOnStep('Delivery Information')
      await requestForm.fillDelivery(request.delivery)
      await expect(requestForm.nextButton).toBeEnabled()
      await requestForm.goNext()
    })

    await test.step('step 4 - cargo information', async () => {
      await requestForm.expectOnStep('Cargo Information')
      await requestForm.fillCargo(request.cargo)
      await expect(requestForm.nextButton).toBeEnabled()
      await requestForm.goNext()
    })

    await test.step('step 5 - special instructions', async () => {
      await requestForm.expectOnStep('Special Instructions')
      await requestForm.fillSpecialInstructions(request.specialInstructions)
      await requestForm.selectPriority(request.priority)
      await expect(requestForm.priorityRadio(request.priority)).toBeChecked()
      await requestForm.goNext()
    })

    await test.step('step 6 - the summary repeats everything that was entered', async () => {
      await requestForm.expectOnStep('Review & Submit')
      const review = requestForm.reviewPanel()

      await expect(review.getByText(request.serviceType, { exact: true })).toBeVisible()
      await expect(review.getByText(request.pickup.street)).toBeVisible()
      await expect(
        review.getByText(`${request.pickup.city}, ${request.pickup.country}`),
      ).toBeVisible()
      await expect(review.getByText(request.delivery.street)).toBeVisible()
      await expect(
        review.getByText(`${request.delivery.city}, ${request.delivery.country}`),
      ).toBeVisible()
      await expect(review.getByText(request.cargo.description)).toBeVisible()
      await expect(review.getByText(`Weight: ${request.cargo.weight} kg`)).toBeVisible()
      await expect(review.getByText(request.specialInstructions)).toBeVisible()
      await expect(review.getByText(`Priority: ${request.priority}`)).toBeVisible()
    })

    let referenceNumber = ''

    await test.step('submit', async () => {
      await requestForm.submit()

      // The mock API sleeps 2s; the button must lock while it is in flight so a
      // double click cannot submit the request twice.
      await expect(requestForm.submitButton).toBeDisabled()
      await expect(page.getByText('Submitting...')).toBeVisible()

      await expect(requestForm.successModal).toBeVisible({ timeout: 15_000 })
      referenceNumber = (
        await requestForm.successModal.getByText(/^TR-\d{4}-\d+$/).innerText()
      ).trim()
    })

    await test.step('the new request shows up in the listing', async () => {
      await requestForm.createAnotherRequest()
      await page.waitForURL(requestsListing.url)
      await requestsListing.waitForRows()

      await expect(requestsListing.rows).toHaveCount(rowsBefore + 1)

      const newRow = requestsListing.row(referenceNumber)
      await expect(newRow).toHaveCount(1)
      await expect(newRow).toContainText(`${request.pickup.city} → ${request.delivery.city}`)
      await expect(newRow).toContainText(`${request.pickup.country} → ${request.delivery.country}`)
      await expect(newRow).toContainText('Submitted')
    })

    await test.step('the request survives navigating away and back', async () => {
      // In-app navigation only: a full reload would reset the mock store and
      // make this assertion pass or fail for the wrong reason.
      await page.getByRole('link', { name: 'Dashboard', exact: true }).click()
      await page.waitForURL('/dashboard')

      await page.goBack()
      await page.waitForURL(requestsListing.url)
      await requestsListing.waitForRows()

      await expect(requestsListing.rows).toHaveCount(rowsBefore + 1)
      await expect(requestsListing.row(referenceNumber)).toHaveCount(1)
    })
  })

  test('keeps Next disabled until every required field of the step is filled', async ({
    requestForm,
  }) => {
    const request = aTransportationRequest()
    await requestForm.goto()

    await test.step('step 1 needs a service type', async () => {
      await expect(requestForm.nextButton).toBeDisabled()
      await requestForm.selectServiceType('Express Delivery')
      await expect(requestForm.nextButton).toBeEnabled()
      await requestForm.goNext()
    })

    await test.step('step 2 needs all six pickup fields', async () => {
      await requestForm.expectOnStep('Pickup Information')
      await expect(requestForm.nextButton).toBeDisabled()

      // Filling them one by one: Next must stay disabled until the last one.
      await requestForm.stepCard.getByLabel('Pickup Address *').fill(request.pickup.street)
      await requestForm.stepCard.getByLabel('City *').fill(request.pickup.city)
      await requestForm.stepCard.getByLabel('Country *').selectOption(request.pickup.country)
      await requestForm.stepCard
        .getByLabel('Contact Person *')
        .fill(request.pickup.contactPerson)
      await requestForm.stepCard.getByLabel('Contact Phone *').fill(request.pickup.contactPhone)
      await expect(requestForm.nextButton).toBeDisabled()

      await requestForm.stepCard
        .getByLabel('Preferred Pickup Date *')
        .fill(request.pickup.pickupDate)
      await expect(requestForm.nextButton).toBeEnabled()
      await requestForm.goNext()
    })

    await test.step('step 3 needs all five delivery fields', async () => {
      await requestForm.expectOnStep('Delivery Information')
      await expect(requestForm.nextButton).toBeDisabled()
      await requestForm.fillDelivery(request.delivery)
      await expect(requestForm.nextButton).toBeEnabled()
      await requestForm.goNext()
    })

    await test.step('step 4 needs a description and a weight greater than zero', async () => {
      await requestForm.expectOnStep('Cargo Information')
      await expect(requestForm.nextButton).toBeDisabled()

      await requestForm.stepCard.getByLabel('Cargo Description *').fill(request.cargo.description)
      await expect(requestForm.nextButton).toBeDisabled()

      await requestForm.stepCard.getByLabel('Weight (kg) *').fill('0')
      await expect(requestForm.nextButton).toBeDisabled()

      await requestForm.stepCard.getByLabel('Weight (kg) *').fill(String(request.cargo.weight))
      await expect(requestForm.nextButton).toBeEnabled()
      await requestForm.goNext()
    })

    await test.step('step 5 has no required fields', async () => {
      await requestForm.expectOnStep('Special Instructions')
      await expect(requestForm.nextButton).toBeEnabled()
    })
  })

  test('preserves entered data when navigating back to a previous step', async ({
    requestForm,
  }) => {
    const request = aTransportationRequest({ serviceType: 'Oversized Cargo' })
    await requestForm.goto()

    await requestForm.selectServiceType(request.serviceType)
    await requestForm.goNext()
    await requestForm.fillPickup(request.pickup)
    await requestForm.goNext()
    await requestForm.expectOnStep('Delivery Information')

    await requestForm.goBack()
    await requestForm.expectOnStep('Pickup Information')
    await expect(requestForm.stepCard.getByLabel('Pickup Address *')).toHaveValue(
      request.pickup.street,
    )
    await expect(requestForm.stepCard.getByLabel('City *')).toHaveValue(request.pickup.city)
    await expect(requestForm.stepCard.getByLabel('Country *')).toHaveValue(request.pickup.country)
    await expect(requestForm.stepCard.getByLabel('Contact Person *')).toHaveValue(
      request.pickup.contactPerson,
    )
    await expect(requestForm.stepCard.getByLabel('Preferred Pickup Date *')).toHaveValue(
      request.pickup.pickupDate,
    )

    await requestForm.goBack()
    await requestForm.expectOnStep('Service Type')
    await expect(requestForm.serviceTypeRadio(request.serviceType)).toBeChecked()
    await expect(requestForm.serviceTypeRadio('Full Truckload (FTL)')).not.toBeChecked()
    await expect(requestForm.backButton).toBeHidden()
  })

  test('the progress timeline jumps to visited steps and ignores unreached ones', async ({
    requestForm,
  }) => {
    const request = aTransportationRequest()
    await requestForm.goto()

    await requestForm.selectServiceType(request.serviceType)
    await requestForm.goNext()
    await requestForm.fillPickup(request.pickup)
    await requestForm.goNext()
    await requestForm.expectOnStep('Delivery Information')

    await test.step('a completed step is one click away', async () => {
      await requestForm.timelineStep('Service Type').click()
      await requestForm.expectOnStep('Service Type')
      await expect(requestForm.serviceTypeRadio(request.serviceType)).toBeChecked()
    })

    await test.step('an unreached step stays out of reach', async () => {
      await requestForm.timelineStep('Review').click()
      await requestForm.expectOnStep('Service Type')
    })

    await test.step('and the completed pickup step is still reachable', async () => {
      await requestForm.timelineStep('Pickup').click()
      await requestForm.expectOnStep('Pickup Information')
      await expect(requestForm.stepCard.getByLabel('City *')).toHaveValue(request.pickup.city)
    })
  })

  test('the review step edits a section and shows the change in the summary', async ({
    requestForm,
  }) => {
    const request = aTransportationRequest()
    const correctedCity = 'Munich'

    await requestForm.goto()
    await requestForm.completeAllSteps(request)

    await expect(
      requestForm
        .reviewPanel()
        .getByText(`${request.delivery.city}, ${request.delivery.country}`),
    ).toBeVisible()

    await requestForm.editSection('Delivery Information').click()
    await requestForm.expectOnStep('Delivery Information')
    await expect(requestForm.stepCard.getByLabel('Delivery Address *')).toHaveValue(
      request.delivery.street,
    )

    await test.step('emptying a required field locks the steps behind it again', async () => {
      await requestForm.stepCard.getByLabel('City *').fill('')
      await expect(requestForm.nextButton).toBeDisabled()

      await requestForm.timelineStep('Review').click()
      await requestForm.expectOnStep('Delivery Information')
    })

    await test.step('the review step is one click away once the step is complete', async () => {
      await requestForm.stepCard.getByLabel('City *').fill(correctedCity)

      await requestForm.timelineStep('Review').click()
      await requestForm.expectOnStep('Review & Submit')
    })

    await expect(
      requestForm.reviewPanel().getByText(`${correctedCity}, ${request.delivery.country}`),
    ).toBeVisible()
    await expect(
      requestForm
        .reviewPanel()
        .getByText(`${request.delivery.city}, ${request.delivery.country}`),
    ).toBeHidden()
  })

  test('reports a required field only once the user has been in it', async ({ requestForm }) => {
    const request = aTransportationRequest()
    const cityError = 'City is required'

    await requestForm.goto()
    await requestForm.selectServiceType(request.serviceType)
    await requestForm.goNext()
    await requestForm.expectOnStep('Pickup Information')

    // Arriving on an empty step must not greet the user with a wall of red.
    await expect(requestForm.stepCard.getByText(/is required/)).toHaveCount(0)

    await test.step('leaving a required field empty explains why Next is locked', async () => {
      await requestForm.stepCard.getByLabel('City *').click()
      await requestForm.stepCard.getByLabel('Contact Person *').click()

      await expect(requestForm.stepCard.getByText(cityError)).toBeVisible()
      // Only the field the user actually visited complains.
      await expect(requestForm.stepCard.getByText('Pickup address is required')).toBeHidden()
    })

    await test.step('and the complaint goes away once the field is filled', async () => {
      await requestForm.stepCard.getByLabel('City *').fill(request.pickup.city)
      await expect(requestForm.stepCard.getByText(cityError)).toBeHidden()
    })
  })
})

test.describe('New Transportation Request - signed out', () => {
  test.use({ storageState: { cookies: [], origins: [] } })

  test('redirects an anonymous visitor to the login page', async ({ page, requestForm }) => {
    await page.goto(requestForm.url)
    await expect(page).toHaveURL(/\/login$/, { timeout: 15_000 })
  })
})
