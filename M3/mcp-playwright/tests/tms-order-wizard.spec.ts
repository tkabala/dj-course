import { test, expect, Page, Locator } from '@playwright/test';

function createTestUser() {
  const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  return {
    name: 'Test User',
    email: `testuser+${id}@example.com`,
    password: 'TestPass123!',
  };
}

async function signUp(page: Page, user: ReturnType<typeof createTestUser>) {
  await page.goto('/');
  await page.getByRole('button', { name: 'Sign Up' }).click();
  await page.getByRole('button', { name: 'Client Place orders' }).click();
  await page.getByRole('textbox', { name: 'Full Name' }).fill(user.name);
  await page.getByRole('textbox', { name: 'Email' }).fill(user.email);
  await page.getByRole('textbox', { name: 'Password' }).fill(user.password);
  await page.getByRole('button', { name: 'Create Account' }).click();
  await expect(page.getByText('Client Dashboard')).toBeVisible();
}

function sectionByHeading(page: Page, heading: string): Locator {
  return page.locator('.bg-slate-800').filter({ has: page.getByRole('heading', { name: heading, level: 3 }) });
}

/** Returns a date string `days` from today in YYYY-MM-DD format. */
function futureDate(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

test.describe('TMS Order Wizard', () => {
  test('should create a new account', async ({ page }) => {
    const user = createTestUser();
    await signUp(page, user);

    await expect(page.getByText(user.name)).toBeVisible();
    await expect(page.getByText(user.email)).toBeVisible();
  });

  test('should log into existing account', async ({ page }) => {
    const user = createTestUser();
    await signUp(page, user);

    // Log out
    await page.locator('nav').getByRole('button').filter({ has: page.locator('svg.lucide-log-out') }).click();
    await expect(page.getByRole('heading', { name: 'TMS Platform' })).toBeVisible();

    // Sign in
    await page.getByRole('textbox', { name: 'Email' }).fill(user.email);
    await page.getByRole('textbox', { name: 'Password' }).fill(user.password);
    await page.getByRole('button', { name: 'Sign In' }).last().click();

    await expect(page.getByText('Client Dashboard')).toBeVisible();
    await expect(page.getByText(user.name)).toBeVisible();
  });

  test('should create a new order', async ({ page }) => {
    const user = createTestUser();
    await signUp(page, user);

    await page.getByRole('button', { name: 'New Order' }).click();
    await expect(page.getByRole('heading', { name: 'New Order' })).toBeVisible();

    // Step 1: Sender
    const sender = sectionByHeading(page, 'Sender Information');
    await sender.getByRole('textbox', { name: 'Company or person name' }).fill('ABC Logistics Sp. z o.o.');
    await sender.getByRole('textbox', { name: 'Street address' }).fill('ul. Transportowa 15');
    await sender.getByRole('textbox', { name: 'City' }).fill('Warszawa');
    await sender.getByPlaceholder('12-345').fill('00-123');

    // Step 1: Receiver
    const receiver = sectionByHeading(page, 'Receiver Information');
    await receiver.getByRole('textbox', { name: 'Company or person name' }).fill('XYZ Trading GmbH');
    await receiver.getByRole('textbox', { name: 'Street address' }).fill('Hauptstrasse 42');
    await receiver.getByRole('textbox', { name: 'City' }).fill('Berlin');
    await receiver.getByPlaceholder('12-345').fill('10115');

    // Step 1: Dates
    const pickupDate = futureDate(5);
    const deliveryDate = futureDate(7);
    const dates = sectionByHeading(page, 'Dates');
    await dates.locator('div:has(> label)', { hasText: 'Pickup Date' }).locator('input[type="date"]').fill(pickupDate);
    await dates.locator('div:has(> label)', { hasText: 'Delivery Date' }).locator('input[type="date"]').fill(deliveryDate);

    // Step 2: Cargo
    await page.getByRole('button', { name: 'Next: Cargo Specification' }).click();
    await page.getByRole('combobox').selectOption('General Cargo');
    await page.getByPlaceholder('0.00').fill('2500');
    await page.getByPlaceholder('1').fill('10');
    await page.getByRole('textbox', { name: 'Provide additional details' }).fill('Electronic components - handle with care');

    // Step 3: Summary
    await page.getByRole('button', { name: 'Next: Summary' }).click();
    await expect(page.getByText('ABC Logistics Sp. z o.o.')).toBeVisible();
    await expect(page.getByText('XYZ Trading GmbH')).toBeVisible();
    await expect(page.getByText('2500 kg')).toBeVisible();
    await expect(page.getByText('10 pcs')).toBeVisible();

    // Submit
    await page.getByRole('button', { name: 'Submit Order' }).click();
    await expect(page.getByText('Pending')).toBeVisible();
    await expect(page.getByRole('button', { name: 'View Details' })).toBeVisible();
  });
});
