import type { Page } from '@playwright/test'
import { waitForHydration } from './hydration'

/** Where globalSetup writes the signed-in storage state consumed by `use.storageState`. */
export const STORAGE_STATE = 'tests/e2e/.auth/state.json'

/**
 * The single signed-in user every test runs as. Mirrors the shape `stores/auth.ts`
 * persists on a successful sign-in, so `initializeAuth()` restores it verbatim.
 */
export const authenticatedUser = {
  id: '1',
  email: 'john.doe@example.com',
  firstName: 'John',
  lastName: 'Doe',
  role: 'COMPANY_ADMIN',
  permissions: ['CREATE_REQUEST', 'VIEW_REQUEST', 'EDIT_REQUEST', 'MANAGE_TEAM'],
  companyId: '1',
  isActive: true,
}

export const authenticatedCompany = {
  id: '1',
  name: 'Example Logistics Ltd.',
}

/**
 * Seeds the localStorage keys the app writes on a successful sign-in (see
 * stores/auth.ts#login and plugins/auth.client.ts) instead of driving the login
 * form. Auth here is a client-only mock (any non-empty email/password is
 * accepted, see features/auth/signin/signin-api.ts) with no server session to
 * align with, so state injection is a faithful, deterministic stand-in for
 * "the user is logged in" - and it keeps every feature test from depending on
 * the sign-in form's own timing.
 *
 * Called once from globalSetup; the resulting state is reused by every test
 * through `use.storageState`, so no test pays for a login round-trip.
 */
export async function seedAuthState(page: Page, baseURL: string): Promise<void> {
  await page.goto(`${baseURL}/login`)
  // Let /login settle before writing to its origin and navigating away: while
  // the dev server is still compiling this route it issues a full reload, which
  // would abort the next goto.
  await waitForHydration(page)
  await page.evaluate(
    ({ user, company }) => {
      localStorage.setItem('auth_user', JSON.stringify(user))
      localStorage.setItem('auth_company', JSON.stringify(company))
      localStorage.setItem('auth_isAuthenticated', 'true')
    },
    { user: authenticatedUser, company: authenticatedCompany },
  )
}
