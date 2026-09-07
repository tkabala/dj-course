import type { Page } from '@playwright/test'

/**
 * Blocks until Vue has taken over the server-rendered markup.
 *
 * Interacting before that point is a silent no-op: the click or fill lands on
 * the DOM, but v-model never sees it because the listeners are not attached
 * yet - which shows up as "the field is filled, yet Next stays disabled".
 * `load`/`domcontentloaded` are both too early, and `networkidle` is a timing
 * heuristic that Playwright explicitly discourages (with a Vite dev server and
 * its HMR connection it is also needlessly slow). Nuxt exposes the actual
 * signal, so use it.
 */
export async function waitForHydration(page: Page): Promise<void> {
  await page.waitForFunction(() => {
    const nuxt = (window as unknown as { useNuxtApp?: () => { isHydrating: boolean } }).useNuxtApp
    return typeof nuxt === 'function' && nuxt().isHydrating === false
  })
}
