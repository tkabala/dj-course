import type { PdfDocumentBuilder } from './pdfDocumentBuilder'

/**
 * A section is a reusable piece of a PDF document (e.g. header, footer, request info, cargo info).
 * Implementations receive the builder and the data they need and render themselves.
 */
export interface PdfSection<TData = unknown> {
  render(builder: PdfDocumentBuilder, data: TData): void
}

/**
 * Convenience helper to create a section from a plain render function.
 */
export function createSection<TData>(render: (builder: PdfDocumentBuilder, data: TData) => void): PdfSection<TData> {
  return { render }
}
