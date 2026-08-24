import type { PdfDocumentBuilder } from './pdfDocumentBuilder'

export interface PdfSection<TData = unknown> {
  render(builder: PdfDocumentBuilder, data: TData): void
}

export function createSection<TData>(render: (builder: PdfDocumentBuilder, data: TData) => void): PdfSection<TData> {
  return { render }
}
