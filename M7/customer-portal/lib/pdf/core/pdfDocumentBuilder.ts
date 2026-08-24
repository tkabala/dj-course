import jsPDF from 'jspdf'
import { PDF_LAYOUT } from './pdfLayout'
import type { PdfSection } from './pdfSection'

export interface PdfDocumentOptions {
  title: string
  subtitle?: string
  logoPath?: string
  includeFooter?: boolean
  footerLines?: string[]
}

export interface FieldOptions {
  labelWidth?: number
  valueX?: number
  maxWidth?: number
  boldLabel?: boolean
}

export interface TableOptions {
  startX?: number
  columnWidths?: number[]
  headerHeight?: number
  rowHeight?: number
  maxWidth?: number
}

export interface TableHeader {
  label: string
  width?: number
  align?: 'left' | 'right' | 'center'
}

export interface TableCell {
  text: string
  align?: 'left' | 'right' | 'center'
  maxWidth?: number
}

export interface TableRow {
  cells: (string | TableCell)[]
  height?: number
}

/**
 * High-level builder around jsPDF that removes duplication from every generator.
 * It tracks the current Y position, handles page breaks, draws headers/footers,
 * and exposes helpers for fields, sections and tables.
 */
export class PdfDocumentBuilder {
  private readonly doc: jsPDF
  private yPos: number
  private readonly logoDataUrl: string | null
  private readonly options: PdfDocumentOptions & {
    logoPath: string
    includeFooter: boolean
    footerLines: string[]
  }

  private constructor(doc: jsPDF, logoDataUrl: string | null, options: PdfDocumentOptions) {
    this.doc = doc
    this.logoDataUrl = logoDataUrl
    this.options = {
      logoPath: PDF_LAYOUT.logo.path,
      includeFooter: true,
      footerLines: [
        'Deliveroo Logistics | ul. Logistyczna 123, 00-001 Warsaw, Poland',
        'Phone: +48 123 456 789 | Email: contact@deliveroo.pl'
      ],
      ...options
    }
    this.yPos = PDF_LAYOUT.page.marginTop
    this.drawHeader()
  }

  static async create(options: PdfDocumentOptions): Promise<PdfDocumentBuilder> {
    const doc = new jsPDF()
    const logoDataUrl = await loadLogo(options.logoPath ?? PDF_LAYOUT.logo.path)
    return new PdfDocumentBuilder(doc, logoDataUrl, options)
  }

  get internalDoc(): jsPDF {
    return this.doc
  }

  get pageWidth(): number {
    return this.doc.internal.pageSize.width
  }

  get pageHeight(): number {
    return this.doc.internal.pageSize.height
  }

  get contentWidth(): number {
    return PDF_LAYOUT.page.contentWidth(this.doc)
  }

  get currentY(): number {
    return this.yPos
  }

  set currentY(value: number) {
    this.yPos = value
  }

  /**
   * Make sure at least `height` units of vertical space are available on the current page.
   * If not, a new page is added (header is NOT redrawn automatically; footer is applied later).
   */
  ensureSpace(height: number): void {
    const available = this.pageHeight - PDF_LAYOUT.page.footerHeight - this.yPos
    if (available < height) {
      this.doc.addPage()
      this.yPos = PDF_LAYOUT.page.marginLeft
    }
  }

  addPageIfNeeded(height: number): void {
    this.ensureSpace(height)
  }

  addVerticalSpace(height = PDF_LAYOUT.spacing.fieldGap): void {
    this.yPos += height
  }

  sectionTitle(title: string): void {
    this.ensureSpace(PDF_LAYOUT.spacing.sectionTitleGap)
    this.setFont(PDF_LAYOUT.fonts.sectionTitle)
    this.doc.setFillColor(...PDF_LAYOUT.colors.sectionBackground)
    this.doc.rect(
      PDF_LAYOUT.page.marginLeft,
      this.yPos - PDF_LAYOUT.spacing.sectionTitleHeight + 1,
      this.contentWidth,
      PDF_LAYOUT.spacing.sectionTitleHeight,
      'F'
    )
    this.doc.text(title, PDF_LAYOUT.page.marginLeft + 2, this.yPos)
    this.yPos += PDF_LAYOUT.spacing.sectionTitleGap
  }

  field(label: string, value: string, options: FieldOptions = {}): void {
    const labelWidth = options.labelWidth ?? 60
    const valueX = options.valueX ?? PDF_LAYOUT.page.marginLeft + labelWidth
    const maxWidth = options.maxWidth ?? this.contentWidth - labelWidth - 2

    this.ensureSpace(PDF_LAYOUT.spacing.fieldGap)
    this.setFont(PDF_LAYOUT.fonts.bodyBold)
    this.doc.text(label, PDF_LAYOUT.page.marginLeft, this.yPos)
    this.setFont(PDF_LAYOUT.fonts.body)
    const lines = this.doc.splitTextToSize(value, maxWidth)
    this.doc.text(lines, valueX, this.yPos)
    this.yPos += Math.max(lines.length * PDF_LAYOUT.spacing.paragraphGap, PDF_LAYOUT.spacing.fieldGap)
  }

  multiLineField(label: string, value: string, options: FieldOptions = {}): void {
    const maxWidth = options.maxWidth ?? this.contentWidth - 4

    this.ensureSpace(PDF_LAYOUT.spacing.fieldGap)
    this.setFont(PDF_LAYOUT.fonts.bodyBold)
    this.doc.text(label, PDF_LAYOUT.page.marginLeft, this.yPos)
    this.setFont(PDF_LAYOUT.fonts.body)
    const lines = this.doc.splitTextToSize(value, maxWidth)
    this.doc.text(lines, PDF_LAYOUT.page.marginLeft + 2, this.yPos + PDF_LAYOUT.spacing.paragraphGap)
    this.yPos += lines.length * PDF_LAYOUT.spacing.paragraphGap + PDF_LAYOUT.spacing.paragraphGap + PDF_LAYOUT.spacing.fieldGap
  }

  paragraph(value: string, maxWidth?: number): void {
    const width = maxWidth ?? this.contentWidth
    this.ensureSpace(PDF_LAYOUT.spacing.fieldGap)
    this.setFont(PDF_LAYOUT.fonts.body)
    const lines = this.doc.splitTextToSize(value, width)
    this.doc.text(lines, PDF_LAYOUT.page.marginLeft, this.yPos)
    this.yPos += lines.length * PDF_LAYOUT.spacing.paragraphGap + PDF_LAYOUT.spacing.fieldGap
  }

  text(value: string, x: number, y: number, options?: { size?: number; bold?: boolean; align?: 'left' | 'right' | 'center' }): void {
    this.doc.setFontSize(options?.size ?? PDF_LAYOUT.fonts.body.size)
    this.doc.setFont(PDF_LAYOUT.fonts.body.font, options?.bold ? 'bold' : PDF_LAYOUT.fonts.body.style)
    this.doc.text(value, x, y, { align: options?.align })
  }

  table(headers: TableHeader[], rows: TableRow[], options: TableOptions = {}): void {
    const startX = options.startX ?? PDF_LAYOUT.page.marginLeft
    const rowHeight = options.rowHeight ?? PDF_LAYOUT.spacing.tableRowHeight
    const headerHeight = options.headerHeight ?? PDF_LAYOUT.spacing.tableHeaderHeight
    const columnWidths = options.columnWidths ?? this.computeColumnWidths(headers, startX)

    this.ensureSpace(headerHeight + rowHeight)

    // Header background
    this.doc.setFillColor(...PDF_LAYOUT.colors.tableBackground)
    this.doc.rect(startX, this.yPos - headerHeight + 2, this.contentWidth, headerHeight, 'F')

    this.setFont(PDF_LAYOUT.fonts.bodyBold)
    let x = startX + 2
    headers.forEach((header, index) => {
      this.doc.text(header.label, x, this.yPos)
      x += columnWidths[index]
    })
    this.yPos += headerHeight + 2

    this.doc.setDrawColor(...PDF_LAYOUT.colors.line)
    this.doc.line(startX, this.yPos - 1, startX + this.contentWidth, this.yPos - 1)

    this.setFont(PDF_LAYOUT.fonts.body)
    rows.forEach((row) => {
      const rowH = row.height ?? rowHeight
      this.ensureSpace(rowH)

      x = startX + 2
      row.cells.forEach((cell, index) => {
        const cellData = typeof cell === 'string' ? { text: cell } : cell
        const width = columnWidths[index] - 4
        const lines = this.doc.splitTextToSize(cellData.text, width)
        this.doc.text(lines, x, this.yPos, { align: cellData.align })
        x += columnWidths[index]
      })

      this.yPos += rowH
      this.doc.setDrawColor(...PDF_LAYOUT.colors.line)
      this.doc.line(startX, this.yPos - 1, startX + this.contentWidth, this.yPos - 1)
    })

    this.yPos += PDF_LAYOUT.spacing.fieldGap
  }

  renderSection<TData>(section: PdfSection<TData>, data: TData): void {
    section.render(this, data)
  }

  save(filename: string): void {
    this.applyFooter()
    this.doc.save(filename)
  }

  toBlob(): Blob {
    this.applyFooter()
    return this.doc.output('blob')
  }

  private drawHeader(): void {
    if (this.logoDataUrl) {
      this.doc.addImage(
        this.logoDataUrl,
        'PNG',
        PDF_LAYOUT.logo.x,
        PDF_LAYOUT.logo.y,
        PDF_LAYOUT.logo.width,
        PDF_LAYOUT.logo.height
      )
    }

    this.setFont(PDF_LAYOUT.fonts.header)
    this.doc.text(this.options.title, PDF_LAYOUT.page.marginLeft, 35)

    if (this.options.subtitle) {
      this.setFont(PDF_LAYOUT.fonts.subHeader)
      this.doc.text(this.options.subtitle, PDF_LAYOUT.page.marginLeft, 42)
    }
  }

  private applyFooter(): void {
    if (!this.options.includeFooter) return

    const pageCount = this.doc.getNumberOfPages()
    this.doc.setDrawColor(...PDF_LAYOUT.colors.line)
    this.doc.setFontSize(PDF_LAYOUT.fonts.footer.size)
    this.doc.setTextColor(...PDF_LAYOUT.colors.footerText)

    for (let i = 1; i <= pageCount; i++) {
      this.doc.setPage(i)
      this.doc.line(
        PDF_LAYOUT.page.marginLeft,
        this.pageHeight - 25,
        this.pageWidth - PDF_LAYOUT.page.marginRight,
        this.pageHeight - 25
      )

      this.options.footerLines.forEach((line, idx) => {
        this.doc.text(line, PDF_LAYOUT.page.marginLeft, this.pageHeight - 18 + idx * 6)
      })

      this.doc.text(`Page ${i} of ${pageCount}`, this.pageWidth - 30, this.pageHeight - 12)
    }

    this.doc.setTextColor(...PDF_LAYOUT.colors.text)
  }

  private setFont(config: { size: number; font: string; style: string }): void {
    this.doc.setFontSize(config.size)
    this.doc.setFont(config.font, config.style)
  }

  private computeColumnWidths(headers: TableHeader[], startX: number): number[] {
    const definedWidths = headers.map((h) => h.width ?? 0)
    const totalDefined = definedWidths.reduce((sum, w) => sum + w, 0)
    const remaining = this.contentWidth - (totalDefined - 0)
    const autoCount = definedWidths.filter((w) => w === 0).length || 1
    const autoWidth = remaining / autoCount

    return definedWidths.map((w) => (w === 0 ? autoWidth : w))
  }
}

async function loadLogo(path: string): Promise<string | null> {
  try {
    const response = await fetch(path)
    const blob = await response.blob()
    return await new Promise((resolve) => {
      const reader = new FileReader()
      reader.onloadend = () => resolve(reader.result as string)
      reader.readAsDataURL(blob)
    })
  } catch (err) {
    console.error('Failed to load PDF logo', err)
    return null
  }
}
