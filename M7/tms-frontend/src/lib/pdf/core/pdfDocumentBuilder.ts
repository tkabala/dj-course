import jsPDF from 'jspdf'
import { PDF_LAYOUT } from './pdfLayout'
import type { PdfSection } from './pdfSection'

export interface PdfDocumentOptions {
  title: string
  subtitle?: string
  logoPath?: string
  includeFooter?: boolean
  footerLines?: string[]
  drawHeader?: (builder: PdfDocumentBuilder) => void
  drawFooter?: (builder: PdfDocumentBuilder) => void
  includeWatermark?: boolean
  watermarkText?: string
}

export interface FieldOptions {
  labelWidth?: number
  valueX?: number
  maxWidth?: number
}

export interface TableHeader {
  label: string
  width?: number
  align?: 'left' | 'right' | 'center'
}

export interface TableRow {
  cells: (string | { text: string; align?: 'left' | 'right' | 'center'; maxWidth?: number })[]
  height?: number
}

export interface TableOptions {
  startX?: number
  columnWidths?: number[]
  headerHeight?: number
  rowHeight?: number
}

export class PdfDocumentBuilder {
  private readonly doc: jsPDF
  private yPos: number
  private readonly logoDataUrlInternal: string | null
  private readonly options: Required<PdfDocumentOptions>

  private constructor(doc: jsPDF, logoDataUrl: string | null, options: PdfDocumentOptions) {
    this.doc = doc
    this.logoDataUrlInternal = logoDataUrl
    this.options = {
      logoPath: PDF_LAYOUT.logo.path,
      includeFooter: true,
      footerLines: [],
      watermarkText: 'SAMPLE DOCUMENT',
      ...options
    }
    this.yPos = PDF_LAYOUT.page.marginTop
    this.drawWatermark()
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

  get logoDataUrl(): string | null {
    return this.logoDataUrlInternal
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
    this.doc.text(title, PDF_LAYOUT.page.marginLeft, this.yPos)
    this.yPos += PDF_LAYOUT.spacing.sectionTitleGap
  }

  field(label: string, value: string, options: FieldOptions = {}): void {
    const labelWidth = options.labelWidth ?? 40
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

  table(headers: TableHeader[], rows: TableRow[], options: TableOptions = {}): void {
    const startX = options.startX ?? PDF_LAYOUT.page.marginLeft
    const rowHeight = options.rowHeight ?? PDF_LAYOUT.spacing.tableRowHeight
    const headerHeight = options.headerHeight ?? PDF_LAYOUT.spacing.tableHeaderHeight
    const columnWidths = options.columnWidths ?? this.computeColumnWidths(headers, startX)

    this.ensureSpace(headerHeight + rowHeight)

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
    if (this.options.drawHeader) {
      this.options.drawHeader(this)
      return
    }

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

  private drawWatermark(): void {
    if (!this.options.includeWatermark) return

    this.doc.setFontSize(50)
    this.doc.setTextColor(200, 200, 200)
    this.doc.setFont('helvetica', 'bold')
    this.doc.text(this.options.watermarkText, this.pageWidth / 2, this.pageHeight / 2, {
      angle: 45,
      align: 'center'
    })
    this.doc.setTextColor(...PDF_LAYOUT.colors.text)
  }

  private applyFooter(): void {
    if (this.options.drawFooter) {
      const pageCount = this.doc.getNumberOfPages()
      for (let i = 1; i <= pageCount; i++) {
        this.doc.setPage(i)
        this.options.drawFooter(this)
      }
      return
    }

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

  private computeColumnWidths(headers: TableHeader[], _startX: number): number[] {
    const definedWidths = headers.map((h) => h.width ?? 0)
    const totalDefined = definedWidths.reduce((sum, w) => sum + w, 0)
    const remaining = this.contentWidth - totalDefined
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
