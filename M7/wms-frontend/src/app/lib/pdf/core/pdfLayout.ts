export const PDF_LAYOUT = {
  page: {
    marginLeft: 20,
    marginRight: 20,
    marginTop: 55,
    footerHeight: 30,
    contentWidth(doc: { internal: { pageSize: { width: number } } }): number {
      return doc.internal.pageSize.width - PDF_LAYOUT.page.marginLeft - PDF_LAYOUT.page.marginRight
    }
  },
  logo: {
    path: '/assets/deliveroo-pdf-logo.png',
    x: 15,
    y: 15,
    width: 15,
    height: 15
  },
  fonts: {
    header: { size: 16, font: 'helvetica', style: 'bold' },
    subHeader: { size: 10, font: 'helvetica', style: 'normal' },
    sectionTitle: { size: 14, font: 'helvetica', style: 'bold' },
    bodyBold: { size: 10, font: 'helvetica', style: 'bold' },
    body: { size: 10, font: 'helvetica', style: 'normal' },
    small: { size: 9, font: 'helvetica', style: 'normal' },
    footer: { size: 8, font: 'helvetica', style: 'normal' }
  } as const,
  colors: {
    sectionBackground: [248, 250, 252] as [number, number, number],
    tableBackground: [240, 240, 240] as [number, number, number],
    line: [200, 200, 200] as [number, number, number],
    footerText: [100, 100, 100] as [number, number, number],
    text: [0, 0, 0] as [number, number, number]
  },
  spacing: {
    sectionTitleHeight: 8,
    sectionTitleGap: 15,
    fieldGap: 8,
    paragraphGap: 6,
    blockGap: 15,
    tableRowHeight: 7,
    tableHeaderHeight: 6
  }
} as const

export type PdfLayout = typeof PDF_LAYOUT
