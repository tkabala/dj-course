import { PdfDocumentBuilder } from './core/pdfDocumentBuilder'
import { formatCurrency, formatDate } from './core/pdfFormatters'

export interface InvoiceData {
  id: string
  number: string
  description: string
  date: Date
  amount: number
  status: 'Paid' | 'Unpaid' | 'Overdue'
  dueDate: Date
}

export async function generateInvoicePDF(invoice: InvoiceData): Promise<void> {
  const builder = await PdfDocumentBuilder.create({
    title: 'Invoice',
    subtitle: 'Deliveroo Logistics'
  })

  builder.sectionTitle('Invoice Details')
  builder.field('Invoice Number:', invoice.number)
  builder.field('Invoice ID:', String(invoice.id))
  builder.multiLineField('Description:', invoice.description)
  builder.field('Amount:', formatCurrency(invoice.amount, 'USD'))
  builder.field('Status:', invoice.status)
  builder.field('Invoice Date:', formatDate(invoice.date))
  builder.field('Due Date:', formatDate(invoice.dueDate))

  builder.save(`Invoice_${invoice.number}.pdf`)
}
