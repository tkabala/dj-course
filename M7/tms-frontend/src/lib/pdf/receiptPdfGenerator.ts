import { PdfDocumentBuilder } from './core/pdfDocumentBuilder'

export interface PaymentReceiptData {
  id: string | number
  amount: string | number
  status: string
  method: string
  invoice?: string
  date: string
}

export async function generateReceiptPDF(payment: PaymentReceiptData): Promise<void> {
  const builder = await PdfDocumentBuilder.create({
    title: 'Payment Receipt',
    subtitle: 'Deliveroo Logistics'
  })

  builder.sectionTitle('Payment Details')
  builder.field('Payment ID:', String(payment.id))
  builder.field('Amount:', String(payment.amount))
  builder.field('Status:', payment.status)
  builder.field('Method:', payment.method)
  builder.field('Invoice:', payment.invoice ?? '-')
  builder.field('Date:', payment.date)

  builder.save(`Receipt_${payment.id}.pdf`)
}
