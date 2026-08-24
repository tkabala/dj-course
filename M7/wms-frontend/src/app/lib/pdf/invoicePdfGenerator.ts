import { PdfDocumentBuilder } from './core/pdfDocumentBuilder'
import { formatCurrency, formatDate } from './core/pdfFormatters'
import type { Invoice } from '../../features/billing-payments/billing.model'

export interface InvoiceData extends Invoice {
  companyInfo?: {
    name: string
    address: string
    city: string
    phone: string
    email: string
  }
  contractorInfo?: {
    address: string
    city: string
    email: string
  }
  taxRate?: number
  paymentTerms?: string
  notes?: string
}

export async function generateInvoicePDF(invoiceData: InvoiceData): Promise<void> {
  const builder = await PdfDocumentBuilder.create({
    title: `Invoice - ${invoiceData.invoiceNumber}`,
    subtitle: 'Deliveroo Logistics'
  })

  const companyInfo = invoiceData.companyInfo ?? {
    name: 'Warehouse Management System',
    address: '123 Industrial Blvd',
    city: 'Chicago, IL 60601',
    phone: '+1-555-0100',
    email: 'billing@wms.com'
  }

  const contractorInfo = invoiceData.contractorInfo ?? {
    address: '123 Business Ave',
    city: 'Business City, BC 12345',
    email: `contact@${invoiceData.contractorName.toLowerCase().replace(/\s+/g, '')}.com`
  }

  builder.sectionTitle('Invoice Information')
  builder.field('Invoice Number:', invoiceData.invoiceNumber)
  builder.field('Status:', invoiceData.status.toUpperCase())
  builder.field('Issue Date:', formatDate(invoiceData.issueDate))
  builder.field('Due Date:', formatDate(invoiceData.dueDate))

  builder.sectionTitle('From')
  builder.field('Company:', companyInfo.name)
  builder.field('Address:', companyInfo.address)
  builder.field('City:', companyInfo.city)
  builder.field('Phone:', companyInfo.phone)
  builder.field('Email:', companyInfo.email)

  builder.sectionTitle('Bill To')
  builder.field('Contractor:', invoiceData.contractorName)
  builder.field('Contractor ID:', invoiceData.contractorId)
  builder.field('Address:', contractorInfo.address)
  builder.field('City:', contractorInfo.city)
  builder.field('Email:', contractorInfo.email)

  builder.sectionTitle('Invoice Items')
  builder.table(
    [
      { label: 'Description', width: 110 },
      { label: 'Qty', width: 25 },
      { label: 'Unit Price', width: 30 },
      { label: 'Total', width: 25 }
    ],
    invoiceData.items.map((item) => ({
      cells: [
        { text: item.description, maxWidth: 106 },
        { text: String(item.quantity), align: 'right' },
        { text: formatCurrency(item.unitPrice, 'USD'), align: 'right' },
        { text: formatCurrency(item.totalPrice, 'USD'), align: 'right' }
      ],
      height: 8
    }))
  )

  const subtotal = invoiceData.items.reduce((sum, item) => sum + item.totalPrice, 0)
  const taxRate = invoiceData.taxRate ?? 0.085
  const tax = subtotal * taxRate
  const total = subtotal + tax

  builder.sectionTitle('Summary')
  builder.field('Subtotal:', formatCurrency(subtotal, 'USD'))
  builder.field('Tax:', `${formatCurrency(tax, 'USD')} (${(taxRate * 100).toFixed(1)}%)`)
  builder.field('Total Amount:', formatCurrency(total, 'USD'))

  builder.sectionTitle('Payment Information')
  builder.field('Payment Terms:', invoiceData.paymentTerms ?? 'Net 30 days')
  builder.multiLineField('Payment Methods:', 'Bank Transfer: Account #123-456-789 or Check: Payable to "WMS Inc."')

  builder.sectionTitle('Notes')
  builder.paragraph(
    invoiceData.notes ??
      'Thank you for your business! Please remit payment within 30 days of the invoice date. ' +
        'For any questions regarding this invoice, please contact our billing department at billing@wms.com.'
  )

  builder.save(`Invoice_${invoiceData.invoiceNumber}.pdf`)
}
