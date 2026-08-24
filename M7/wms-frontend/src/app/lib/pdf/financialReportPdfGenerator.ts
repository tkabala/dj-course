import { PdfDocumentBuilder } from './core/pdfDocumentBuilder'
import { formatCurrency, formatDate, formatPercentage } from './core/pdfFormatters'
import type { BillingOverview, Invoice } from '../../features/billing-payments/billing.model'

export interface FinancialReportData {
  overview: BillingOverview
  invoices: Invoice[]
  reportPeriod?: string
}

export async function generateFinancialReportPDF(data: FinancialReportData): Promise<void> {
  const builder = await PdfDocumentBuilder.create({
    title: 'Financial Report',
    subtitle: 'Deliveroo Logistics'
  })

  const reportPeriod =
    data.reportPeriod ?? `As of ${formatDate(new Date())}`

  builder.sectionTitle('Report Period')
  builder.field('Period:', reportPeriod)

  builder.sectionTitle('Revenue Summary')
  builder.field('Total Revenue:', formatCurrency(data.overview.totalRevenue, 'USD'))
  builder.field('Total Invoices:', String(data.overview.totalInvoices))
  builder.field('Paid Invoices:', String(data.overview.paidInvoices))
  builder.field('Overdue Invoices:', String(data.overview.overdueInvoices))
  builder.field('Average Invoice Value:', formatCurrency(data.overview.avgInvoiceValue, 'USD'))

  builder.sectionTitle('Performance Metrics')
  builder.field('Average Payment Time:', `${data.overview.avgPaymentTime} days`)
  builder.field('Collection Rate:', formatPercentage(data.overview.collectionRate))

  const paidRevenue = data.invoices
    .filter((inv) => inv.status === 'paid')
    .reduce((sum, inv) => sum + inv.amount, 0)
  const overdueRevenue = data.invoices
    .filter((inv) => inv.status === 'overdue')
    .reduce((sum, inv) => sum + inv.amount, 0)
  const pendingRevenue = data.invoices
    .filter((inv) => inv.status === 'sent')
    .reduce((sum, inv) => sum + inv.amount, 0)
  const draftRevenue = data.invoices
    .filter((inv) => inv.status === 'draft')
    .reduce((sum, inv) => sum + inv.amount, 0)

  builder.field('Paid Revenue:', formatCurrency(paidRevenue, 'USD'))
  builder.field('Overdue Revenue:', formatCurrency(overdueRevenue, 'USD'))
  builder.field('Pending Revenue:', formatCurrency(pendingRevenue, 'USD'))

  builder.sectionTitle('Invoice Status Breakdown')
  builder.table(
    [
      { label: 'Status', width: 80 },
      { label: 'Count', width: 40 },
      { label: 'Revenue', width: 50 }
    ],
    [
      {
        cells: [
          'Paid',
          String(data.invoices.filter((inv) => inv.status === 'paid').length),
          { text: formatCurrency(paidRevenue, 'USD'), align: 'right' }
        ]
      },
      {
        cells: [
          'Sent',
          String(data.invoices.filter((inv) => inv.status === 'sent').length),
          { text: formatCurrency(pendingRevenue, 'USD'), align: 'right' }
        ]
      },
      {
        cells: [
          'Overdue',
          String(data.invoices.filter((inv) => inv.status === 'overdue').length),
          { text: formatCurrency(overdueRevenue, 'USD'), align: 'right' }
        ]
      },
      {
        cells: [
          'Draft',
          String(data.invoices.filter((inv) => inv.status === 'draft').length),
          { text: formatCurrency(draftRevenue, 'USD'), align: 'right' }
        ]
      }
    ]
  )

  const contractorRevenue = new Map<string, { name: string; total: number; count: number }>()
  data.invoices.forEach((invoice) => {
    const existing = contractorRevenue.get(invoice.contractorId)
    if (existing) {
      existing.total += invoice.amount
      existing.count += 1
    } else {
      contractorRevenue.set(invoice.contractorId, {
        name: invoice.contractorName,
        total: invoice.amount,
        count: 1
      })
    }
  })

  const topContractors = Array.from(contractorRevenue.values())
    .sort((a, b) => b.total - a.total)
    .slice(0, 5)

  if (topContractors.length > 0) {
    builder.sectionTitle('Top 5 Contractors by Revenue')
    builder.table(
      [
        { label: 'Contractor', width: 90 },
        { label: 'Invoices', width: 40 },
        { label: 'Total Revenue', width: 40 }
      ],
      topContractors.map((contractor) => ({
        cells: [
          contractor.name,
          String(contractor.count),
          { text: formatCurrency(contractor.total, 'USD'), align: 'right' }
        ]
      }))
    )
  }

  const recentInvoices = [...data.invoices]
    .sort((a, b) => new Date(b.issueDate).getTime() - new Date(a.issueDate).getTime())
    .slice(0, 10)

  if (recentInvoices.length > 0) {
    builder.sectionTitle('Recent Invoices (Last 10)')
    builder.table(
      [
        { label: 'Invoice #', width: 35 },
        { label: 'Contractor', width: 55 },
        { label: 'Date', width: 35 },
        { label: 'Status', width: 30 },
        { label: 'Amount', width: 35 }
      ],
      recentInvoices.map((invoice) => ({
        cells: [
          invoice.invoiceNumber,
          invoice.contractorName,
          formatDate(invoice.issueDate),
          invoice.status.toUpperCase(),
          { text: formatCurrency(invoice.amount, 'USD'), align: 'right' }
        ]
      }))
    )
  }

  builder.sectionTitle('Report Summary')
  builder.field('Report Generated:', formatDate(new Date()))
  builder.field('Total Accounts:', String(contractorRevenue.size))
  builder.multiLineField(
    'Notes:',
    'This financial report provides a comprehensive overview of billing and payment activities. ' +
      'For detailed invoice information, please refer to individual invoice documents.'
  )

  const saveDate = formatDate(new Date()).replace(/\s+/g, '_')
  builder.save(`Financial_Report_${saveDate}.pdf`)
}
