import { PdfDocumentBuilder } from './core/pdfDocumentBuilder'
import { formatCurrency, formatDate, formatPercentage } from './core/pdfFormatters'

export interface MetricsData {
  totalShipments: number
  onTimeDelivery: number
  totalCost: number
  storageVolume: number
}

export interface RoutePerformanceData {
  route: string
  shipments: number
  onTimePercentage: number
  avgCost: number
  totalRevenue: number
}

export interface ReportsData {
  dateRange: {
    from: string
    to: string
  }
  metrics: MetricsData
  routePerformance: RoutePerformanceData[]
}

export async function generateReportsPDF(reportsData: ReportsData): Promise<void> {
  const builder = await PdfDocumentBuilder.create({
    title: 'Logistics Report',
    subtitle: 'Deliveroo Logistics'
  })

  builder.sectionTitle('Report Period')
  builder.field('Period:', `${formatDate(reportsData.dateRange.from)} - ${formatDate(reportsData.dateRange.to)}`)

  builder.sectionTitle('Key Metrics')
  builder.field('Total Shipments:', String(reportsData.metrics.totalShipments))
  builder.field('On-Time Delivery:', formatPercentage(reportsData.metrics.onTimeDelivery))
  builder.field('Total Cost:', formatCurrency(reportsData.metrics.totalCost, 'EUR'))
  builder.field('Storage Volume:', `${reportsData.metrics.storageVolume.toLocaleString()} m³`)

  builder.sectionTitle('Route Performance')
  builder.table(
    [
      { label: 'Route', width: 60 },
      { label: 'Shipments', width: 30 },
      { label: 'On-Time %', width: 30 },
      { label: 'Avg Cost', width: 30 },
      { label: 'Revenue', width: 30 }
    ],
    reportsData.routePerformance.map((route) => ({
      cells: [
        route.route,
        String(route.shipments),
        formatPercentage(route.onTimePercentage, 0),
        formatCurrency(route.avgCost, 'EUR'),
        formatCurrency(route.totalRevenue, 'EUR')
      ]
    }))
  )

  const fromDateStr = reportsData.dateRange.from.replace(/-/g, '')
  const toDateStr = reportsData.dateRange.to.replace(/-/g, '')
  builder.save(`Logistics_Report_${fromDateStr}_${toDateStr}.pdf`)
}
