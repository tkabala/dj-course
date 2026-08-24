import { PdfDocumentBuilder } from './core/pdfDocumentBuilder'
import { formatCurrency, formatDate, formatDateTime } from './core/pdfFormatters'
import type { InventoryItem } from '../../features/inventory/inventory.model'
import type { CargoEvent, CargoLocationHistory, CargoDocument } from '../../features/cargo-management/cargo.model'

export interface CargoReportData extends InventoryItem {
  events?: CargoEvent[]
  locationHistory?: CargoLocationHistory[]
  documents?: CargoDocument[]
}

export async function generateCargoReportPDF(cargoData: CargoReportData): Promise<void> {
  const builder = await PdfDocumentBuilder.create({
    title: `Cargo Report - ${cargoData.sku}`,
    subtitle: 'Deliveroo Logistics'
  })

  builder.sectionTitle('Basic Information')
  builder.field('SKU:', cargoData.sku)
  builder.field('Name:', cargoData.name)
  builder.multiLineField('Description:', cargoData.description)
  builder.field('Category:', cargoData.category)
  builder.field('Status:', cargoData.status.toUpperCase())

  builder.sectionTitle('Quantity & Storage')
  builder.field('Quantity:', `${cargoData.quantity} ${cargoData.unit}`)
  builder.field('Location:', cargoData.location)
  builder.field('Zone:', `${cargoData.zoneName} (Zone ID: ${cargoData.zoneId})`)
  builder.field('Shelf Location:', `${cargoData.shelfLocation} (Shelf ID: ${cargoData.shelfId})`)

  builder.sectionTitle('Physical Attributes')
  builder.field('Weight:', `${cargoData.weight} kg`)
  builder.field('Volume:', `${cargoData.volume} m³`)
  builder.field('Value:', formatCurrency(cargoData.value, cargoData.currency))

  builder.sectionTitle('Additional Details')
  if (cargoData.batchNumber) {
    builder.field('Batch Number:', cargoData.batchNumber)
  }
  if (cargoData.serialNumber) {
    builder.field('Serial Number:', cargoData.serialNumber)
  }
  if (cargoData.expiryDate) {
    builder.field('Expiry Date:', formatDate(cargoData.expiryDate))
  }
  builder.field('Last Updated:', formatDateTime(cargoData.lastUpdated))

  if (cargoData.contractorId && cargoData.contractorName) {
    builder.sectionTitle('Contractor Information')
    builder.field('Contractor Name:', cargoData.contractorName)
    builder.field('Contractor ID:', cargoData.contractorId)
  }

  if (cargoData.events && cargoData.events.length > 0) {
    builder.sectionTitle('Event Timeline')
    builder.table(
      [
        { label: 'Type', width: 30 },
        { label: 'Title', width: 60 },
        { label: 'Employee', width: 45 },
        { label: 'Date', width: 35 }
      ],
      cargoData.events.map((event) => ({
        cells: [event.type.substring(0, 12), event.title.substring(0, 25), event.employee.substring(0, 18), formatDateTime(event.timestamp)]
      }))
    )
  }

  if (cargoData.locationHistory && cargoData.locationHistory.length > 0) {
    builder.sectionTitle('Location History')
    builder.table(
      [
        { label: 'Location', width: 60 },
        { label: 'Details', width: 55 },
        { label: 'Date', width: 35 },
        { label: 'Duration', width: 30 }
      ],
      cargoData.locationHistory.map((history) => ({
        cells: [
          history.location.substring(0, 20),
          history.details.substring(0, 18),
          formatDate(history.movedDate),
          history.duration
        ]
      }))
    )
  }

  if (cargoData.documents && cargoData.documents.length > 0) {
    builder.sectionTitle('Documentation')
    builder.table(
      [
        { label: 'Document Name', width: 90 },
        { label: 'Type', width: 35 },
        { label: 'Size', width: 25 },
        { label: 'Upload Date', width: 30 }
      ],
      cargoData.documents.map((document) => ({
        cells: [document.name.substring(0, 30), document.type, document.size, formatDate(document.uploadDate)]
      }))
    )
  }

  builder.sectionTitle('Report Summary')
  builder.field('Report Generated:', formatDateTime(new Date()))
  builder.field('Report Type:', 'Comprehensive Cargo Report')
  builder.multiLineField(
    'Notes:',
    'This cargo report provides a comprehensive overview of the cargo item including its current status, location, physical attributes, and historical data. For more detailed information or updates, please access the warehouse management system.'
  )

  const reportDate = formatDate(new Date()).replace(/\s+/g, '_')
  builder.save(`Cargo_Report_${cargoData.sku}_${reportDate}.pdf`)
}
