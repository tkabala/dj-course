import { PdfDocumentBuilder } from './core/pdfDocumentBuilder'

export interface TrackingEvent {
  id: number | string
  status: string
  location: string
  timestamp: string
  description: string
}

export interface ShipmentInfo {
  id: string | number
  origin: string
  destination: string
  driver: string
  eta?: string
  status?: string
}

export async function generateShipmentRoutePDF(shipment: ShipmentInfo, events: TrackingEvent[]): Promise<void> {
  const builder = await PdfDocumentBuilder.create({
    title: `Shipment Route - #${shipment.id}`,
    subtitle: 'Deliveroo Logistics'
  })

  builder.sectionTitle('Route Overview')
  builder.multiLineField('From:', shipment.origin)
  builder.multiLineField('To:', shipment.destination)
  builder.multiLineField('Driver:', shipment.driver)

  if (shipment.eta) {
    builder.field('ETA:', shipment.eta)
  }

  if (shipment.status) {
    builder.field('Status:', shipment.status)
  }

  builder.sectionTitle('Timeline')

  events.forEach((event, index) => {
    builder.ensureSpace(20)

    const isLast = index === events.length - 1
    const fillColor: [number, number, number] = isLast ? [33, 150, 243] : [34, 197, 94]
    builder.internalDoc.setFillColor(fillColor[0], fillColor[1], fillColor[2])
    builder.internalDoc.circle(25, builder.currentY, 2, 'F')

    builder.internalDoc.setFontSize(10)
    builder.internalDoc.setFont('helvetica', 'bold')
    builder.internalDoc.text(event.status, 30, builder.currentY)
    builder.internalDoc.setFont('helvetica', 'normal')
    builder.internalDoc.text(event.timestamp, 140, builder.currentY)

    builder.currentY += 4
    builder.internalDoc.setFontSize(9)
    builder.internalDoc.text(event.location, 30, builder.currentY)
    builder.currentY += 4
    builder.internalDoc.setTextColor(100, 100, 100)
    builder.internalDoc.text(event.description, 30, builder.currentY)
    builder.internalDoc.setTextColor(0, 0, 0)
    builder.currentY += 10
  })

  builder.save(`Shipment_${shipment.id}_Route.pdf`)
}
