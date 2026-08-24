import { PdfDocumentBuilder } from './core/pdfDocumentBuilder'
import { formatCurrency, formatDate, formatEnum } from './core/pdfFormatters'

export interface TransportationRequestFormData {
  serviceType: string
  status?: string
  pickupLocation: LocationData
  deliveryLocation: LocationData
  cargo: CargoData
  requestedPickupDate: string | Date
  requestedDeliveryDate?: string | Date
  specialInstructions?: string
  requiresInsurance: boolean
  requiresCustomsClearance: boolean
  priority: string
  currency: string
  vehicleRequirements?: VehicleRequirementsData
  estimatedCost?: number
  finalCost?: number
  trackingNumber?: string
  currentLocation?: string
}

export interface TransportationRequestPdfOptions {
  requestNumber?: string
  createdAt?: Date | string
}

export interface LocationData {
  address: {
    street: string
    city: string
    country: string
  }
  contactPerson: string
  contactPhone: string
  contactEmail?: string
  loadingType?: string
  facilityType?: string
  accessInstructions?: string
  operatingHours?: Record<string, { open: string; close: string }>
}

export interface CargoData {
  description: string
  cargoType: string
  weight: number
  dimensions?: {
    length: number
    width: number
    height: number
    unit: string
  }
  packaging: string
  quantity: number
  unitType: string
  value: number
  currency: string
  fragile?: boolean
  stackable?: boolean
  hazardousClass?: string
  temperatureRequirements?: { min: number; max: number; unit: string }
}

export interface VehicleRequirementsData {
  vehicleType: string
  capacity: number
  specialEquipment?: string[]
  driverRequirements?: string[]
}

export async function generateTransportationRequestPDF(
  formData: TransportationRequestFormData,
  options: TransportationRequestPdfOptions = {}
): Promise<void> {
  const builder = await PdfDocumentBuilder.create({
    title: 'Transportation Request',
    subtitle: 'Deliveroo Logistics'
  })

  renderRequestInformation(builder, formData, options)
  renderLocation(builder, 'Pickup Location', formData.pickupLocation, formData.requestedPickupDate, 'Requested Pickup Date:')
  renderLocation(builder, 'Delivery Location', formData.deliveryLocation, formData.requestedDeliveryDate, 'Requested Delivery Date:')
  renderCargo(builder, formData.cargo)
  renderServiceRequirements(builder, formData)

  const filename = options.requestNumber
    ? `Transportation_Request_${options.requestNumber}.pdf`
    : `Transportation_Request_${new Date().toISOString().split('T')[0]}.pdf`

  builder.save(filename)
}

function renderRequestInformation(
  builder: PdfDocumentBuilder,
  formData: TransportationRequestFormData,
  options: TransportationRequestPdfOptions
): void {
  builder.sectionTitle('Request Information')

  if (options.requestNumber) {
    builder.field('Request Number:', options.requestNumber)
  }

  if (formData.status) {
    builder.field('Status:', formatEnum(formData.status))
  }

  builder.field('Service Type:', formatEnum(formData.serviceType))
  builder.field('Priority:', formatEnum(formData.priority))

  if (options.createdAt) {
    builder.field('Created:', formatDate(options.createdAt))
  }
}

function renderLocation(
  builder: PdfDocumentBuilder,
  title: string,
  location: LocationData,
  date: string | Date | undefined,
  dateLabel: string
): void {
  builder.sectionTitle(title)

  const address = `${location.address.street}, ${location.address.city}, ${location.address.country}`
  builder.multiLineField('Address:', address)
  builder.field('Contact Person:', location.contactPerson)
  builder.field('Phone:', location.contactPhone)

  if (location.contactEmail) {
    builder.field('Email:', location.contactEmail)
  }

  if (date) {
    builder.field(dateLabel, formatDate(date))
  }

  if (location.loadingType) {
    builder.field('Loading Type:', formatEnum(location.loadingType))
  }

  if (location.facilityType) {
    builder.field('Facility Type:', formatEnum(location.facilityType))
  }

  if (location.accessInstructions) {
    builder.multiLineField('Access Instructions:', location.accessInstructions)
  }
}

function renderCargo(builder: PdfDocumentBuilder, cargo: CargoData): void {
  builder.sectionTitle('Cargo Information')

  builder.multiLineField('Description:', cargo.description)
  builder.field('Cargo Type:', formatEnum(cargo.cargoType))
  builder.field('Weight:', `${cargo.weight} kg`)

  if (cargo.dimensions) {
    builder.field(
      'Dimensions:',
      `${cargo.dimensions.length} × ${cargo.dimensions.width} × ${cargo.dimensions.height} ${cargo.dimensions.unit}`
    )
  }

  builder.field('Packaging:', formatEnum(cargo.packaging))
  builder.field('Quantity:', `${cargo.quantity} ${cargo.unitType}`)

  if (cargo.value > 0) {
    builder.field('Estimated Value:', formatCurrency(cargo.value, cargo.currency))
  }

  if (cargo.fragile !== undefined) {
    builder.field('Fragile:', cargo.fragile ? 'Yes' : 'No')
  }

  if (cargo.stackable !== undefined) {
    builder.field('Stackable:', cargo.stackable ? 'Yes' : 'No')
  }

  if (cargo.hazardousClass) {
    builder.field('Hazardous Class:', formatEnum(cargo.hazardousClass))
  }

  if (cargo.temperatureRequirements) {
    builder.field(
      'Temperature Requirements:',
      `${cargo.temperatureRequirements.min} - ${cargo.temperatureRequirements.max} °${cargo.temperatureRequirements.unit}`
    )
  }
}

function renderServiceRequirements(builder: PdfDocumentBuilder, formData: TransportationRequestFormData): void {
  builder.sectionTitle('Service Requirements')

  builder.field('Requires Insurance:', formData.requiresInsurance ? 'Yes' : 'No')
  builder.field('Requires Customs Clearance:', formData.requiresCustomsClearance ? 'Yes' : 'No')

  if (formData.vehicleRequirements) {
    builder.field('Vehicle Type:', formatEnum(formData.vehicleRequirements.vehicleType))
    builder.field('Vehicle Capacity:', String(formData.vehicleRequirements.capacity))

    if (formData.vehicleRequirements.specialEquipment?.length) {
      builder.field('Special Equipment:', formData.vehicleRequirements.specialEquipment.map(formatEnum).join(', '))
    }

    if (formData.vehicleRequirements.driverRequirements?.length) {
      builder.field('Driver Requirements:', formData.vehicleRequirements.driverRequirements.map(formatEnum).join(', '))
    }
  }

  if (formData.specialInstructions) {
    builder.multiLineField('Special Instructions:', formData.specialInstructions)
  }

  if (formData.estimatedCost) {
    builder.field('Estimated Cost:', formatCurrency(formData.estimatedCost, formData.currency))
  }

  if (formData.finalCost) {
    builder.field('Final Cost:', formatCurrency(formData.finalCost, formData.currency))
  }

  if (formData.trackingNumber) {
    builder.field('Tracking Number:', formData.trackingNumber)
  }

  if (formData.currentLocation) {
    builder.field('Current Location:', formData.currentLocation)
  }
}
