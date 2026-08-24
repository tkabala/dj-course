import { PdfDocumentBuilder } from './core/pdfDocumentBuilder'
import { formatCurrency, formatDate, formatEnum } from './core/pdfFormatters'

export interface WarehousingRequestFormData {
  storageType: string
  status?: string
  securityLevel: string
  estimatedVolume: number
  estimatedWeight: number
  estimatedStorageDuration: {
    value: number
    unit: 'days' | 'weeks' | 'months' | 'years'
  }
  plannedStartDate: string | Date
  plannedEndDate?: string | Date
  handlingServices: string[]
  valueAddedServices: string[]
  requiresTemperatureControl: boolean
  requiresHumidityControl: boolean
  requiresSpecialHandling: boolean
  specialInstructions?: string
  billingType: string
  cargo: WarehousingCargoData
  priority: string
  currency: string
  estimatedCost?: number
  finalCost?: number
  inventoryStatus?: string
}

export interface WarehousingCargoData {
  description: string
  cargoType: string
  packaging: string
  quantity: number
  unitType: string
  value: number
  currency: string
  weight?: number
  dimensions?: {
    length: number
    width: number
    height: number
    unit: string
  }
  hazardousClass?: string
  temperatureRequirements?: { min: number; max: number; unit: string }
  stackable?: boolean
  fragile?: boolean
}

export interface WarehousingRequestPdfOptions {
  requestNumber?: string
  createdAt?: Date | string
  storageLocation?: string
}

export async function generateWarehousingRequestPDF(
  formData: WarehousingRequestFormData,
  options: WarehousingRequestPdfOptions = {}
): Promise<void> {
  const builder = await PdfDocumentBuilder.create({
    title: 'Warehousing Request',
    subtitle: 'Deliveroo Logistics'
  })

  renderRequestInformation(builder, formData, options)
  renderStorageInformation(builder, formData, options)
  renderCargo(builder, formData.cargo)
  renderServiceRequirements(builder, formData)

  const filename = options.requestNumber
    ? `Warehousing_Request_${options.requestNumber}.pdf`
    : `Warehousing_Request_${new Date().toISOString().split('T')[0]}.pdf`

  builder.save(filename)
}

function renderRequestInformation(
  builder: PdfDocumentBuilder,
  formData: WarehousingRequestFormData,
  options: WarehousingRequestPdfOptions
): void {
  builder.sectionTitle('Request Information')

  if (options.requestNumber) {
    builder.field('Request Number:', options.requestNumber)
  }

  if (formData.status) {
    builder.field('Status:', formatEnum(formData.status))
  }

  if (formData.inventoryStatus) {
    builder.field('Inventory Status:', formatEnum(formData.inventoryStatus))
  }

  builder.field('Storage Type:', formatEnum(formData.storageType))
  builder.field('Priority:', formatEnum(formData.priority))

  if (options.createdAt) {
    builder.field('Created:', formatDate(options.createdAt))
  }
}

function renderStorageInformation(
  builder: PdfDocumentBuilder,
  formData: WarehousingRequestFormData,
  options: WarehousingRequestPdfOptions
): void {
  builder.sectionTitle('Storage Information')

  builder.field('Estimated Volume:', `${formData.estimatedVolume} m³`)
  builder.field('Estimated Weight:', `${formData.estimatedWeight} kg`)
  builder.field('Security Level:', formatEnum(formData.securityLevel))

  if (options.storageLocation) {
    builder.field('Storage Location:', options.storageLocation)
  }

  builder.field('Planned Start Date:', formatDate(formData.plannedStartDate))

  if (formData.plannedEndDate) {
    builder.field('Planned End Date:', formatDate(formData.plannedEndDate))
  }

  const duration = formData.estimatedStorageDuration ?? { value: 0, unit: 'months' }
  builder.field('Storage Duration:', `${duration.value} ${duration.unit}`)
  builder.field('Billing Type:', formatEnum(formData.billingType))
}

function renderCargo(builder: PdfDocumentBuilder, cargo: WarehousingCargoData): void {
  builder.sectionTitle('Cargo Information')

  builder.multiLineField('Description:', cargo.description || 'No description provided')
  builder.field('Cargo Type:', formatEnum(cargo.cargoType))
  builder.field('Packaging:', formatEnum(cargo.packaging))
  builder.field('Quantity:', `${cargo.quantity || 0} ${cargo.unitType || ''}`)

  if (cargo.weight) {
    builder.field('Weight:', `${cargo.weight} kg`)
  }

  if (cargo.dimensions) {
    builder.field(
      'Dimensions:',
      `${cargo.dimensions.length} × ${cargo.dimensions.width} × ${cargo.dimensions.height} ${cargo.dimensions.unit}`
    )
  }

  if (cargo.value && cargo.value > 0) {
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

function renderServiceRequirements(builder: PdfDocumentBuilder, formData: WarehousingRequestFormData): void {
  builder.sectionTitle('Service Requirements')

  if (formData.handlingServices && formData.handlingServices.length > 0) {
    builder.field('Handling Services:', formData.handlingServices.map(formatEnum).join(', '))
  }

  if (formData.valueAddedServices && formData.valueAddedServices.length > 0) {
    builder.field('Value Added Services:', formData.valueAddedServices.map(formatEnum).join(', '))
  }

  builder.field('Requires Temperature Control:', formData.requiresTemperatureControl ? 'Yes' : 'No')
  builder.field('Requires Humidity Control:', formData.requiresHumidityControl ? 'Yes' : 'No')
  builder.field('Requires Special Handling:', formData.requiresSpecialHandling ? 'Yes' : 'No')

  if (formData.specialInstructions) {
    builder.multiLineField('Special Instructions:', formData.specialInstructions)
  }

  if (formData.estimatedCost) {
    builder.field('Estimated Cost:', formatCurrency(formData.estimatedCost, formData.currency))
  }

  if (formData.finalCost) {
    builder.field('Final Cost:', formatCurrency(formData.finalCost, formData.currency))
  }
}
