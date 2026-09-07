import { randomUUID } from 'node:crypto'

export interface LocationDetails {
  street: string
  city: string
  country: string
  contactPerson: string
  contactPhone: string
}

export interface CargoDetails {
  description: string
  weight: number
}

export interface TransportationRequestData {
  serviceType: string
  pickup: LocationDetails & { pickupDate: string }
  delivery: LocationDetails
  cargo: CargoDetails
  specialInstructions: string
  priority: Priority
}

export type Priority = 'Low' | 'Normal' | 'High' | 'Urgent'

/** `yyyy-mm-dd`, the format a native `<input type="date">` accepts. */
function daysFromNow(days: number): string {
  return new Date(Date.now() + days * 24 * 60 * 60 * 1000).toISOString().slice(0, 10)
}

/**
 * A complete, valid request. The cargo description carries a random suffix so a
 * test can point at *its own* row in the requests listing - the mock backend
 * seeds six requests and every worker adds more.
 */
export function aTransportationRequest(
  overrides: Partial<TransportationRequestData> = {},
): TransportationRequestData {
  return {
    serviceType: 'Full Truckload (FTL)',
    pickup: {
      street: 'ul. Testowa 1',
      city: 'Warsaw',
      country: 'Poland',
      contactPerson: 'Anna Kowalska',
      contactPhone: '+48111222333',
      pickupDate: daysFromNow(7),
    },
    delivery: {
      street: 'Musterstrasse 5',
      city: 'Berlin',
      country: 'Germany',
      contactPerson: 'Hans Müller',
      contactPhone: '+49444555666',
    },
    cargo: {
      description: `Palletized electronics ${randomUUID().slice(0, 8)}`,
      weight: 2500,
    },
    specialInstructions: 'Please call 30 minutes before arrival.',
    priority: 'High',
    ...overrides,
  }
}
