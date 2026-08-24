export function formatDate(date: Date | string | undefined | null, options?: Intl.DateTimeFormatOptions): string {
  if (!date) return 'Not specified'
  const d = typeof date === 'string' ? new Date(date) : date
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    ...options
  }).format(d)
}

export function formatDateTime(date: Date | string | undefined | null): string {
  return formatDate(date, { hour: '2-digit', minute: '2-digit' })
}

export function formatCurrency(amount: number | undefined | null, currency = 'EUR'): string {
  if (amount === undefined || amount === null) return '-'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency || 'EUR'
  }).format(amount)
}

export function formatEnum(value: string | undefined | null): string {
  if (!value) return 'Not specified'
  return value
    .replace(/_/g, ' ')
    .toLowerCase()
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

export function formatBoolean(value: boolean | undefined | null): string {
  return value ? 'Yes' : 'No'
}
