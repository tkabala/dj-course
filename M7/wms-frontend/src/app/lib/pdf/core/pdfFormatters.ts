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

export function formatCurrency(amount: number | undefined | null, currency = 'USD'): string {
  if (amount === undefined || amount === null) return '-'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency || 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(amount)
}

export function formatPercentage(value: number | undefined | null, fractionDigits = 1): string {
  if (value === undefined || value === null) return '-'
  return `${value.toFixed(fractionDigits)}%`
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
