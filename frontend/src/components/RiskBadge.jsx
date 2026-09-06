const STYLES = {
  Low: 'text-(--color-low) bg-(--color-low-bg)',
  Medium: 'text-(--color-medium) bg-(--color-medium-bg)',
  High: 'text-(--color-high) bg-(--color-high-bg)',
}

export default function RiskBadge({ band, size = 'md' }) {
  const sizeClass = size === 'lg' ? 'px-3.5 py-1.5 text-sm' : 'px-2.5 py-1 text-xs'
  return (
    <span
      className={`inline-flex items-center rounded-full font-medium ${sizeClass} ${STYLES[band] || 'text-(--color-ink-muted) bg-(--color-canvas)'}`}
    >
      {band}
    </span>
  )
}
