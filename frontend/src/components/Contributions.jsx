export default function Contributions({ contributions }) {
  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.shap_contribution)), 0.0001)

  return (
    <div className="flex flex-col gap-3">
      {contributions.map((c, i) => {
        const widthPct = (Math.abs(c.shap_contribution) / maxAbs) * 100
        const increases = c.direction.includes('increase')
        return (
          <div key={i}>
            <div className="flex items-baseline justify-between gap-2 text-xs">
              <span className="font-medium text-(--color-ink)">{c.feature}</span>
              <span className="text-(--color-ink-faint)">value: {formatValue(c.value)}</span>
            </div>
            <div className="mt-1 flex items-center gap-2">
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-(--color-canvas)">
                <div
                  className={`h-full rounded-full ${increases ? 'bg-(--color-high)' : 'bg-(--color-low)'}`}
                  style={{ width: `${widthPct}%` }}
                />
              </div>
              <span
                className={`w-28 shrink-0 text-right text-xs ${increases ? 'text-(--color-high)' : 'text-(--color-low)'}`}
              >
                {c.direction}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function formatValue(v) {
  const n = Number(v)
  if (!Number.isNaN(n) && Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (!Number.isNaN(n) && !Number.isInteger(n)) return n.toFixed(3)
  return String(v)
}
