import { useState } from 'react'

// The "why this was built this way" panel required on every page: a short,
// honest summary of what was decided for this section and the reasoning
// behind it, written for someone evaluating the project, not just using it.
export default function DecisionNotes({ title = 'Design decisions', points }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-xl border border-(--color-border) bg-(--color-surface)">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between gap-3 px-5 py-3.5 text-left"
      >
        <span className="text-sm font-medium text-(--color-ink)">{title}</span>
        <span
          className={`text-(--color-ink-faint) transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        >
          &#9662;
        </span>
      </button>
      {open && (
        <div className="border-t border-(--color-border) px-5 py-4">
          <ul className="flex flex-col gap-3">
            {points.map((p, i) => (
              <li key={i} className="text-sm leading-relaxed text-(--color-ink-muted)">
                <span className="font-medium text-(--color-ink)">{p.title}. </span>
                {p.detail}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
