import { useState } from 'react'

const TABLES = [
  {
    name: 'Applicants',
    rows: '307,511 people',
    fields:
      'Gender, age, children, family status, education, housing type, region rating, income, loan credit amount, annuity, income type (Working / Pensioner / Unemployed / etc), occupation, organization, 3 external credit scores, and whether they repaid or defaulted.',
  },
  {
    name: 'Bureau credits',
    rows: 'prior loans from OTHER lenders (many per applicant)',
    fields:
      'Status (Active / Closed / Sold), credit type (car loan, credit card, consumer credit, etc), credit amount, remaining debt, worst overdue amount.',
  },
  {
    name: 'Previous applications',
    rows: 'prior loan applications at Home Credit itself (many per applicant)',
    fields:
      'Amount requested vs actually granted, outcome (Approved / Refused / Cancelled / Unused offer), reason if refused, whether they were a new or repeat client.',
  },
]

export default function DataDictionary() {
  const [open, setOpen] = useState(true)

  return (
    <div className="rounded-xl border border-(--color-border) bg-(--color-surface)">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between gap-3 px-5 py-3.5 text-left"
      >
        <span className="text-sm font-medium text-(--color-ink)">
          What data can I ask about?
        </span>
        <span
          className={`text-(--color-ink-faint) transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        >
          &#9662;
        </span>
      </button>
      {open && (
        <div className="flex flex-col gap-4 border-t border-(--color-border) px-5 py-4">
          <p className="text-sm leading-relaxed text-(--color-ink-muted)">
            This assistant is deliberately scoped to 3 curated tables rather than the full raw
            dataset, a hallucination-control choice: a smaller, well-defined schema means the
            model cannot invent a column or table that doesn't exist, and every answer traces back
            to real, verifiable data. A question outside this scope gets an honest "I can't answer
            that from this data," not a guess.
          </p>
          {TABLES.map((t) => (
            <div key={t.name}>
              <div className="text-sm font-semibold text-(--color-ink)">
                {t.name} <span className="font-normal text-(--color-ink-faint)">, {t.rows}</span>
              </div>
              <p className="mt-1 text-sm leading-relaxed text-(--color-ink-muted)">{t.fields}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
