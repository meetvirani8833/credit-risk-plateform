import { useEffect, useState } from 'react'
import Card from '../components/Card'
import DecisionNotes from '../components/DecisionNotes'
import RiskBadge from '../components/RiskBadge'
import { api } from '../lib/api'

export default function Rules() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.rules().then(setData).catch((e) => setError(e.message))
  }, [])

  return (
    <div className="flex flex-col gap-8">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-(--color-ink)">
          Business Rules
        </h1>
        <p className="mt-1.5 text-sm text-(--color-ink-muted)">
          Plain IF-THEN rules extracted from the model, each validated against real, held-out
          outcomes.
        </p>
      </header>

      {error && (
        <Card className="border-(--color-high) text-sm text-(--color-high)">Error: {error}</Card>
      )}

      {data && (
        <>
          <Card className="flex flex-wrap gap-8">
            <Stat label="Baseline default rate" value={`${data.baseline_default_rate_pct}%`} />
            <Stat
              label="Surrogate fidelity (vs full model)"
              value={data.surrogate_fidelity_correlation.toFixed(3)}
            />
            <Stat label="Rules that passed quality filters" value={data.rules.length} />
          </Card>

          <div className="flex flex-col gap-5">
            {data.rules.map((rule, i) => (
              <Card key={i}>
                <div className="flex items-start justify-between gap-3">
                  <div className="text-xs font-medium text-(--color-ink-faint)">RULE {i + 1}</div>
                  <RiskBadge band={rule.risk_band} />
                </div>
                <p className="mt-2 text-sm leading-relaxed text-(--color-ink)">
                  <span className="font-medium">IF</span> {rule.sentence}
                </p>
                <div className="mt-4 grid grid-cols-2 gap-4 border-t border-(--color-border) pt-4 sm:grid-cols-3">
                  <Stat label="Coverage" value={`${rule.coverage_pct}% (${rule.coverage_count.toLocaleString()} applicants)`} small />
                  <Stat label="Actual default rate" value={`${rule.actual_default_rate_pct}%`} small />
                  <Stat label="Baseline" value={`${data.baseline_default_rate_pct}%`} small />
                </div>
                {rule.caveat && (
                  <p className="mt-4 rounded-lg bg-(--color-medium-bg) px-3.5 py-2.5 text-xs leading-relaxed text-(--color-medium)">
                    {rule.caveat}
                  </p>
                )}
              </Card>
            ))}
          </div>
        </>
      )}

      <DecisionNotes
        points={[
          {
            title: 'How these rules were derived',
            detail:
              'A shallow decision tree was trained to approximate the LightGBM model\'s own predictions, since the real model is an ensemble of hundreds of trees and not directly readable. Each path through that small tree becomes one rule.',
          },
          {
            title: 'Why EXT_SOURCE scores are excluded, despite being the strongest predictors',
            detail:
              'They are opaque external scores the bank does not compute or control, so an applicant cannot be told concretely what to change, unlike a factor such as debt-to-income ratio. This reduces fidelity to the underlying model (from 0.689 to 0.393) in exchange for rules a credit team can act on directly.',
          },
          {
            title: 'Why some technically valid rules were dropped',
            detail:
              'A rule must cover at least 3% of applicants, and its actual default rate must differ from baseline by at least 2.5 percentage points, otherwise it does not meaningfully separate risk from noise. 3 of 6 raw candidates were removed on this basis.',
          },
        ]}
      />
    </div>
  )
}

function Stat({ label, value, small }) {
  return (
    <div>
      <div className={`font-semibold text-(--color-ink) ${small ? 'text-sm' : 'text-lg'}`}>
        {value}
      </div>
      <div className="mt-0.5 text-xs text-(--color-ink-muted)">{label}</div>
    </div>
  )
}
