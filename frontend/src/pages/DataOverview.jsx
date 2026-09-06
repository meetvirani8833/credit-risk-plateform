import { useEffect, useState } from 'react'
import Card from '../components/Card'
import ChartImage from '../components/ChartImage'
import DecisionNotes from '../components/DecisionNotes'
import { api } from '../lib/api'

export default function DataOverview() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.edaInsights().then(setData).catch((e) => setError(e.message))
  }, [])

  return (
    <div className="flex flex-col gap-8">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-(--color-ink)">Data Overview</h1>
        <p className="mt-1.5 text-sm text-(--color-ink-muted)">
          307,511 loan applications from the Home Credit Default Risk dataset, an 8.07% default
          rate, and the 5 findings that shaped every decision made downstream.
        </p>
      </header>

      {error && (
        <Card className="border-(--color-high) text-sm text-(--color-high)">
          Could not load EDA data: {error}
        </Card>
      )}

      {data && (
        <>
          <Card>
            <h2 className="text-sm font-semibold text-(--color-ink)">Data quality, at a glance</h2>
            <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Stat label="Default rate" value={`${data.data_quality.class_imbalance_pct}%`} />
              <Stat
                label="Columns with missing values"
                value={`${data.data_quality.columns_with_missing} / ${data.data_quality.total_columns}`}
              />
              <Stat
                label="DAYS_EMPLOYED sentinel rows"
                value={data.data_quality.days_employed_sentinel_rows.toLocaleString()}
              />
              <Stat label="Duplicate rows" value="0" />
            </div>
            <ul className="mt-5 flex flex-col gap-2 border-t border-(--color-border) pt-4">
              {data.data_quality.notes.map((n, i) => (
                <li key={i} className="text-sm leading-relaxed text-(--color-ink-muted)">
                  {n}
                </li>
              ))}
            </ul>
          </Card>

          <div className="flex flex-col gap-5">
            <h2 className="text-sm font-semibold text-(--color-ink)">5 business insights</h2>
            {data.insights.map((insight, i) => (
              <Card key={i} className="flex flex-col gap-4">
                <div>
                  <div className="text-xs font-medium text-(--color-ink-faint)">
                    Insight {i + 1}
                  </div>
                  <h3 className="mt-1 text-sm font-semibold text-(--color-ink)">
                    {insight.title}
                  </h3>
                  <p className="mt-2 text-sm leading-relaxed text-(--color-ink-muted)">
                    {insight.summary}
                  </p>
                </div>
                <ChartImage src={api.chartUrl(insight.chart)} alt={insight.title} />
              </Card>
            ))}
          </div>
        </>
      )}

      <DecisionNotes
        points={[
          {
            title: 'Why these 5 insights',
            detail:
              'Each one directly informed a later build decision: the imbalance drove the model\'s class-weighting strategy, the missing-value pattern justified dropping the sparse building-metadata block, and the bureau-history finding is why bureau.csv was joined into the model at all.',
          },
          {
            title: 'Why static charts, not live recomputation',
            detail:
              'The full dataset (307K rows across 7 tables) is not loaded into this running app, the charts are pre-generated once during EDA and served as images, keeping the deployed backend light and fast to start.',
          },
          {
            title: 'Why report a sentinel-value bug here',
            detail:
              'DAYS_EMPLOYED uses 365243 as a placeholder for "not employed" on 18% of rows. Left as-is it would tell the model these applicants have worked for 1,000 years. It is replaced with a missing value plus an explicit IS_EMPLOYED flag.',
          },
        ]}
      />
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div>
      <div className="text-lg font-semibold text-(--color-ink)">{value}</div>
      <div className="mt-0.5 text-xs text-(--color-ink-muted)">{label}</div>
    </div>
  )
}
