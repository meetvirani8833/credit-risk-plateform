import Card from '../components/Card'
import DecisionNotes from '../components/DecisionNotes'
import RiskBadge from '../components/RiskBadge'
import { useApplicantPrediction } from '../lib/useApplicantPrediction'

export default function Prediction() {
  const { samples, selectedId, setSelectedId, result, loading, error } = useApplicantPrediction()

  return (
    <div className="flex flex-col gap-8">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-(--color-ink)">
          Risk Prediction
        </h1>
        <p className="mt-1.5 text-sm text-(--color-ink-muted)">
          Pick a held-out applicant (never seen during training) and score them with the trained
          LightGBM model.
        </p>
      </header>

      <Card>
        <label className="text-xs font-medium text-(--color-ink-muted)">Sample applicant</label>
        <select
          value={selectedId ?? ''}
          onChange={(e) => setSelectedId(Number(e.target.value))}
          className="mt-2 w-full rounded-lg border border-(--color-border) bg-(--color-canvas) px-3 py-2.5 text-sm text-(--color-ink) outline-none focus:border-(--color-ink-faint)"
        >
          {samples.map((s) => (
            <option key={s.sk_id_curr} value={s.sk_id_curr}>
              {s.label}
            </option>
          ))}
        </select>
      </Card>

      {error && (
        <Card className="border-(--color-high) text-sm text-(--color-high)">Error: {error}</Card>
      )}

      {loading && <Card className="text-sm text-(--color-ink-muted)">Scoring applicant...</Card>}

      {result && !loading && (
        <Card className="flex flex-col items-start gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="text-xs font-medium text-(--color-ink-faint)">
              Applicant #{result.sk_id_curr}
            </div>
            <div className="mt-1 text-3xl font-semibold tracking-tight text-(--color-ink)">
              {(result.risk_score * 100).toFixed(1)}%
            </div>
            <div className="mt-0.5 text-xs text-(--color-ink-muted)">predicted default probability</div>
          </div>
          <RiskBadge band={result.risk_band} size="lg" />
        </Card>
      )}

      <DecisionNotes
        points={[
          {
            title: 'Why these applicants',
            detail:
              'A curated set of 40 held-out rows from application_test.csv, real applicants the model never trained on, spanning a realistic mix of risk levels (34 Low, 4 Medium, 2 High). This avoids requiring a hand-typed form with 30+ raw fields.',
          },
          {
            title: 'Why LightGBM',
            detail:
              'Handles missing values and categorical columns natively. EXT_SOURCE_1/2/3, the strongest predictors, are 20-30% missing, and mean-imputing them would blunt the signal a tree model can use directly.',
          },
          {
            title: 'Why the risk band is percentile-based',
            detail:
              'With an 8% base default rate, fixed thresholds like 0.2/0.5 leave "High" almost empty. Bands are set from the top 10% / next 20% / bottom 70% of the validation score distribution, then validated against real outcomes: Low 3.8%, Medium 12.8%, High 28.6% actual default rate.',
          },
        ]}
      />
    </div>
  )
}
