import Card from '../components/Card'
import Contributions from '../components/Contributions'
import DecisionNotes from '../components/DecisionNotes'
import RiskBadge from '../components/RiskBadge'
import { useApplicantPrediction } from '../lib/useApplicantPrediction'

export default function Explainability() {
  const { samples, selectedId, setSelectedId, result, loading, error } = useApplicantPrediction()

  return (
    <div className="flex flex-col gap-8">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-(--color-ink)">Explainability</h1>
        <p className="mt-1.5 text-sm text-(--color-ink-muted)">
          Every prediction comes with the specific factors that drove it, computed with SHAP for
          that individual applicant.
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

      {result && !loading && (
        <Card>
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-sm font-semibold text-(--color-ink)">
              Why applicant #{result.sk_id_curr} scored {(result.risk_score * 100).toFixed(1)}%
            </h2>
            <RiskBadge band={result.risk_band} />
          </div>
          <p className="mt-1.5 text-xs text-(--color-ink-muted)">
            Bar length is the size of each feature's contribution, color shows direction.
          </p>
          <div className="mt-5">
            <Contributions contributions={result.top_contributions} />
          </div>
        </Card>
      )}

      <DecisionNotes
        points={[
          {
            title: 'Why SHAP over a simpler method',
            detail:
              'shap.TreeExplainer computes exact Shapley values for tree models like LightGBM, using an algorithm built specifically for trees rather than a sampling-based approximation. For this specific applicant, it calculates exactly how much each feature pushed the prediction away from the average.',
          },
          {
            title: 'Why only the top 6 features are shown',
            detail:
              'The model uses 85 features in total. Showing all of them would not be useful for a non-technical reviewer, so only the features that mattered most for this decision are shown, sorted by the size of their contribution.',
          },
          {
            title: 'Checked against ground truth',
            detail:
              'A known defaulter in this public dataset scored 84% risk, with low EXT_SOURCE values correctly identified as the top risk-increasing factors in the correct direction. This was verified directly against the dataset.',
          },
        ]}
      />
    </div>
  )
}
