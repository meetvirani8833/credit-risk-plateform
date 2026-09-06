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
          Every prediction comes with the specific factors that drove it, computed with SHAP, not
          a generic list of "important features" for the whole model.
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
              'shap.TreeExplainer is exact (not approximated) for tree models like LightGBM: it computes, for this specific applicant, exactly how much each feature pushed the prediction away from the average, using a fast algorithm built for trees, no sampling involved.',
          },
          {
            title: 'Why only the top 6 features',
            detail:
              'The model actually uses 85 features. A non-technical reviewer does not want all 85, just what mattered most for this decision, sorted by the size of their contribution.',
          },
          {
            title: 'Checked against ground truth',
            detail:
              'A known real defaulter in this public dataset scored 84% risk with low EXT_SOURCE values correctly identified as the top risk-increasing factors, in the correct direction, this was verified directly, not assumed.',
          },
        ]}
      />
    </div>
  )
}
