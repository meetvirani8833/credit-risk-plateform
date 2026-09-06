"""Inference and scoring: turn a raw applicant row into a risk score, risk
band, and a SHAP-based explanation of what drove that score.

Run as: python -m src.ml.predict  (demos on a few sample applicants)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from data.preprocessor import DROP_COLS_ALWAYS, build_feature_table  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

RISK_BAND_ORDER = ["Low", "Medium", "High"]


class RiskModel:
    """Loads the trained artifact once and exposes score / explain methods."""

    def __init__(self, model_path: Path = MODEL_PATH):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.categorical_cols = artifact["categorical_cols"]
        self.risk_bands = artifact["risk_bands"]
        self.metrics = artifact["metrics"]
        self._explainer = shap.TreeExplainer(self.model.booster_)

    def _prepare(self, app_rows: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
        """Run the same clean -> engineer -> bureau-merge pipeline used in training."""
        df, categorical_cols = build_feature_table(app_rows, bureau)
        df = df.drop(columns=[c for c in DROP_COLS_ALWAYS if c in df.columns])
        # Align to the exact training feature set and column order, missing
        # engineered columns (should not happen, but fail safe) become NaN.
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        X = df[self.feature_names].copy()
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X

    def _assign_band(self, prob: float) -> str:
        if prob >= self.risk_bands["high_cutoff"]:
            return "High"
        if prob >= self.risk_bands["medium_cutoff"]:
            return "Medium"
        return "Low"

    def score(self, app_rows: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
        """Return SK_ID_CURR, risk_score (probability of default), risk_band."""
        X = self._prepare(app_rows, bureau)
        probs = self.model.predict_proba(X)[:, 1]
        bands = [self._assign_band(p) for p in probs]
        return pd.DataFrame(
            {
                "SK_ID_CURR": app_rows["SK_ID_CURR"].values,
                "risk_score": probs,
                "risk_band": bands,
            }
        )

    def explain(self, app_rows: pd.DataFrame, bureau: pd.DataFrame, top_n: int = 6) -> list[dict]:
        """Per-row SHAP explanation: top contributing features and their direction.

        Returns a list (one dict per row) with the risk score, band, and the
        top_n features that pushed the prediction up or down the most, in
        plain-language-ready form for the UI and chatbot.
        """
        X = self._prepare(app_rows, bureau)
        probs = self.model.predict_proba(X)[:, 1]
        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):  # some SHAP/LightGBM versions return [class0, class1]
            shap_values = shap_values[1]

        results = []
        for i in range(len(X)):
            row_shap = pd.Series(shap_values[i], index=X.columns)
            top_features = row_shap.abs().sort_values(ascending=False).head(top_n).index
            contributions = [
                {
                    "feature": feat,
                    "value": X.iloc[i][feat],
                    "shap_contribution": float(row_shap[feat]),
                    "direction": "increases risk" if row_shap[feat] > 0 else "decreases risk",
                }
                for feat in top_features
            ]
            results.append(
                {
                    "SK_ID_CURR": int(app_rows["SK_ID_CURR"].iloc[i]),
                    "risk_score": float(probs[i]),
                    "risk_band": self._assign_band(probs[i]),
                    "top_contributions": contributions,
                }
            )
        return results


def main():
    sys.path.append(str(PROJECT_ROOT / "src"))
    from data.loader import load_table

    app = load_table("application_train", nrows=20)
    bureau = load_table("bureau")

    risk_model = RiskModel()
    explanations = risk_model.explain(app.head(3), bureau)
    for exp in explanations:
        print(f"\nApplicant {exp['SK_ID_CURR']}: risk_score={exp['risk_score']:.3f}, band={exp['risk_band']}")
        for c in exp["top_contributions"]:
            print(f"  {c['feature']:30s} value={c['value']!s:15s} {c['direction']} (shap={c['shap_contribution']:+.3f})")


if __name__ == "__main__":
    main()
