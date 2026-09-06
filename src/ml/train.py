"""Model training pipeline: clean data -> engineer features -> train LightGBM
with class-imbalance handling -> evaluate -> save artifacts.

Run as: python -m src.ml.train
"""
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from data.loader import load_table  # noqa: E402
from data.preprocessor import DROP_COLS_ALWAYS, build_feature_table  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "documents"
RANDOM_STATE = 42

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
# Columns present only because they were used to build engineered features,
# or that would leak the label / are not usable as model inputs directly.
NON_FEATURE_COLS = [ID_COL, TARGET_COL, "TARGET_LABEL"]


def load_training_data() -> tuple[pd.DataFrame, list[str]]:
    app = load_table("application_train")
    bureau = load_table("bureau")
    df, categorical_cols = build_feature_table(app, bureau)
    df = df.drop(columns=[c for c in DROP_COLS_ALWAYS if c in df.columns])
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    return df, categorical_cols


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COL]
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    return X, y


def compute_risk_bands(val_probs: np.ndarray) -> dict[str, float]:
    """Percentile-based risk-band cutoffs, not fixed probability thresholds.

    With an 8% base default rate, most predicted probabilities cluster low,
    so fixed cutoffs like 0.2/0.5 would leave the "High" band almost empty.
    Instead: top 10% of predicted scores = High, next 20% = Medium, bottom
    70% = Low. Cutoffs are computed once on the validation set and frozen
    into the saved model artifact, they are not recomputed at inference time.
    """
    high_cutoff = float(np.quantile(val_probs, 0.90))
    medium_cutoff = float(np.quantile(val_probs, 0.70))
    return {"medium_cutoff": medium_cutoff, "high_cutoff": high_cutoff}


def main():
    print("Loading and building features...")
    df, categorical_cols = load_training_data()
    X, y = split_features_target(df)
    feature_names = X.columns.tolist()

    print(f"Feature matrix: {X.shape}, categorical columns: {len(categorical_cols)}")
    print(f"Class balance: {y.value_counts(normalize=True).round(4).to_dict()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight (class imbalance handling): {scale_pos_weight:.2f}")

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",  # must be set here, not just in fit()'s eval_metric, otherwise
        # LightGBM keeps its default binary_logloss metric alongside it and early
        # stopping tracks the FIRST metric in that combined list (logloss), which
        # plateaus/overfits almost immediately while AUC is still improving.
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=categorical_cols,
        callbacks=[
            lgb.early_stopping(100, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, val_probs)
    pr_auc = average_precision_score(y_val, val_probs)

    print(f"\nValidation ROC-AUC: {roc_auc:.4f}")
    print(f"Validation PR-AUC:  {pr_auc:.4f}")
    print(f"Best iteration: {model.best_iteration_}")

    risk_bands = compute_risk_bands(val_probs)
    print(f"Risk band cutoffs: {risk_bands}")

    MODELS_DIR.mkdir(exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": feature_names,
        "categorical_cols": categorical_cols,
        "risk_bands": risk_bands,
        "metrics": {"roc_auc": roc_auc, "pr_auc": pr_auc},
    }
    joblib.dump(artifact, MODELS_DIR / "model.joblib")
    print(f"\nSaved model artifact to {MODELS_DIR / 'model.joblib'}")

    DOCS_DIR.mkdir(exist_ok=True)
    with open(DOCS_DIR / "model_metrics.json", "w") as f:
        json.dump(
            {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "best_iteration": int(model.best_iteration_),
                "scale_pos_weight": float(scale_pos_weight),
                "risk_bands": risk_bands,
                "n_features": len(feature_names),
                "train_rows": int(len(X_train)),
                "val_rows": int(len(X_val)),
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to {DOCS_DIR / 'model_metrics.json'}")

    return artifact, X_val, y_val, val_probs


if __name__ == "__main__":
    main()
