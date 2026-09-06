"""Metrics, plots, and classification report for a trained model artifact.

Run as: python -m src.ml.evaluate
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from data.loader import load_table  # noqa: E402
from data.preprocessor import DROP_COLS_ALWAYS, build_feature_table  # noqa: E402
from ml.train import ID_COL, NON_FEATURE_COLS, RANDOM_STATE, TARGET_COL  # noqa: E402

FIG_DIR = PROJECT_ROOT / "documents" / "model_figures"


def rebuild_validation_split():
    """Rebuild the exact same train/val split used in train.py (same random_state)."""
    app = load_table("application_train")
    bureau = load_table("bureau")
    df, _ = build_feature_table(app, bureau)
    df = df.drop(columns=[c for c in DROP_COLS_ALWAYS if c in df.columns])
    y = df[TARGET_COL]
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_val, y_val


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(PROJECT_ROOT / "models" / "model.joblib")
    model = artifact["model"]
    risk_bands = artifact["risk_bands"]

    X_val, y_val = rebuild_validation_split()
    val_probs = model.predict_proba(X_val)[:, 1]
    val_pred = (val_probs >= 0.5).astype(int)

    print("=== Classification report at threshold 0.5 ===")
    print(classification_report(y_val, val_pred, target_names=["Repaid", "Not Repaid"]))

    # Threshold 0.5 is an arbitrary default, not a business-chosen operating
    # point. Also report metrics at the F1-optimal threshold, so the
    # documented precision/recall reflects a real tradeoff decision rather
    # than an artifact of scale_pos_weight pushing scores past 0.5 for most
    # applicants (see README section 4 for the experiment behind this).
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    val_pred_best = (val_probs >= best_threshold).astype(int)
    print(f"=== Classification report at F1-optimal threshold ({best_threshold:.3f}) ===")
    print(classification_report(y_val, val_pred_best, target_names=["Repaid", "Not Repaid"]))

    # ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_val, val_probs, ax=axes[0], name="LightGBM")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    PrecisionRecallDisplay.from_predictions(y_val, val_probs, ax=axes[1], name="LightGBM")
    axes[1].axhline(y_val.mean(), linestyle="--", color="gray", label="Baseline (random)")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_roc_pr_curves.png", dpi=120)
    plt.close()

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_val, val_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Repaid", "Not Repaid"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (threshold = 0.5)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_confusion_matrix.png", dpi=120)
    plt.close()

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=model.feature_name_)
    top20 = importances.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    top20.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Top 20 Feature Importances (LightGBM, split count)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_feature_importance.png", dpi=120)
    plt.close()
    print("\nTop 10 features:")
    print(top20.sort_values(ascending=False).head(10))

    # Risk band distribution on validation set
    bands = np.select(
        [val_probs >= risk_bands["high_cutoff"], val_probs >= risk_bands["medium_cutoff"]],
        ["High", "Medium"],
        default="Low",
    )
    band_counts = pd.Series(bands).value_counts()
    band_default_rate = pd.DataFrame({"TARGET": y_val.values, "BAND": bands}).groupby("BAND")["TARGET"].mean() * 100
    print("\nRisk band counts (validation set):")
    print(band_counts)
    print("\nActual default rate within each predicted band:")
    print(band_default_rate)

    fig, ax = plt.subplots(figsize=(5, 4))
    band_default_rate.reindex(["Low", "Medium", "High"]).plot(kind="bar", ax=ax, color=["#55A868", "#DD8452", "#C44E52"])
    ax.set_ylabel("Actual non-repayment rate (%)")
    ax.set_title("Does the risk band actually separate risk?")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_risk_band_validation.png", dpi=120)
    plt.close()

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
