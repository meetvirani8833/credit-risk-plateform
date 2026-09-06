"""Derive business-readable IF-THEN rules from the trained LightGBM model.

Approach: global surrogate model. LightGBM itself is an ensemble of
hundreds of trees, not directly readable as rules. Instead, a shallow
decision tree (depth 3) is trained to approximate LightGBM's OWN predicted
risk scores (not the raw labels), and each root-to-leaf path in that small
tree becomes one rule. Every rule is then validated against real,
held-out outcomes, a rule that looks clean but covers 12 applicants or
doesn't actually separate risk in reality is worthless, however
interpretable it reads.

Deliberate scoping decision: the surrogate is trained on a curated set of
NUMERIC features only (EXT_SOURCE_*, age, income ratios, bureau
aggregates), not the full 85-feature set LightGBM sees. High-cardinality
categoricals like ORGANIZATION_TYPE (185 distinct values) would need
one-hot encoding to feed a sklearn tree, and the resulting splits
("ORGANIZATION_TYPE_Business Entity Type 3 <= 0.5") are not the kind of
clean, readable rule this module exists to produce. This trades some
surrogate fidelity for rule readability, the fidelity tradeoff is measured
and reported explicitly rather than hidden, see FIDELITY section of the
output.

Run as: python -m src.ml.derive_rules
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, _tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from ml.train import RANDOM_STATE, load_training_data, split_features_target  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "documents"

# Curated for readability AND actionability: every one is a continuous
# number where a threshold split reads as a plain sentence, no categorical
# encoding needed. EXT_SOURCE_1/2/3 are deliberately excluded even though
# they are the model's strongest predictors (see README): they are opaque
# external scores the bank doesn't compute or control, "your external score
# is low" is not a policy-actionable or applicant-explainable rule the way
# "your debt-to-income ratio is high" is. Excluding them trades surrogate
# fidelity for business actionability, on purpose, and the fidelity cost is
# measured and reported rather than hidden.
RULE_CANDIDATE_FEATURES = [
    "AGE_YEARS", "EMPLOYED_YEARS",
    "AMT_INCOME_TOTAL", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM_YEARS",
    "BUREAU_CREDIT_COUNT", "BUREAU_ACTIVE_CREDIT_COUNT", "BUREAU_MAX_OVERDUE",
    "REGION_RATING_CLIENT", "CNT_CHILDREN",
]

FEATURE_LABELS = {
    "EXT_SOURCE_1": "external credit score 1",
    "EXT_SOURCE_2": "external credit score 2",
    "EXT_SOURCE_3": "external credit score 3",
    "AGE_YEARS": "applicant age (years)",
    "EMPLOYED_YEARS": "years employed",
    "AMT_INCOME_TOTAL": "annual income",
    "CREDIT_INCOME_RATIO": "credit-to-income ratio",
    "ANNUITY_INCOME_RATIO": "annuity-to-income ratio",
    "CREDIT_TERM_YEARS": "loan term (years)",
    "BUREAU_CREDIT_COUNT": "number of prior bureau credits",
    "BUREAU_ACTIVE_CREDIT_COUNT": "number of active bureau credits",
    "BUREAU_MAX_OVERDUE": "worst prior overdue amount",
    "REGION_RATING_CLIENT": "region risk rating (1 best, 3 worst)",
    "CNT_CHILDREN": "number of children",
}

MIN_LEAF_FRACTION = 0.03  # each rule must cover at least 3% of applicants to be reportable
MIN_EFFECT_SIZE_PCT = 2.5  # actual default rate must differ from baseline by at least this many
# percentage points, otherwise the rule isn't really separating risk in reality, whatever the
# surrogate's own (possibly poorly calibrated) predicted band says


def extract_rules(tree: DecisionTreeRegressor, feature_names: list[str]) -> list[dict]:
    """Walk a fitted sklearn tree and return one dict per leaf: its path
    conditions (list of "feature <= value" / "feature > value" strings) and
    the leaf's predicted value (the surrogate's approximation of LightGBM's
    risk score for applicants matching that path)."""
    tree_ = tree.tree_
    rules = []

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feat = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [(feat, "<=", threshold)])
            recurse(tree_.children_right[node], conditions + [(feat, ">", threshold)])
        else:
            rules.append({"conditions": conditions, "predicted_risk": float(tree_.value[node][0][0])})

    recurse(0, [])
    return rules


def conditions_to_mask(df: pd.DataFrame, conditions: list[tuple]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for feat, op, threshold in conditions:
        if op == "<=":
            mask &= df[feat] <= threshold
        else:
            mask &= df[feat] > threshold
    return mask


def detect_narrow_range_caveat(conditions: list[tuple]) -> str | None:
    """Flag a rule whose path bounds the SAME feature on both sides
    (e.g. "> 0.81 AND <= 0.90"), a narrow window is more likely to be an
    artifact of this particular train/validation split than a robust
    pattern, worth keeping (a human can judge it) but worth surfacing
    rather than presenting with the same confidence as a single, wide
    threshold rule."""
    by_feature: dict[str, list[tuple]] = {}
    for feat, op, threshold in conditions:
        by_feature.setdefault(feat, []).append((op, threshold))
    for feat, bounds in by_feature.items():
        if len(bounds) > 1:
            values = [t for _, t in bounds]
            label = FEATURE_LABELS.get(feat, feat)
            return (
                f"Narrow range on '{label}' ({min(values):.2f} to {max(values):.2f}), "
                f"bounded from both sides by this single decision tree. Treat as a "
                f"lower-confidence pattern worth re-checking on more data before "
                f"codifying into policy, unlike the other rules here which use one "
                f"wide threshold per feature."
            )
    return None


def conditions_to_sentence(conditions: list[tuple]) -> str:
    parts = []
    for feat, op, threshold in conditions:
        label = FEATURE_LABELS.get(feat, feat)
        comparison = "at most" if op == "<=" else "above"
        parts.append(f"{label} is {comparison} {threshold:.2f}")
    return " AND ".join(parts)


def assign_band_from_actual(actual_default_rate_pct: float, baseline_pct: float) -> str:
    """Label a rule's band from its VALIDATED real-world outcome, not the
    surrogate's own raw regression output run through the full model's risk
    band cutoffs. Those cutoffs were calibrated for the full LightGBM score
    distribution, applying them to a deliberately simplified, lower-fidelity
    surrogate's output is comparing two different scales and produced a
    genuinely wrong label during testing (a rule with an actual default rate
    above baseline was mislabeled "Low" this way). Bands close to baseline
    are already excluded by MIN_EFFECT_SIZE_PCT before this is called, so
    only a High/Low distinction is needed here."""
    return "High" if actual_default_rate_pct > baseline_pct else "Low"


def main():
    artifact = joblib.load(MODELS_DIR / "model.joblib")
    model = artifact["model"]

    df, _ = load_training_data()
    X, y = split_features_target(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    lgbm_probs_train = model.predict_proba(X_train)[:, 1]
    lgbm_probs_val = model.predict_proba(X_val)[:, 1]

    # Median-impute the curated numeric subset for sklearn's tree (it can't
    # handle NaN natively the way LightGBM can), medians computed on train only.
    X_train_rules = X_train[RULE_CANDIDATE_FEATURES].copy()
    X_val_rules = X_val[RULE_CANDIDATE_FEATURES].copy()
    medians = X_train_rules.median()
    X_train_rules = X_train_rules.fillna(medians)
    X_val_rules = X_val_rules.fillna(medians)

    surrogate = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=max(200, int(MIN_LEAF_FRACTION * len(X_train_rules))),
        random_state=RANDOM_STATE,
    )
    surrogate.fit(X_train_rules, lgbm_probs_train)

    # Fidelity check: how well does this simplified, numeric-only surrogate
    # actually track the real model it's supposed to explain? Reported
    # honestly rather than assumed, this is the cost of the readability
    # tradeoff described in the module docstring.
    surrogate_probs_val = surrogate.predict(X_val_rules)
    fidelity_corr = np.corrcoef(surrogate_probs_val, lgbm_probs_val)[0, 1]
    print(f"Surrogate fidelity (correlation with real model on validation set): {fidelity_corr:.3f}\n")

    raw_rules = extract_rules(surrogate, RULE_CANDIDATE_FEATURES)

    validated_rules = []
    for rule in raw_rules:
        mask = conditions_to_mask(X_val_rules, rule["conditions"])
        coverage = mask.sum()
        if coverage == 0:
            continue
        actual_default_rate = y_val[mask].mean()
        validated_rules.append(
            {
                "conditions": rule["conditions"],
                "sentence": conditions_to_sentence(rule["conditions"]),
                "caveat": detect_narrow_range_caveat(rule["conditions"]),
                "coverage_count": int(coverage),
                "coverage_pct": round(coverage / len(X_val_rules) * 100, 2),
                "actual_default_rate_pct": round(actual_default_rate * 100, 2),
            }
        )

    # Quality over quantity: drop rules covering less than MIN_LEAF_FRACTION
    # of the validation set even if the surrogate tree produced a leaf for
    # them (can happen with skewed train/val splits), then sort by how
    # extreme (best or worst) the actual outcome is, the most business-useful
    # rules are the ones furthest from the 8% baseline default rate.
    baseline = y_val.mean() * 100
    validated_rules = [r for r in validated_rules if r["coverage_pct"] >= MIN_LEAF_FRACTION * 100]
    dropped_weak = [r for r in validated_rules if abs(r["actual_default_rate_pct"] - baseline) < MIN_EFFECT_SIZE_PCT]
    validated_rules = [r for r in validated_rules if abs(r["actual_default_rate_pct"] - baseline) >= MIN_EFFECT_SIZE_PCT]
    validated_rules.sort(key=lambda r: abs(r["actual_default_rate_pct"] - baseline), reverse=True)
    if dropped_weak:
        print(f"Dropped {len(dropped_weak)} rule(s) with real-world effect under {MIN_EFFECT_SIZE_PCT}pp "
              f"(technically valid leaves, but don't meaningfully separate risk in reality):")
        for r in dropped_weak:
            print(f"  - {r['sentence']} -> actual {r['actual_default_rate_pct']}% (baseline {baseline:.2f}%)")
        print()

    for r in validated_rules:
        r["risk_band"] = assign_band_from_actual(r["actual_default_rate_pct"], baseline)

    print(f"Baseline default rate (validation set): {baseline:.2f}%\n")
    print(f"{len(validated_rules)} rules passed both the {MIN_LEAF_FRACTION*100:.0f}% coverage bar "
          f"and the {MIN_EFFECT_SIZE_PCT}pp effect-size bar:\n")
    for i, r in enumerate(validated_rules, 1):
        print(f"RULE {i}: IF {r['sentence']}")
        print(f"  -> Risk band (from validated outcome): {r['risk_band']}")
        print(f"  -> Covers {r['coverage_count']:,} applicants ({r['coverage_pct']}% of validation set)")
        print(f"  -> ACTUAL default rate in this group: {r['actual_default_rate_pct']}% "
              f"(baseline: {baseline:.2f}%)")
        if r["caveat"]:
            print(f"  -> CAVEAT: {r['caveat']}")
        print()

    DOCS_DIR.mkdir(exist_ok=True)
    output = {
        "surrogate_fidelity_correlation": round(float(fidelity_corr), 3),
        "baseline_default_rate_pct": round(baseline, 2),
        "rules": validated_rules,
    }
    with open(DOCS_DIR / "business_rules.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {DOCS_DIR / 'business_rules.json'}")

    return output


if __name__ == "__main__":
    main()
