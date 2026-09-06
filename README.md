# AI-Powered Credit Risk Intelligence Platform

Built for the NeoStats AI Engineer Round 1 use case, on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset.

> Status: work in progress, this README is filled in incrementally as each module is built.

## 1. Architecture Overview

_TODO: component diagram + description once modules are built._

## 2. Setup & Run Instructions

### Prerequisites
- Docker + Docker Compose
- An OpenAI API key (get one at https://platform.openai.com/api-keys), needed for the talk-to-data chatbot

### Steps
```bash
git clone <this-repo>
cd credit_risk_platform
cp .env.example .env         # then paste your OpenAI key into .env
docker-compose up --build
```
Then open http://localhost:8501 in your browser.

_TODO: dataset placement instructions (evaluator must download data/raw from Kaggle, or we ship a small processed sample)._

## 3. Model Selection & Class Imbalance Strategy

**Model: LightGBM (gradient-boosted trees).** Chosen over logistic regression
or a neural network because:
- It handles missing values natively (`EXT_SOURCE_1/2/3`, the strongest
  predictors in the dataset, are 20 to 30% missing, see `notebooks/eda.ipynb`
  section 5.1), so no lossy imputation is needed for the most important
  features.
- It handles categorical columns directly (`ORGANIZATION_TYPE`,
  `OCCUPATION_TYPE`, etc.) without one-hot blowup.
- It is fast enough to train on the full 307k-row table on a laptop CPU in
  under a minute, which matters for a time-boxed assignment and for anyone
  re-running training inside the Docker container.

**Feature scope**: `application_train` plus 4 aggregated features from
`bureau.csv` (prior credit count, active-credit count, worst overdue amount,
days since most recent bureau record). This scope was chosen after EDA
showed bureau history has a real, independent effect on default rate
(10.1% for applicants with no bureau history vs 7.7% for those with some),
while going deeper into `previous_application`, `POS_CASH_balance`,
`credit_card_balance`, or `installments_payments` would cost several more
hours of aggregation work for likely diminishing returns, since
`EXT_SOURCE_*` already carries most of the predictive signal. The sparse
building/apartment metadata block (40+ columns, over 50% missing, see EDA
section 3.1) was dropped entirely rather than imputed.

**Class imbalance strategy**: only 8.07% of applicants defaulted in
training. Rather than resampling (SMOTE, undersampling), `scale_pos_weight`
was set to the negative-to-positive class ratio (about 11.4) directly in
LightGBM, which reweights the loss function to penalize missed defaulters
more heavily, without duplicating or discarding any real rows. Evaluation
uses ROC-AUC and PR-AUC instead of accuracy, since a model that always
predicts "Repaid" would score 92% accuracy while being useless.

**A bug worth documenting**: the first training run appeared to succeed (no
errors, model saved) but silently used only 2 boosting rounds. LightGBM's
early stopping was tracking its default `binary_logloss` metric instead of
the `auc` passed to `eval_metric`, because the metric to use for early
stopping must also be set on the estimator itself (`metric="auc"`), not only
inside `.fit()`. The logloss metric overfit almost immediately while the
true validation AUC was still climbing (from 0.72 at round 2 to 0.76 by
round 102). Fixed by setting `metric="auc"` on the model constructor and
`first_metric_only=True` on the early stopping callback. Caught by manually
inspecting `model.evals_result_` after the run looked suspiciously fast.

## 4. Evaluation Metrics & Results

Validation set: stratified 20% holdout (61,502 rows), same class balance as
training.

| Metric | Value |
|---|---|
| ROC-AUC | 0.7725 |
| PR-AUC (average precision) | 0.2629 (vs 0.081 random baseline) |
| Best boosting round | 661 |

Classification report at the default 0.5 probability threshold:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Repaid | 0.96 | 0.74 | 0.83 |
| Not Repaid | 0.18 | 0.67 | 0.28 |

0.5 is an arbitrary cutoff, not a chosen operating point, and `scale_pos_weight`
(used to handle class imbalance) pushes most predicted scores upward, which
makes 0.5 sit in a low-precision part of the curve. To confirm this was a
threshold artifact and not a weaker model, we tried several `scale_pos_weight`
values (1x, 3x, 5x, 11.4x) and measured ROC-AUC and PR-AUC for each: all four
landed within 0.7725 to 0.7739 ROC-AUC and 0.2629 to 0.2663 PR-AUC, noise-level
differences. The weight does not change how well the model ranks risky vs
safe applicants, it only reshapes where the raw scores land. So instead of
changing the model, we report the classification metrics at the F1-optimal
threshold (0.69) found via `precision_recall_curve`:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Repaid | 0.94 | 0.91 | 0.93 |
| Not Repaid | 0.28 | 0.37 | 0.32 |

Same model, same ROC-AUC, a more defensible operating point. Which threshold
to actually use in production is a business decision (screening for recall
vs minimizing false positives), the risk bands below are the mechanism this
platform actually uses to communicate risk, not a hard threshold. See
`documents/model_figures/01_roc_pr_curves.png`.

**Risk bands** (Low / Medium / High) are set from percentiles of the
validation score distribution, not fixed probability cutoffs, because with
an 8% base rate, fixed cutoffs like 0.2/0.5 would leave "High" almost empty.
Top 10% of predicted scores = High, next 20% = Medium, bottom 70% = Low.
Validated directly against actual outcomes:

| Band | Applicants (validation) | Actual non-repayment rate |
|---|---|---|
| Low | 43,051 | 3.8% |
| Medium | 12,300 | 12.8% |
| High | 6,151 | 28.6% |

A 7.5x gap in actual default rate between the Low and High bands confirms
the bands are meaningfully separating risk, not just an arbitrary split. See
`documents/model_figures/04_risk_band_validation.png`.

**Top features by importance**: `ORGANIZATION_TYPE`, `CREDIT_TERM_YEARS`
(engineered), `EXT_SOURCE_3/1/2`, `OCCUPATION_TYPE`, and
`BUREAU_DAYS_CREDIT_MAX`, the last one being a `bureau.csv` feature ranking
7th overall, concrete evidence that the bureau join was worth the extra
engineering time. See `documents/model_figures/03_feature_importance.png`.

## 5. Prompt Engineering & Token Optimization

_TODO_

## 6. Rule Derivation Logic & Sample Outputs

_TODO_

## 7. Known Limitations & Possible Improvements

_TODO_
