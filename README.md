# AI-Powered Credit Risk Intelligence Platform

Built for the NeoStats AI Engineer Round 1 use case, on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset.

**Live deployment**: frontend at [credit-risk-plateform.vercel.app](https://credit-risk-plateform.vercel.app), backend API at [credit-risk-api-l0nw.onrender.com](https://credit-risk-api-l0nw.onrender.com). Both run on free tiers and may take a few seconds to wake up on the first request after a period of inactivity.

## 1. Architecture Overview

```
                         ┌─────────────────────┐
                         │   React SPA (Vercel) │
                         │  5 pages: EDA,        │
                         │  Prediction,          │
                         │  Explainability,      │
                         │  Rules, Chat          │
                         └──────────┬────────────┘
                                    │ HTTPS (fetch)
                                    v
                         ┌─────────────────────┐
                         │  FastAPI backend      │
                         │  (Render, api/)       │
                         │  /eda /predict /rules │
                         │  /chat (rate-limited) │
                         └──┬────────────────┬───┘
                            │                │
              model.joblib  │                │  DATABASE_URL
              + SHAP        │                │  (falls back:
              (src/ml/)     │                │   full local db ->
                            v                v   committed demo db)
                    ┌───────────────┐ ┌──────────────────────┐
                    │ LightGBM model │ │ Postgres (Neon) or    │
                    │ + surrogate    │ │ SQLite: applications, │
                    │ rules          │ │ bureau_credits,       │
                    └───────────────┘ │ previous_applications │
                                       └──────────┬────────────┘
                                                   │
                                       LangGraph pipeline (src/talk_to_data/):
                                       rewrite -> generate SQL -> validate ->
                                       execute -> summarize, via OpenAI
```

Two deployment paths, same codebase:
- **Local / Docker**: `docker-compose up`, backend + frontend containers on your machine, SQLite by default.
- **Production**: frontend on Vercel, backend on Render, database on Neon Postgres, exactly what the live links above run.

## 2. Setup & Run Instructions

### Option A: Docker (recommended, matches the submission requirement)

**Prerequisites**: Docker + Docker Compose, an OpenAI API key ([get one here](https://platform.openai.com/api-keys)).

```bash
git clone <this-repo>
cd credit_risk_platform
cp .env.example .env         # paste your OpenAI key into .env
docker-compose up --build
```
Then open **http://localhost:3000** (frontend) in your browser. The backend API is at http://localhost:8000.

This works with **zero dataset setup**: the chatbot and prediction pages use a small (~4MB) committed demo dataset (`data/sample/`) by default, so the full application is usable immediately. The trained model (`models/model.joblib`) is the real one, trained on the full dataset, only the chatbot's queryable database and the prediction dropdown's sample applicants are the smaller demo set in this zero-config path.

**To use the full dataset instead** (optional): download the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data) into `data/raw/`, then run:
```bash
docker compose exec backend python -m src.data.build_database
docker compose restart backend
```
This builds `data/processed/credit_risk.db` (persisted via the `./data` volume mount) from the full dataset, which the app then prefers automatically over the demo database, no code or config changes needed, see the fallback chain in `src/data/db.py`.

### Option B: Run locally without Docker

```bash
pip install -r requirements.txt
cp .env.example .env         # paste your OpenAI key into .env
uvicorn api.main:app --reload --port 8000       # backend
cd frontend && npm install && npm run dev        # frontend, separate terminal
```

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

## 5. Talk-to-Data: Architecture, Prompt Engineering, and Hallucination Control

**Pipeline**: a 6-node LangGraph state graph (`src/talk_to_data/nl_to_sql.py`):
`rewrite_question -> generate_sql -> validate_query -> execute_query -> summarize_result`,
with a bounded (1 retry) self-correction loop back to `generate_sql` on
either validation or execution failure, and an honest `fallback` node
instead of ever guessing an answer.

**Why not a bigger retrieval-based NL-to-SQL architecture** (semantic table
retrieval, a join-path graph database, multi-step entity resolution): those
solve the problem of a schema too large to fit in one prompt. This
database has exactly 3 curated tables (`applications`, `bureau_credits`,
`previous_applications`, ~40 columns total), which fits entirely in one
system prompt already, so that machinery would be solving a problem this
dataset doesn't have. The real multi-table risk here is different: joining
a one-row-per-applicant table to a many-rows-per-applicant table
(`bureau_credits`, `previous_applications`) and aggregating naively causes
silent row duplication (fan-out). The schema description given to the LLM
includes an explicit worked example of aggregating the many-side table in
a CTE before joining, verified working in testing (see sample outputs below).

**Prompt engineering choices** (`src/talk_to_data/prompt_templates.py`):
- The schema description is a curated, human-readable subset (24/9/9
  columns), not the raw 122+17+37 column dump, this is the single biggest
  lever on both token cost and hallucinated column names.
- `NO_QUERY: <reason>` is a required output for unanswerable questions
  (e.g. asking for a phone number, not in this schema), tested and
  confirmed the model uses this instead of guessing.
- `rewrite_question` resolves conversational follow-ups ("what about for
  women?") into standalone questions using the last 4 turns of history,
  tested and confirmed correct disambiguation across ambiguous prior context.

**Hallucination control beyond the prompt** (defense in depth, the prompt
is a request, not a guarantee):
1. `validate_query()` independently re-checks every generated query: single
   statement only, SELECT/WITH only, a blocklist of dangerous keywords
   (INSERT/DROP/ATTACH/PRAGMA/etc.), and that every referenced table
   (accounting for CTE aliases) is one of the 3 known tables.
2. The database connection is opened **read-only** (`mode=ro`), a second,
   independent layer, even a bug in validation can't cause a write.
3. A **value-grounding check** catches an invented or misspelled category
   value (e.g. `NAME_INCOME_TYPE = 'Retired'`, when the real value is
   `'Pensioner'`) against a cached set of real distinct values, before
   the query ever runs, and suggests the closest real value.

**Token optimization**: only the curated schema (not the full raw column
list) is ever sent, results are capped at 30 rows before being serialized
into the answer-generation prompt regardless of how large the underlying
query result is, and `temperature=0` is used throughout for reproducible,
non-creative SQL generation.

**Tested sample outputs** (7 query patterns run against live OpenAI calls,
see chat log / `documents/` for the full transcript):

| Question | Type | Result |
|---|---|---|
| "How many applicants took cash loans versus revolving loans?" | Single-table aggregate | 278,232 vs 29,279, correct |
| "What's the average income by education type?" | Single-table GROUP BY | 5 groups with counts, correct |
| "What's the average income of applicants with more than 2 active bureau credits?" | Cross-table JOIN, fan-out-safe | Used the CTE aggregate-then-join pattern correctly, avg income 190,120 |
| "What's the default rate for applicants with a refused prior application vs those without?" | Cross-table JOIN, real finding | 10.32% vs 6.98%, a genuine, intuitive result |
| "what about by family status instead?" (follow-up) | Conversation memory | Correctly resolved to a standalone income-by-family-status question |
| "What is applicant 100002's phone number?" | Out-of-schema | Correctly returned `NO_QUERY` instead of guessing |
| "How many applicants have income type 'Unemployeed'?" (typo) | Robustness | LLM self-corrected the typo before generating SQL |

**Two real bugs found and fixed during this testing** (not just written and
assumed correct): (1) the table-safety check initially flagged CTE aliases
like `bureau_agg` as unknown tables, rejecting the exact fan-out-safe
pattern the prompt asks for, fixed by extracting CTE names from the `WITH`
clause. (2) `NO_QUERY` responses were initially treated as a retryable
mistake, causing an infinite loop until LangGraph's recursion limit was
hit, fixed by routing `NO_QUERY` straight to `fallback`.

## 6. Rule Derivation Logic & Sample Outputs

**Approach**: a global surrogate model. LightGBM is an ensemble of hundreds
of trees, not directly readable as IF-THEN rules. `src/ml/derive_rules.py`
trains a shallow (depth 3) `DecisionTreeRegressor` to approximate LightGBM's
own predicted risk scores, then reads each root-to-leaf path in that small
tree as one candidate rule. Every rule is then validated against real,
held-out outcomes (not the surrogate's own approximation), a rule that
reads cleanly but doesn't actually separate real risk, or only covers a
handful of applicants, is discarded rather than shipped.

**A deliberate scoping decision**: the surrogate is trained only on
numeric, business-actionable features (age, employment length, income
ratios, bureau aggregates), excluding `EXT_SOURCE_1/2/3` even though they
are the model's strongest predictors. An opaque external score isn't
something a credit policy team can act on or explain to an applicant the
way "your debt-to-income ratio is high" is. This trades surrogate fidelity
for actionability on purpose: correlation with the real model's scores on
the validation set is **0.393** with external scores excluded, honestly
reported here rather than hidden, versus 0.689 if they were included.

**Quality filters applied** (quality over quantity, by design): a rule must
cover at least 3% of applicants, and its actual validated default rate must
differ from the 8.07% baseline by at least 2.5 percentage points, 3 of 6
raw candidate rules were dropped by these filters for being statistically
real leaves that don't meaningfully separate risk. A narrow-range detector
also flags any rule that bounds the same feature from both sides (e.g. a
loan term between 0.81 and 0.90 years), a sign of possible overfitting to
this particular data split, rather than silently presenting it with the
same confidence as a clean, wide-threshold rule.

**Final rules** (validated on the held-out validation set, baseline default
rate 8.07%):

1. **IF** years employed > 4.52 **AND** loan term is between 0.81 and 0.90
   years **-> Low risk**. Covers 3.36% of applicants, actual default rate
   1.93%. *Caveat: narrow, double-bounded range on loan term, flagged
   automatically as a lower-confidence pattern worth re-checking on more
   data before codifying into policy.*
2. **IF** years employed ≤ 4.52 **AND** region risk rating > 1.5 **AND**
   loan term ≤ 1.79 years **-> High risk**. Covers 24.09% of applicants,
   actual default rate 12.93%.
3. **IF** years employed > 4.52 **AND** loan term > 0.90 years **AND**
   active bureau credits ≤ 1 **-> Low risk**. Covers 20.05% of applicants,
   actual default rate 5.21%.

Full output, including the 3 dropped candidate rules and why, is saved to
`documents/business_rules.json` on every run.

## 7. Known Limitations & Possible Improvements

**Modeling**
- PR-AUC (0.263) is modest in absolute terms, an inherent consequence of an
  8% base rate rather than a modeling gap, ROC-AUC and the validated risk
  bands are the more meaningful metrics here. See section 4.
- Feature scope stops at `application_train` plus 4 aggregated `bureau.csv`
  features. `previous_application`, `POS_CASH_balance`,
  `credit_card_balance`, and `installments_payments` were deliberately left
  unaggregated (see section 3) given `EXT_SOURCE_*` already carries most of
  the signal, but payment-history features from those tables (missed
  installments, utilization trends) would likely add real predictive value.
- A single stratified train/validation split was used, not k-fold cross
  validation, so the reported metrics carry some split-specific variance
  that CV would average out.
- The surrogate rule tree trades fidelity for actionability on purpose
  (0.393 correlation with the real model, see section 6), so the business
  rules are directionally correct and validated against real outcomes, but
  are a simplification of what the LightGBM model actually captures.

**Talk-to-data**
- Scoped to 3 curated tables (see section 5), so it cannot answer questions
  needing `previous_application` payment history, `POS_CASH_balance`, or
  `credit_card_balance` detail, this is a deliberate hallucination-control
  tradeoff, not an oversight.
- Depends on OpenAI availability, latency, and cost. Rate limiting
  (40 requests/hour/IP, see `api/rate_limit.py`) protects the shared demo
  key from abuse but is IP-based, not per-user, and a shared IP (office
  NAT, VPN) would share the same budget.
- No response caching, so identical or near-identical questions asked
  twice both make a fresh OpenAI call.

**Deployment**
- Both Render (backend) and Vercel (frontend) run on free tiers. Render
  free-tier instances cold-start after inactivity, mitigated with a
  GitHub Actions cron job hitting `/health` every 10 minutes, but a
  request during a redeploy can still briefly 404 before the new instance
  is fully live.
- The production Neon Postgres database caps `bureau_credits` and
  `previous_applications` at 400,000 rows each to stay under the free
  tier's 512MB storage limit, so cross-table chatbot answers reflect a
  representative sample of bureau/previous-application history in
  production, not the full table (the local Docker/SQLite path has no
  such cap).
- No authentication on any API endpoint, acceptable for a demo/assignment
  but not for a real deployment handling applicant financial data.

**Possible improvements given more time**
- Aggregate `POS_CASH_balance`, `credit_card_balance`, and
  `installments_payments` into the feature set for a likely ROC-AUC gain.
- k-fold cross-validation and probability calibration (e.g. isotonic
  regression) for more robust risk-band cutoffs.
- A monotonic-constrained surrogate tree, to keep rules business-actionable
  while improving fidelity above 0.393 without re-including opaque
  external scores.
- Per-user (not just per-IP) rate limiting and basic auth in front of the
  chatbot for a production deployment.
- A model/data drift monitoring job, since a static `model.joblib` will
  degrade as the applicant population shifts over time.
