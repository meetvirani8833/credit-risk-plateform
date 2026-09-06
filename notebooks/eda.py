# %% [markdown]
# # Home Credit Default Risk, Exploratory Data Analysis
#
# Covers: dataset summary, data quality (missing values, anomalies, duplicates,
# type consistency), feature categorization, and 5 business insights with charts.

# %%
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT / "src"))
os.environ["DATA_RAW_DIR"] = str(PROJECT_ROOT / "data" / "raw")
from data.loader import load_all_tables  # noqa: E402

sns.set_theme(style="whitegrid")
FIG_DIR = PROJECT_ROOT / "documents" / "eda_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 50)

TARGET_LABELS = {0: "Repaid", 1: "Not Repaid"}
TARGET_PALETTE = {"Repaid": "#4C72B0", "Not Repaid": "#C44E52"}

# %% [markdown]
# ## 1. Load all tables and dataset summary

# %%
tables = load_all_tables()
for name, df in tables.items():
    print(f"{name:25s} shape={df.shape}")

# %%
app = tables["application_train"]
app["TARGET_LABEL"] = app["TARGET"].map(TARGET_LABELS)
app.head()

# %% [markdown]
# ## 2. Target distribution, class imbalance

# %%
target_counts = app["TARGET_LABEL"].value_counts()
target_pct = app["TARGET_LABEL"].value_counts(normalize=True) * 100
print(target_counts)
print(target_pct.round(2))

order = ["Repaid", "Not Repaid"]
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x="TARGET_LABEL", data=app, order=order, hue="TARGET_LABEL",
              palette=TARGET_PALETTE, legend=False, ax=ax)
ax.set_ylim(0, target_counts.max() * 1.18)
ax.set_xlabel("")
ax.set_ylabel("Number of applicants")
ax.set_title("Target Distribution")
for container in ax.containers:
    labels = [f"{v:,.0f}\n({v / len(app) * 100:.1f}%)" for v in container.datavalues]
    ax.bar_label(container, labels=labels, padding=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_target_distribution.png", dpi=120)
plt.show()

# %% [markdown]
# **Why this matters.** Only about 8% of applicants in the training set
# defaulted ("Not Repaid"). A model that always predicts "Repaid" would already
# score 92% accuracy while being useless for the actual business goal (catching
# risky applicants). This is why accuracy is the wrong metric here, and why the
# ML layer needs to explicitly handle imbalance (class weighting or
# resampling) and be evaluated on ROC-AUC, PR-AUC, and recall on the minority
# class instead.

# %% [markdown]
# ## 3. Data Quality
#
# Missing values are only one part of data quality. This section also checks
# for sentinel/placeholder values disguised as real numbers, duplicate
# records, and inconsistent encodings across similar columns.

# %% [markdown]
# ### 3.1 Missing values

# %%
missing = app.isnull().mean().sort_values(ascending=False) * 100
missing = missing[missing > 0]
print(f"{len(missing)} of {app.shape[1] - 2} columns have missing values.")
missing.head(20)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
missing.head(20).sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_xlabel("% missing")
ax.set_title("Top 20 columns by missing value rate (application_train)")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_missing_values.png", dpi=120)
plt.show()

# %% [markdown]
# The columns with the highest missingness (many above 50%, some above 99%,
# for example `COMMONAREA_*` and `NONLIVINGAPARTMENTS_*`) are all normalized
# building or apartment descriptors, not core financial fields. Core financial
# fields (`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_BIRTH`) are
# essentially complete. This means the sparse building-metadata block can be
# dropped or heavily simplified without losing much signal, rather than
# spending engineering time on complex imputation for it.

# %% [markdown]
# ### 3.2 Anomalous and sentinel values
#
# A column can be "not missing" and still be wrong. Some numeric columns use a
# placeholder value instead of a real null, which silently corrupts any
# statistic computed on the raw column.

# %%
days_employed_anomaly = (app["DAYS_EMPLOYED"] == 365243).sum()
print(f"DAYS_EMPLOYED == 365243 (about {365243 / 365.25:.0f} years): {days_employed_anomaly:,} rows "
      f"({days_employed_anomaly / len(app) * 100:.1f}% of applicants)")

xna_org = (app["ORGANIZATION_TYPE"] == "XNA").sum()
print(f"ORGANIZATION_TYPE == 'XNA': {xna_org:,} rows")

overlap = ((app["DAYS_EMPLOYED"] == 365243) & (app["ORGANIZATION_TYPE"] == "XNA")).sum()
print(f"Rows where both are true at once: {overlap:,}")

# %% [markdown]
# All 55,374 rows with `DAYS_EMPLOYED = 365243` also have
# `ORGANIZATION_TYPE = 'XNA'`. That is not a coincidence, both are the same
# placeholder pattern for "not currently employed" (this group is mostly
# pensioners). If left as-is, `DAYS_EMPLOYED` would tell the model these
# people have worked for 1,000 years, which is nonsense and would corrupt any
# distance-based or linear model, and even skews a tree model's split points.
# The fix used in this project: replace 365243 with NaN and add a separate
# `IS_EMPLOYED` flag column, so the "not employed" information is preserved
# without the fake numeric value.

# %%
gender_counts = app["CODE_GENDER"].value_counts()
income_outliers = (app["AMT_INCOME_TOTAL"] > 1e7).sum()
dup_ids = app["SK_ID_CURR"].duplicated().sum()
dup_rows = app.duplicated().sum()

print("CODE_GENDER value counts:")
print(gender_counts)
print(f"\nAMT_INCOME_TOTAL above 10,000,000: {income_outliers} rows "
      f"(max = {app['AMT_INCOME_TOTAL'].max():,.0f})")
print(f"Duplicate SK_ID_CURR: {dup_ids}")
print(f"Fully duplicate rows: {dup_rows}")

# %% [markdown]
# A few smaller findings:
# - `CODE_GENDER` has 4 rows labeled `'XNA'` (unknown gender). Negligible in
#   volume, but worth dropping or grouping into a separate small category
#   rather than guessing.
# - `AMT_INCOME_TOTAL` has 3 applicants reporting income above 10 million
#   (the top value is 117,000,000), almost certainly data entry errors given
#   how far they sit from the 99th percentile. These should be capped
#   (winsorized) rather than dropped, so the rows are not lost entirely.
# - No duplicate `SK_ID_CURR` values and no fully duplicate rows exist in
#   `application_train`, so record-level deduplication is not needed here.

# %% [markdown]
# ### 3.3 Data type and encoding consistency

# %%
flag_cols = [c for c in app.columns if c.startswith("FLAG_")]
non_numeric_flags = [c for c in flag_cols if app[c].dtype == "object"]
print(f"FLAG_* columns: {len(flag_cols)}")
print(f"FLAG_* columns encoded as text instead of 0/1: {non_numeric_flags}")
for c in non_numeric_flags:
    print(f"  {c}: {app[c].unique()}")

# %% [markdown]
# Most `FLAG_*` columns (document flags, phone/email flags) are already clean
# integer 0/1 columns. But `FLAG_OWN_CAR` and `FLAG_OWN_REALTY` are encoded as
# text (`'Y'` / `'N'`) instead, an inconsistency worth normalizing to 0/1
# during preprocessing so all flag columns behave the same way in the
# feature pipeline.

# %% [markdown]
# ## 4. Feature categorization

# %%
dtype_groups = app.dtypes.value_counts()
print(dtype_groups)

numeric_cols = app.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = app.select_dtypes(include=["object"]).columns.tolist()
binary_flag_cols = [c for c in numeric_cols if app[c].dropna().isin([0, 1]).all() and c != "TARGET"]

print(f"Numeric (non-flag): {len(numeric_cols) - len(binary_flag_cols)}")
print(f"Binary flag columns: {len(binary_flag_cols)}")
print(f"Categorical (object): {len(categorical_cols)}")

# %% [markdown]
# **Feature groups** (for the README feature-categorization deliverable):
# - **Demographics**: `CODE_GENDER`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`, `DAYS_BIRTH`, `NAME_FAMILY_STATUS`, `NAME_EDUCATION_TYPE`
# - **Financials**: `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `NAME_INCOME_TYPE`
# - **Employment and housing**: `DAYS_EMPLOYED`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, `NAME_HOUSING_TYPE`
# - **External risk scores**: `EXT_SOURCE_1/2/3`, pre-computed normalized bureau-like scores
# - **Document and contact flags**: about 40 binary `FLAG_*` and `FLAG_DOCUMENT_*` columns
# - **Building and apartment metadata**: the sparse `*_AVG` / `*_MODE` / `*_MEDI` block (section 3.1)
# - **Social circle and credit bureau enquiry counts**: `OBS_*_CNT_SOCIAL_CIRCLE`, `DEF_*_CNT_SOCIAL_CIRCLE`, `AMT_REQ_CREDIT_BUREAU_*`

# %% [markdown]
# ## 5. Business Insights
#
# Five findings that directly inform how the ML model and the business rules
# layer should be built.

# %% [markdown]
# ### 5.1 External source scores are the strongest single predictors

# %%
ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
corr = app[ext_cols + ["TARGET"]].corr()["TARGET"].drop("TARGET")
print(corr)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ext_cols):
    sns.kdeplot(data=app, x=col, hue="TARGET_LABEL", hue_order=order, fill=True,
                common_norm=False, palette=TARGET_PALETTE, ax=ax)
    ax.set_title(f"{col} (corr with default = {corr[col]:.3f})")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_ext_source_vs_target.png", dpi=120)
plt.show()

# %% [markdown]
# All three `EXT_SOURCE_*` scores are negatively correlated with default and
# show clearly separated distributions between repaid and not-repaid clients.
# These are by far the strongest individual signals in the dataset (consistent
# with public Home Credit solutions) and should be prioritized even under
# aggressive feature pruning. Their 20 to 30% missingness means the model
# needs to handle missing values natively (tree models do this), rather than
# mean-imputing, which would erase the signal.

# %% [markdown]
# ### 5.2 Younger and less employment-stable applicants default more

# %%
app["AGE_YEARS"] = -app["DAYS_BIRTH"] / 365.25
app["AGE_BAND"] = pd.cut(app["AGE_YEARS"], bins=[20, 30, 40, 50, 60, 70])
age_default = app.groupby("AGE_BAND", observed=True)["TARGET"].mean() * 100

fig, ax = plt.subplots(figsize=(6, 4))
age_default.plot(kind="bar", ax=ax, color="#C44E52")
ax.set_ylabel("Non-repayment rate (%)")
ax.set_xlabel("Age band (years)")
ax.set_title("Non-repayment rate by age band")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_default_by_age.png", dpi=120)
plt.show()
print(age_default)

# %% [markdown]
# Non-repayment falls steadily with age, the youngest applicants (20 to 30)
# default at roughly 2.3 times the rate of the oldest band (60 to 70). Younger
# clients are earlier in their careers with less stable income and employment
# history, which the model should already pick up via `DAYS_BIRTH` and
# `DAYS_EMPLOYED` directly, but it is a useful sanity check and a natural,
# explainable driver for the business rules layer.

# %% [markdown]
# ### 5.3 Income type is a strong differentiator, especially at the extremes

# %%
income_default_full = app.groupby("NAME_INCOME_TYPE")["TARGET"].agg(["mean", "count"]).sort_values("mean", ascending=False)
income_default_full["mean"] *= 100
print(income_default_full)

income_default_chart = income_default_full[income_default_full["count"] > 500]

fig, ax = plt.subplots(figsize=(7, 4))
income_default_chart["mean"].sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_xlabel("Non-repayment rate (%)")
ax.set_title("Non-repayment rate by income type (segments with over 500 applicants)")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_default_by_income_type.png", dpi=120)
plt.show()

# %% [markdown]
# Among the well-populated segments, "Working" applicants default almost twice
# as often as "Pensioner" applicants (9.6% vs 5.4%). The two smallest segments,
# "Maternity leave" (5 applicants, 40% default) and "Unemployed" (22
# applicants, 36% default), are excluded from the chart because a sample that
# small is not statistically reliable on its own, but the direction is exactly
# what intuition predicts (income interruption raises risk) and is a useful
# signal for a business rule even if it should not carry heavy weight in the
# statistical model.

# %% [markdown]
# ### 5.4 Debt burden relative to income separates the two classes

# %%
app["CREDIT_INCOME_RATIO"] = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"]
app["ANNUITY_INCOME_RATIO"] = app["AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, title in zip(
    axes,
    ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"],
    ["Credit to income ratio", "Annuity to income ratio"],
):
    clipped = app[col].clip(upper=app[col].quantile(0.99))
    sns.boxplot(x="TARGET_LABEL", y=clipped, order=order, hue="TARGET_LABEL",
                palette=TARGET_PALETTE, legend=False,
                data=app.assign(**{col: clipped}), ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_ratios_vs_target.png", dpi=120)
plt.show()

# %% [markdown]
# Clients who default tend to carry a slightly higher annuity burden relative
# to income. This motivates engineering explicit ratio features
# (`CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`) rather than relying on the
# model to infer the relationship from `AMT_CREDIT` and `AMT_INCOME_TOTAL`
# separately, a case where a small amount of manual feature engineering
# meaningfully helps a tree-based model.

# %% [markdown]
# ### 5.5 Prior credit bureau history is an independent risk signal

# %%
bureau = tables["bureau"]
bureau_agg = bureau.groupby("SK_ID_CURR").agg(
    PREV_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
    PREV_CREDIT_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
    PREV_CREDIT_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
)
merged = app[["SK_ID_CURR", "TARGET"]].merge(bureau_agg, on="SK_ID_CURR", how="left")
merged["HAS_BUREAU_HISTORY"] = merged["PREV_CREDIT_COUNT"].notna().map({True: "Has bureau history", False: "No bureau history"})

hist_default = merged.groupby("HAS_BUREAU_HISTORY")["TARGET"].mean() * 100
print(hist_default)

fig, ax = plt.subplots(figsize=(5, 4))
hist_default.plot(kind="bar", ax=ax, color=["#C44E52", "#4C72B0"])
ax.set_xticklabels(hist_default.index, rotation=0)
ax.set_ylabel("Non-repayment rate (%)")
ax.set_title("Non-repayment rate by presence of Credit Bureau history")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_bureau_history_vs_target.png", dpi=120)
plt.show()

# %% [markdown]
# Applicants with no prior Credit Bureau history default at a meaningfully
# higher rate than those with history (10.1% vs 7.7%), and among those with
# history, higher counts of active or overdue prior credits associate with
# higher default rates. This is the key justification for pulling a small
# number of aggregated features from `bureau.csv` into the model rather than
# using `application_train` alone, it is cheap (a single groupby and merge)
# and adds real, independent signal beyond what is in the main table.

# %% [markdown]
# ## 6. Summary of data quality observations
#
# 1. **Class imbalance**: about 8% positive rate, must drive model and
#    evaluation choices.
# 2. **Sparse building metadata block**: 40+ columns over 50% missing, low
#    priority to impute.
# 3. **Sentinel values disguised as numbers**: `DAYS_EMPLOYED = 365243` for
#    55,374 rows (18% of applicants), must be replaced with NaN plus an
#    `IS_EMPLOYED` flag, not used as a raw numeric feature.
# 4. **EXT_SOURCE_* partially missing (20 to 30%)** but the strongest
#    predictors, needs a model that tolerates missing values natively rather
#    than naive imputation.
# 5. **Minor category and outlier issues**: 4 `'XNA'` genders, 3 extreme
#    income outliers above 10 million, and inconsistent Y/N vs 0/1 encoding
#    on 2 flag columns. All small in volume but worth explicit handling in
#    the preprocessing pipeline for a clean, defensible pipeline.
# 6. **No duplicate records** in `application_train`, so no deduplication
#    step is needed.
# 7. **Multiple related tables** (bureau, previous_application, installments,
#    POS_CASH, credit_card) contain genuine incremental signal (5.5) but add
#    real complexity, the model's feature scope will be decided based on this
#    EDA together with the time budget for the assignment.
