"""Cleaning, encoding, imputation, and feature engineering.

Every transformation here is directly justified by a finding from
notebooks/eda.py, see the docstring on each function for the specific
insight it addresses.
"""
import numpy as np
import pandas as pd

DAYS_EMPLOYED_SENTINEL = 365243
INCOME_CAP = 2_000_000  # winsorize cap, EDA section 3.2: 3 rows above 10M are data errors
CATEGORICAL_DTYPE_COLS: list[str] = []  # populated by clean_application_table


def clean_application_table(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the anomalies found in EDA section 3 (application_train/test)."""
    df = df.copy()

    # 3.2: DAYS_EMPLOYED sentinel (365243 = "not currently employed", mostly pensioners).
    # Replace with NaN and preserve the information as an explicit flag instead of a fake day count.
    df["IS_EMPLOYED"] = (df["DAYS_EMPLOYED"] != DAYS_EMPLOYED_SENTINEL).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)

    # 3.3: FLAG_OWN_CAR / FLAG_OWN_REALTY are 'Y'/'N' text, unlike every other FLAG_* column (0/1 int).
    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[col] = df[col].map({"Y": 1, "N": 0}).astype("Int64")

    # 3.2: 4 rows with CODE_GENDER == 'XNA', negligible volume, drop rather than guess.
    df = df[df["CODE_GENDER"] != "XNA"].copy()

    # 3.2: 3 extreme AMT_INCOME_TOTAL outliers (up to 117,000,000), cap rather than drop the row.
    df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=INCOME_CAP)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio and age features motivated by EDA insights 5.2 and 5.4."""
    df = df.copy()
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM_YEARS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"] / 12
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    return df


def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau.csv down to one row per SK_ID_CURR (EDA insight 5.5).

    Kept intentionally small (4 features): prior credit count, active-credit
    count, worst overdue amount, and days since the most recent bureau
    record. This is the "application_train + bureau aggregates" scope agreed
    after the EDA review, not a full bureau feature set.
    """
    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_CREDIT_COUNT=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
        BUREAU_MAX_OVERDUE=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),  # least-negative = most recent
    )
    agg["HAS_BUREAU_HISTORY"] = 1
    return agg.reset_index()


def build_feature_table(app: pd.DataFrame, bureau: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Full pipeline: clean -> engineer -> merge bureau aggregates.

    Returns the feature dataframe (still containing SK_ID_CURR and TARGET if
    present) plus the list of categorical column names, so the caller can
    tell LightGBM which columns to treat as categorical.
    """
    df = clean_application_table(app)
    df = engineer_features(df)

    bureau_features = build_bureau_features(bureau)
    df = df.merge(bureau_features, on="SK_ID_CURR", how="left")
    df["HAS_BUREAU_HISTORY"] = df["HAS_BUREAU_HISTORY"].fillna(0).astype(int)

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    return df, categorical_cols


DROP_COLS_ALWAYS = [
    # EDA section 3.1: sparse building/apartment metadata block, >50% missing,
    # low signal relative to imputation effort. Dropped rather than imputed.
    c for c in [
        "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG",
        "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG",
        "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG",
        "NONLIVINGAREA_AVG", "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE",
        "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE",
        "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE",
        "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE", "APARTMENTS_MEDI", "BASEMENTAREA_MEDI",
        "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI",
        "ENTRANCES_MEDI", "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI",
        "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI", "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE", "TOTALAREA_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
    ]
]
