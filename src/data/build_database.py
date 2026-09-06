"""Build the talk-to-data database (applications, bureau_credits,
previous_applications) from the raw Home Credit CSVs.

Targets local SQLite by default (for development and the Dockerized local
deployment), or Postgres (Neon, for the production backend) if DATABASE_URL
is set, same schema either way, via the shared engine in data/db.py.

Loads a curated subset of columns (not all 122+17+37 raw columns) into a
handful of clearly named tables, so the NL-to-SQL LLM has a small, readable
schema to reason over instead of the full raw column list.

Run as: python -m src.data.build_database
        DATABASE_URL=postgresql+psycopg2://... python -m src.data.build_database
"""
import os
import sys
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from data.db import get_engine, sqlite_url  # noqa: E402
from data.loader import load_table  # noqa: E402

LOCAL_DB_PATH = PROJECT_ROOT / "data" / "processed" / "credit_risk.db"

# Curated column subset per table: enough for the 5+ required NL-to-SQL query
# patterns (demographics, financials, risk, repayment history) without
# dragging in the full 122-column raw schema, which would blow up the LLM's
# prompt and increase hallucination risk on column names.
APPLICATION_COLS = [
    "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT",
    "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]

BUREAU_COLS = [
    "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_TYPE", "DAYS_CREDIT",
    "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_MAX_OVERDUE", "CREDIT_DAY_OVERDUE",
]

PREVIOUS_APPLICATION_COLS = [
    "SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_APPLICATION", "AMT_CREDIT",
    "NAME_CONTRACT_STATUS", "CODE_REJECT_REASON", "DAYS_DECISION", "NAME_CLIENT_TYPE",
]


def build(db_url: str | None = None, sample_rows: int | None = None):
    """sample_rows caps how many application rows (and their related bureau/
    previous-application rows) get loaded, used to build the small demo
    dataset committed for the public deployment, see README section on the
    two data modes. None means the full dataset (local/Docker path)."""
    target_url = db_url or os.environ.get("DATABASE_URL") or sqlite_url(LOCAL_DB_PATH, read_only=False)
    engine = get_engine(target_url)

    print(f"Building database at: {target_url.split('@')[-1] if '@' in target_url else target_url}")

    print("Loading application_train...")
    app = load_table("application_train")[APPLICATION_COLS].copy()
    if sample_rows:
        app = app.sample(n=min(sample_rows, len(app)), random_state=42)
    app["AGE_YEARS"] = (-app["DAYS_BIRTH"] / 365.25).round(1)
    app["YEARS_EMPLOYED"] = (-app["DAYS_EMPLOYED"] / 365.25).where(app["DAYS_EMPLOYED"] != 365243)
    app["TARGET_LABEL"] = app["TARGET"].map({0: "Repaid", 1: "Not Repaid"})
    with engine.begin() as conn:
        app.to_sql("applications", conn, if_exists="replace", index=False)
    print(f"  -> applications: {len(app):,} rows")

    sample_ids = set(app["SK_ID_CURR"]) if sample_rows else None

    print("Loading bureau...")
    bureau = load_table("bureau")[BUREAU_COLS].copy()
    if sample_ids is not None:
        bureau = bureau[bureau["SK_ID_CURR"].isin(sample_ids)]
    with engine.begin() as conn:
        bureau.to_sql("bureau_credits", conn, if_exists="replace", index=False)
    print(f"  -> bureau_credits: {len(bureau):,} rows")

    print("Loading previous_application...")
    prev = load_table("previous_application")[PREVIOUS_APPLICATION_COLS].copy()
    if sample_ids is not None:
        prev = prev[prev["SK_ID_CURR"].isin(sample_ids)]
    with engine.begin() as conn:
        prev.to_sql("previous_applications", conn, if_exists="replace", index=False)
    print(f"  -> previous_applications: {len(prev):,} rows")

    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_app_sk_id ON applications (\"SK_ID_CURR\")"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_bureau_sk_id ON bureau_credits (\"SK_ID_CURR\")"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_prev_sk_id ON previous_applications (\"SK_ID_CURR\")"))

    print("\nDatabase build complete.")


if __name__ == "__main__":
    build()
