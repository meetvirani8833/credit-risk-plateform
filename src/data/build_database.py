"""Build the SQLite database the talk-to-data chatbot queries against.

Loads a curated subset of columns (not all 122+37+... raw columns) into a
handful of clearly named tables, so the NL-to-SQL LLM has a small, readable
schema to reason over instead of the full raw column list. Run as:
python -m src.data.build_database
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from data.loader import load_table  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "processed" / "credit_risk.db"

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


def build():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    print("Loading application_train...")
    app = load_table("application_train")[APPLICATION_COLS].copy()
    app["AGE_YEARS"] = (-app["DAYS_BIRTH"] / 365.25).round(1)
    app["YEARS_EMPLOYED"] = (-app["DAYS_EMPLOYED"] / 365.25).where(app["DAYS_EMPLOYED"] != 365243)
    app["TARGET_LABEL"] = app["TARGET"].map({0: "Repaid", 1: "Not Repaid"})
    app.to_sql("applications", conn, if_exists="replace", index=False)
    print(f"  -> applications: {len(app):,} rows")

    print("Loading bureau...")
    bureau = load_table("bureau")[BUREAU_COLS].copy()
    bureau.to_sql("bureau_credits", conn, if_exists="replace", index=False)
    print(f"  -> bureau_credits: {len(bureau):,} rows")

    print("Loading previous_application...")
    prev = load_table("previous_application")[PREVIOUS_APPLICATION_COLS].copy()
    prev.to_sql("previous_applications", conn, if_exists="replace", index=False)
    print(f"  -> previous_applications: {len(prev):,} rows")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_app_sk_id ON applications(SK_ID_CURR)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bureau_sk_id ON bureau_credits(SK_ID_CURR)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prev_sk_id ON previous_applications(SK_ID_CURR)")
    conn.commit()
    conn.close()
    print(f"\nDatabase built at {DB_PATH}")


if __name__ == "__main__":
    build()
