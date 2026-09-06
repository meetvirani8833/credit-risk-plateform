"""Load and join dataset tables"""
import os
from pathlib import Path

import pandas as pd

DATA_RAW_DIR = Path(os.environ.get("DATA_RAW_DIR", "data/raw"))

TABLE_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
}


def load_table(name: str, nrows: int | None = None) -> pd.DataFrame:
    """Load a single Home Credit table by its short name (see TABLE_FILES)."""
    if name not in TABLE_FILES:
        raise ValueError(f"Unknown table '{name}'. Options: {list(TABLE_FILES)}")
    path = DATA_RAW_DIR / TABLE_FILES[name]
    return pd.read_csv(path, nrows=nrows)


def load_all_tables(nrows: int | None = None) -> dict[str, pd.DataFrame]:
    """Load every table into a dict keyed by short name."""
    return {name: load_table(name, nrows=nrows) for name in TABLE_FILES}
