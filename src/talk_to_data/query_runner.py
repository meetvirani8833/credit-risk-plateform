"""SQL validation and execution against the read-only credit risk database.

The LLM's prompt asks for safe, single-SELECT, LIMIT-bounded SQL, but a
prompt is a request, not a guarantee. Every query is re-validated here in
code before touching the database, and the database connection itself is
opened read-only as a second, independent layer of defense.
"""
import difflib
import re
import sqlite3
from pathlib import Path

import pandas as pd

# Columns worth grounding: small, fixed-vocabulary categorical columns where
# a typo'd or invented literal (e.g. 'Unemployeed' instead of 'Unemployed')
# would silently return zero rows instead of erroring, the kind of mistake
# an LLM makes confidently. Cached once per process since these tables are
# large (millions of rows for bureau_credits/previous_applications) and the
# vocabularies never change at runtime.
GROUNDED_COLUMNS = {
    "NAME_INCOME_TYPE": "applications",
    "NAME_EDUCATION_TYPE": "applications",
    "NAME_FAMILY_STATUS": "applications",
    "NAME_HOUSING_TYPE": "applications",
    "CODE_GENDER": "applications",
    "CREDIT_ACTIVE": "bureau_credits",
    "CREDIT_TYPE": "bureau_credits",
    "NAME_CONTRACT_STATUS": "previous_applications",
    "CODE_REJECT_REASON": "previous_applications",
    "NAME_CLIENT_TYPE": "previous_applications",
}

_value_cache: dict[str, set[str]] = {}


def load_grounded_values(db_path: Path) -> dict[str, set[str]]:
    """Cache the distinct real values of each column in GROUNDED_COLUMNS."""
    global _value_cache
    if _value_cache:
        return _value_cache
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        for col, table in GROUNDED_COLUMNS.items():
            rows = conn.execute(f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL").fetchall()
            _value_cache[col] = {str(r[0]) for r in rows}
    finally:
        conn.close()
    return _value_cache


def check_value_grounding(sql: str, db_path: Path) -> str | None:
    """Return a correction hint string if the SQL filters a grounded column
    against a literal that isn't a real value, else None.

    Scaled-down version of an entity-resolution check: instead of a vector
    search over ambiguous entity types, this is an exact/fuzzy match against
    a small, fully-enumerable set of known values, sqlite has none of these
    columns above ~20 distinct values, so a cached set is enough, no
    embeddings needed.
    """
    values_by_col = load_grounded_values(db_path)
    for col, real_values in values_by_col.items():
        for match in re.finditer(rf"\b{col}\s*=\s*'([^']*)'", sql, re.IGNORECASE):
            literal = match.group(1)
            if literal in real_values:
                continue
            suggestion = difflib.get_close_matches(literal, real_values, n=1)
            hint = (
                f"Column {col} does not contain the value '{literal}'. "
                f"Valid values are: {sorted(real_values)}."
            )
            if suggestion:
                hint += f" Did you mean '{suggestion[0]}'?"
            return hint
    return None

FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create", "attach",
    "detach", "pragma", "vacuum", "replace", "grant", "revoke", "reindex",
    "trigger", "into outfile", "exec", "execute",
]

ALLOWED_TABLES = {"applications", "bureau_credits", "previous_applications"}

DEFAULT_ROW_LIMIT = 200


class UnsafeQueryError(Exception):
    pass


def validate_sql(sql: str, db_path: Path | None = None) -> str:
    """Raise UnsafeQueryError if the query isn't a single, safe SELECT, or
    if it filters a known categorical column against a value that doesn't
    actually exist (see check_value_grounding).

    Returns the (possibly LIMIT-appended) query to actually execute.
    """
    cleaned = sql.strip().rstrip(";").strip()

    if not cleaned:
        raise UnsafeQueryError("Empty query.")

    statements = [s for s in cleaned.split(";") if s.strip()]
    if len(statements) != 1:
        raise UnsafeQueryError("Only a single SQL statement is allowed.")

    lowered = cleaned.lower()
    if not lowered.startswith("select") and not lowered.startswith("with"):
        raise UnsafeQueryError("Only SELECT queries are allowed.")

    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            raise UnsafeQueryError(f"Query contains a forbidden keyword: '{kw}'.")

    # CTE aliases (WITH x AS (...), y AS (...)) are not real tables, but they
    # are legitimate to reference in FROM/JOIN, this is exactly the
    # aggregate-the-many-side-first pattern the prompt asks the LLM to use
    # to avoid one-to-many join fan-out, so they must not be flagged as
    # unknown tables.
    cte_names = set(re.findall(r"\bwith\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", lowered)) | set(
        re.findall(r",\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", lowered)
    )

    referenced_tables = set(re.findall(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered)) | set(
        re.findall(r"\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered)
    )
    unknown_tables = referenced_tables - ALLOWED_TABLES - cte_names
    if unknown_tables:
        raise UnsafeQueryError(f"Query references unknown table(s): {unknown_tables}.")

    if db_path is not None:
        grounding_error = check_value_grounding(cleaned, db_path)
        if grounding_error:
            raise UnsafeQueryError(grounding_error)

    if "limit" not in lowered:
        cleaned = f"{cleaned} LIMIT {DEFAULT_ROW_LIMIT}"

    return cleaned


def execute_sql(safe_sql: str, db_path: Path) -> pd.DataFrame:
    """Run an already-validated query against a read-only connection.

    Split from validate_sql so the LangGraph pipeline can treat validation
    (catches bad SQL before it runs) and execution (catches runtime errors,
    e.g. valid SQL that still fails) as two separately retriable steps.
    """
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return pd.read_sql_query(safe_sql, conn)
    finally:
        conn.close()


def run_query(sql: str, db_path: Path) -> pd.DataFrame:
    """Convenience wrapper: validate then execute in one call."""
    safe_sql = validate_sql(sql, db_path)
    return execute_sql(safe_sql, db_path)
