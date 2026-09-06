"""One SQLAlchemy engine factory, shared by the local build script and the
talk-to-data query layer, so the same code targets SQLite locally (for fast
iteration, no external dependency) and Postgres (Neon) in production via a
single DATABASE_URL connection string.

SQLite URL format: sqlite:///file:<path>?mode=ro&uri=true (read-only, a
real OS-level guarantee on top of the SQL-level validation in query_runner.py).
Postgres URL format: postgresql+psycopg2://user:password@host/dbname
(read-only enforcement there should come from using a read-only DB role,
this is a deployment-time recommendation, not something the app can force).
"""
import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def sqlite_url(db_path: Path, read_only: bool = True) -> str:
    posix_path = Path(db_path).resolve().as_posix()
    if read_only:
        return f"sqlite:///file:{posix_path}?mode=ro&uri=true"
    return f"sqlite:///{posix_path}"


def default_database_url() -> str:
    """DATABASE_URL env var wins (Postgres in production), otherwise fall
    back to the local SQLite build, this is what makes local dev and Docker
    work with zero extra configuration."""
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    project_root = Path(__file__).resolve().parents[2]
    default_path = project_root / "data" / "processed" / "credit_risk.db"
    return sqlite_url(default_path)


@lru_cache(maxsize=8)
def get_engine(db_url: str) -> Engine:
    return create_engine(db_url)
