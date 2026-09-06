"""FastAPI backend for the credit risk platform.

Thin HTTP layer over the existing src/ modules (model, SHAP explainability,
rule derivation, LangGraph talk-to-data agent), see api/routers/*.py. Run
locally with: uvicorn api.main:app --reload
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from api.deps import EDA_FIGURES_DIR, AppState  # noqa: E402
from api.routers import chat, eda, predict, rules  # noqa: E402
from data.db import get_engine  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.app_state = AppState()
    yield


app = FastAPI(title="Credit Risk Intelligence API", lifespan=lifespan)

# FRONTEND_ORIGIN should be set to the deployed Vercel URL in production,
# "*" is fine for local development only.
frontend_origin = os.environ.get("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin] if frontend_origin != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/eda_figures", StaticFiles(directory=str(EDA_FIGURES_DIR)), name="eda_figures")

app.include_router(eda.router)
app.include_router(predict.router)
app.include_router(rules.router)
app.include_router(chat.router)


@app.get("/health")
def health():
    """Hit by the keep-alive cron to prevent the free-tier backend process
    from cold-starting between evaluator visits. Does NOT touch the
    database, see /health/db for that."""
    return {"status": "ok"}


@app.get("/health/db")
def health_db(request: Request):
    """Runs a trivial query so the keep-alive cron can also prevent Neon's
    free-tier compute from auto-suspending after ~5 minutes of no database
    activity, a separate cold-start risk from Render's, pinging /health
    alone does not touch Neon at all."""
    from sqlalchemy import text

    engine = get_engine(request.app.state.app_state.database_url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}
