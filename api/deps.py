"""Process-wide singletons the API routes share: the trained model (loading
it involves reading model.joblib and building a SHAP TreeExplainer, both
expensive enough to do once at startup, not per-request), the sample
applicants used for the prediction dropdown, the derived business rules,
and per-session chatbot agents.

Loaded once at FastAPI startup (see main.py's lifespan) and stashed on
`app.state`, not as module-level globals, so tests can construct a fresh
app without import-time side effects.
"""
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data.db import default_database_url  # noqa: E402
from ml.predict import RiskModel  # noqa: E402
from talk_to_data.nl_to_sql import TalkToDataAgent  # noqa: E402

SAMPLE_APPLICANTS_PATH = PROJECT_ROOT / "data" / "sample" / "demo_predict_applicants.csv"
SAMPLE_BUREAU_PATH = PROJECT_ROOT / "data" / "sample" / "demo_predict_bureau.csv"
BUSINESS_RULES_PATH = PROJECT_ROOT / "documents" / "business_rules.json"
EDA_FIGURES_DIR = PROJECT_ROOT / "documents" / "eda_figures"


class AppState:
    def __init__(self):
        self.risk_model = RiskModel()
        self.sample_applicants = pd.read_csv(SAMPLE_APPLICANTS_PATH)
        self.sample_bureau = pd.read_csv(SAMPLE_BUREAU_PATH)
        with open(BUSINESS_RULES_PATH) as f:
            self.business_rules = json.load(f)
        self.database_url = default_database_url()
        # One TalkToDataAgent per chat session id, so conversation history
        # (used by the rewrite_question node) is isolated per browser tab
        # rather than shared across every visitor to the deployed app.
        self.chat_sessions: dict[str, TalkToDataAgent] = {}

    def get_or_create_session(self, session_id: str) -> TalkToDataAgent:
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = TalkToDataAgent(self.database_url)
        return self.chat_sessions[session_id]
