"""Pydantic request/response models for the API."""
from pydantic import BaseModel


class SampleApplicant(BaseModel):
    sk_id_curr: int
    label: str  # human-readable summary for the dropdown, e.g. "#316484, income 202,500, age 45"


class PredictRequest(BaseModel):
    sk_id_curr: int


class Contribution(BaseModel):
    feature: str
    value: str
    direction: str
    shap_contribution: float


class PredictResponse(BaseModel):
    sk_id_curr: int
    risk_score: float
    risk_band: str
    top_contributions: list[Contribution]


class Rule(BaseModel):
    sentence: str
    risk_band: str
    coverage_count: int
    coverage_pct: float
    actual_default_rate_pct: float
    caveat: str | None = None


class RulesResponse(BaseModel):
    surrogate_fidelity_correlation: float
    baseline_default_rate_pct: float
    rules: list[Rule]


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    sql: str | None = None
    rewritten_question: str | None = None
    result_preview: list[dict] | None = None
