from fastapi import APIRouter, Request

from api.schemas import Rule, RulesResponse

router = APIRouter(prefix="/rules", tags=["rules"])


@router.get("", response_model=RulesResponse)
def get_rules(request: Request):
    data = request.app.state.app_state.business_rules
    return RulesResponse(
        surrogate_fidelity_correlation=data["surrogate_fidelity_correlation"],
        baseline_default_rate_pct=data["baseline_default_rate_pct"],
        rules=[
            Rule(
                sentence=r["sentence"],
                risk_band=r["risk_band"],
                coverage_count=r["coverage_count"],
                coverage_pct=r["coverage_pct"],
                actual_default_rate_pct=r["actual_default_rate_pct"],
                caveat=r.get("caveat"),
            )
            for r in data["rules"]
        ],
    )
