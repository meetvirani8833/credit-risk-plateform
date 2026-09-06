from fastapi import APIRouter, HTTPException, Request

from api.schemas import Contribution, PredictRequest, PredictResponse, SampleApplicant

router = APIRouter(prefix="/predict", tags=["predict"])


@router.get("/samples", response_model=list[SampleApplicant])
def list_sample_applicants(request: Request):
    state = request.app.state.app_state
    df = state.sample_applicants
    samples = []
    for _, row in df.iterrows():
        label = (
            f"#{int(row['SK_ID_CURR'])}  "
            f"income {row['AMT_INCOME_TOTAL']:,.0f}  "
            f"credit {row['AMT_CREDIT']:,.0f}  "
            f"age {-row['DAYS_BIRTH'] / 365.25:.0f}"
        )
        samples.append(SampleApplicant(sk_id_curr=int(row["SK_ID_CURR"]), label=label))
    return samples


@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    state = request.app.state.app_state
    df = state.sample_applicants
    row = df[df["SK_ID_CURR"] == req.sk_id_curr]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Unknown sample applicant {req.sk_id_curr}")

    explanations = state.risk_model.explain(row, state.sample_bureau, top_n=6)
    exp = explanations[0]
    return PredictResponse(
        sk_id_curr=exp["SK_ID_CURR"],
        risk_score=exp["risk_score"],
        risk_band=exp["risk_band"],
        top_contributions=[
            Contribution(
                feature=c["feature"],
                value=str(c["value"]),
                direction=c["direction"],
                shap_contribution=c["shap_contribution"],
            )
            for c in exp["top_contributions"]
        ],
    )
