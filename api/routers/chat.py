from fastapi import APIRouter, Request

from api.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    state = request.app.state.app_state
    agent = state.get_or_create_session(req.session_id)
    result = agent.ask(req.message)

    result_preview = None
    if result["result"] is not None and not result["result"].empty:
        result_preview = result["result"].head(10).to_dict(orient="records")

    return ChatResponse(
        answer=result["answer"],
        sql=result["sql"],
        rewritten_question=result["rewritten_question"],
        result_preview=result_preview,
    )
