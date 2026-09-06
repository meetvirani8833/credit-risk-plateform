"""The talk-to-data pipeline as a LangGraph state graph.

rewrite_question -> generate_sql -> validate_query -> execute_query -> summarize_result
                          ^               |                  |
                          |  (retry,      v (retry,           v (retry,
                          +-- invalid) fallback if exhausted)  fails) fallback if exhausted)

Scoped deliberately smaller than a general-schema NL-to-SQL agent (no
semantic table retrieval, no join-path graph database): with only 3 tables
and one join key (SK_ID_CURR), the full schema fits in one prompt, so that
machinery would be solving a problem this dataset doesn't have. What IS a
real problem here is (a) conversation follow-ups, handled by
rewrite_question, and (b) one-to-many join fan-out and invented category
values corrupting results, handled by the schema's relationship note and by
validate_query's value-grounding check, see query_runner.py.
"""
import os
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from talk_to_data.prompt_templates import (
    ANSWER_SYSTEM_PROMPT,
    NL_TO_SQL_SYSTEM_PROMPT,
    REWRITE_QUESTION_PROMPT,
)
from talk_to_data.query_runner import UnsafeQueryError, execute_sql, validate_sql

load_dotenv()

MAX_RETRIES = 1
MAX_ROWS_FOR_ANSWER = 30
CONVERSATION_TURNS_KEPT = 4


class ChatState(TypedDict, total=False):
    original_question: str
    conversation_history: list[dict]  # [{"question": ..., "answer": ...}, ...]
    rewritten_question: str
    sql: str
    validation_error: str | None
    execution_error: str | None
    result: "object"  # pd.DataFrame, kept loosely typed to avoid importing pandas here
    answer: str
    retries: int
    db_url: str


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    response = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def rewrite_question_node(state: ChatState) -> ChatState:
    history = state.get("conversation_history", [])[-CONVERSATION_TURNS_KEPT:]
    if not history:
        return {**state, "rewritten_question": state["original_question"]}

    history_text = "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in history)
    prompt = REWRITE_QUESTION_PROMPT.format(
        conversation_history=history_text, current_question=state["original_question"]
    )
    rewritten = _call_llm("You rewrite questions exactly as instructed.", prompt)
    return {**state, "rewritten_question": rewritten}


def generate_sql_node(state: ChatState) -> ChatState:
    user_prompt = f"Question: {state['rewritten_question']}"
    if state.get("validation_error"):
        user_prompt += f"\n\nYour previous query failed validation, fix it:\n{state['validation_error']}"
    elif state.get("execution_error"):
        user_prompt += f"\n\nYour previous query failed to execute, fix it:\n{state['execution_error']}"

    raw = _call_llm(NL_TO_SQL_SYSTEM_PROMPT, user_prompt).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("sql\n"):
            raw = raw[len("sql\n"):]
    return {**state, "sql": raw.strip(), "validation_error": None, "execution_error": None}


def validate_query_node(state: ChatState) -> ChatState:
    sql = state["sql"]
    if sql.startswith("NO_QUERY:"):
        # Not a retryable mistake, the model is correctly declining. Force
        # retries past the limit so routing sends this straight to fallback
        # instead of looping the model on a query it's right to refuse.
        return {**state, "validation_error": "NO_QUERY", "retries": MAX_RETRIES + 1}
    try:
        safe_sql = validate_sql(sql, state["db_url"])
        return {**state, "sql": safe_sql, "validation_error": None}
    except UnsafeQueryError as e:
        return {**state, "validation_error": str(e), "retries": state.get("retries", 0) + 1}


def execute_query_node(state: ChatState) -> ChatState:
    try:
        result = execute_sql(state["sql"], state["db_url"])
        return {**state, "result": result, "execution_error": None}
    except Exception as e:  # noqa: BLE001 sqlite driver errors vary
        return {**state, "execution_error": str(e), "retries": state.get("retries", 0) + 1}


def fallback_node(state: ChatState) -> ChatState:
    if state["sql"].startswith("NO_QUERY:"):
        reason = state["sql"][len("NO_QUERY:"):].strip()
        answer = f"I can't answer that from this data: {reason}"
    else:
        err = state.get("validation_error") or state.get("execution_error") or "unknown error"
        answer = f"I generated a query but couldn't get a reliable answer, even after retrying: {err}"
    return {**state, "answer": answer, "result": None}


def summarize_result_node(state: ChatState) -> ChatState:
    result = state["result"]
    preview = result.head(MAX_ROWS_FOR_ANSWER).to_string(index=False) if not result.empty else "(no rows returned)"
    user_prompt = f"Question: {state['rewritten_question']}\n\nSQL used: {state['sql']}\n\nResults:\n{preview}"
    answer = _call_llm(ANSWER_SYSTEM_PROMPT, user_prompt)
    return {**state, "answer": answer}


def route_after_validate(state: ChatState) -> str:
    if state.get("validation_error") is None:
        return "execute_query"
    if state.get("retries", 0) > MAX_RETRIES:
        return "fallback"
    return "generate_sql"


def route_after_execute(state: ChatState) -> str:
    if state.get("execution_error") is None:
        return "summarize_result"
    if state.get("retries", 0) > MAX_RETRIES:
        return "fallback"
    return "generate_sql"


def build_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(ChatState)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("validate_query", validate_query_node)
    graph.add_node("execute_query", execute_query_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("summarize_result", summarize_result_node)

    graph.set_entry_point("rewrite_question")
    graph.add_edge("rewrite_question", "generate_sql")
    graph.add_edge("generate_sql", "validate_query")
    graph.add_conditional_edges(
        "validate_query", route_after_validate,
        {"execute_query": "execute_query", "generate_sql": "generate_sql", "fallback": "fallback"},
    )
    graph.add_conditional_edges(
        "execute_query", route_after_execute,
        {"summarize_result": "summarize_result", "generate_sql": "generate_sql", "fallback": "fallback"},
    )
    graph.add_edge("summarize_result", END)
    graph.add_edge("fallback", END)
    return graph.compile()


class TalkToDataAgent:
    """Thin wrapper: holds the compiled graph, the db connection string, and
    conversation history across turns. In the FastAPI backend, one instance
    is kept per session id (see api/deps.py); locally, the caller can just
    keep one instance alive for the process lifetime."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.graph = build_graph()
        self.history: list[dict] = []

    def ask(self, question: str) -> dict:
        initial_state: ChatState = {
            "original_question": question,
            "conversation_history": self.history,
            "db_url": self.db_url,
            "retries": 0,
        }
        final_state = self.graph.invoke(initial_state)
        self.history.append({"question": question, "answer": final_state.get("answer", "")})
        return {
            "question": question,
            "rewritten_question": final_state.get("rewritten_question"),
            "sql": final_state.get("sql"),
            "result": final_state.get("result"),
            "answer": final_state.get("answer"),
        }
