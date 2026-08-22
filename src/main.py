from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.generation import answer_question
from src.retrieval import hybrid_search

app = FastAPI(title="CrossGYM RAG Assistant")

COREFERENCE_MARKERS = [
    "її", "його", "їх", "нього", "неї", "них", "цей", "ця", "це", "той", "та", "те",
    "там", "туди", "воно", "цьому", "тому",
]


def _needs_coreference_resolution(question: str) -> bool:
    """A short follow-up ("а її інстаграм?") often refers back to an entity the
    question itself doesn't name. But blindly widening every retrieval query with
    the previous answer backfires on self-contained questions: it dilutes the
    embedding with irrelevant text (a prior greeting, small talk) and can push the
    correct chunk out of the top-k. Only widen when the question actually contains
    a pronoun/reference word that needs resolving."""
    words = question.lower().split()
    return any(word.strip(".,!?") in COREFERENCE_MARKERS for word in words)


def require_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


class HistoryTurn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[HistoryTurn] = []


class Source(BaseModel):
    source: str
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(request: AskRequest) -> AskResponse:
    history = [turn.model_dump() for turn in request.history]

    retrieval_query = request.question
    if _needs_coreference_resolution(request.question):
        last_answer = next((turn["content"] for turn in reversed(history) if turn["role"] == "assistant"), None)
        if last_answer:
            retrieval_query = f"{last_answer}\n{request.question}"

    documents = hybrid_search(retrieval_query, match_count=settings.match_count)
    if not documents:
        return AskResponse(answer="Не маю інформації про це.", sources=[])
    result = answer_question(request.question, documents, history=history)
    return AskResponse(**result)
