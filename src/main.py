from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.generation import answer_question
from src.retrieval import hybrid_search

app = FastAPI(title="CrossGYM RAG Assistant")


def require_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


class AskRequest(BaseModel):
    question: str


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
    documents = hybrid_search(request.question, match_count=settings.match_count)
    if not documents:
        return AskResponse(answer="Не маю інформації про це.", sources=[])
    result = answer_question(request.question, documents)
    return AskResponse(**result)
