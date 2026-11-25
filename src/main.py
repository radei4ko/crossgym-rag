import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.embeddings import embed_query
from src.pipeline import respond
from src.routing import FOLLOW_UP_RE

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Loading the sentence-transformers model is a ~4-5s one-time cost. Without
    # this, whoever sends the first message after a deploy/restart pays that
    # latency live instead of it happening during container startup.
    embed_query("warmup")
    yield


app = FastAPI(title="CrossGYM RAG Assistant", lifespan=_lifespan)


def _needs_coreference_resolution(question: str) -> bool:
    """True when the question contains a reference word ("її", "там", "his", "there") that
    only makes sense together with the previous turns. Function words that merely look
    like pronouns ("та", "це", "тому") are deliberately not markers."""
    return bool(FOLLOW_UP_RE.search(question))


def require_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
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
    return AskResponse(**respond(request.question, history))
