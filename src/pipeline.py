"""The whole "question in, answer out" flow, independent of HTTP so that /ask and the
eval harness run exactly the same code."""

import logging
import re
from typing import Optional

from src.generation import answer_question
from src.lang import EN, RU, UK, clean_user_text, pure_language_request, resolve_language
from src.retrieval import fetch_sources, hybrid_search
from src.routing import FOLLOW_UP_RE, narrow_trainers, select_sources

logger = logging.getLogger("crossgym.pipeline")

HISTORY_MESSAGES = 4  # last two exchanges; older turns only add noise and language drift
FALLBACK_MATCH_COUNT = 12

GREETING = {
    UK: "Привіт! Я асистент мережі спортзалів CrossGYM. Можу підказати ціни, розклад, тренерів і адреси залів. Що вас цікавить?",
    RU: "Привет! Я ассистент сети спортзалов CrossGYM. Могу подсказать цены, расписание, тренеров и адреса залов. Что вас интересует?",
    EN: "Hi! I'm the assistant of the CrossGYM gym chain. I can help with prices, schedules, trainers and hall addresses. What would you like to know?",
}
LANGUAGE_ACK = {
    UK: "Звісно, відповідатиму українською. Що вас цікавить?",
    RU: "Конечно, буду отвечать по-русски. Что вас интересует?",
    EN: "Sure, I'll answer in English. What would you like to know?",
}
NO_INFO = {
    UK: "Не маю інформації про це.",
    RU: "У меня нет информации об этом.",
    EN: "I don't have information about that.",
}
UNAVAILABLE = {
    UK: "Вибачте, зараз не можу відповісти. Спробуйте, будь ласка, ще раз за хвилину.",
    RU: "Извините, сейчас не могу ответить. Попробуйте, пожалуйста, ещё раз через минуту.",
    EN: "Sorry, I can't answer right now. Please try again in a minute.",
}

_GREETINGS = [
    "добрий день", "доброго дня", "добрий вечір", "доброго вечора", "доброго ранку", "добрий ранок",
    "привіт", "вітаю", "здоров", "хай", "привет", "здравствуйте", "здравствуй", "добрый день",
    "добрый вечер", "доброе утро", "здрасте", "good morning", "good evening", "good afternoon",
    "hello there", "hi there", "hello", "hey", "hi", "start",
]
_IDENTITY = [
    "хто ти", "ти хто", "хто ти такий", "що ти вмієш", "що ти можеш", "що вмієш", "кто ты", "ты кто",
    "что ты умеешь", "что ты можешь", "что умеешь", "who are you", "what can you do", "what do you do",
]
_GREETINGS.sort(key=len, reverse=True)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def is_smalltalk(question: str) -> bool:
    """A bare greeting and/or "who are you". "Привіт, скільки коштує абонемент?" is NOT
    small talk: that one goes to the model, which is told not to open with a greeting."""
    text = _normalise(question)
    if not text:
        return False
    consumed = True
    while consumed:
        consumed = False
        for greeting in _GREETINGS:
            if text == greeting or text.startswith(greeting + " "):
                text = text[len(greeting):].strip()
                consumed = True
                break
    return text == "" or text in _IDENTITY


def _sanitise_history(history: Optional[list[dict]]) -> list[dict]:
    clean = []
    for turn in history or []:
        role, content = turn.get("role"), turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            clean.append({"role": role, "content": clean_user_text(content) if role == "user" else content})
    return clean[-HISTORY_MESSAGES:]


def retrieve(question: str, history: list[dict]) -> list[dict]:
    sources = select_sources(question, history)
    if sources:
        documents = fetch_sources(sources)
        if documents:
            documents = narrow_trainers(question, documents)
            logger.info("routed to %s (%d chunks)", sources, len(documents))
            return documents
    query = question
    if history and FOLLOW_UP_RE.search(question):
        last_answer = next((t["content"] for t in reversed(history) if t["role"] == "assistant"), None)
        if last_answer:
            query = f"{last_answer}\n{question}"
    logger.info("fallback hybrid search for %r", question)
    return narrow_trainers(question, hybrid_search(query, match_count=FALLBACK_MATCH_COUNT))


def respond(question: str, history: Optional[list[dict]] = None) -> dict:
    history = _sanitise_history(history)
    lang = resolve_language(question, history)

    if is_smalltalk(question):
        return {"answer": GREETING[lang], "sources": []}
    requested = pure_language_request(question)
    if requested:  # "english?" has nothing to look up: acknowledge in that language, no search, no model
        return {"answer": LANGUAGE_ACK[requested], "sources": []}

    documents = retrieve(question, history)
    if not documents:
        return {"answer": NO_INFO[lang], "sources": []}
    try:
        result = answer_question(question, documents, history=history, lang=lang)
        if not result["answer"].strip():  # e.g. the whole reply was an ungrounded phone number
            result["answer"] = NO_INFO[lang]
        return result
    except Exception:  # provider outage / rate limit: answer politely instead of a bare HTTP 500
        logger.exception("generation failed for %r", question)
        return {"answer": UNAVAILABLE[lang], "sources": []}
