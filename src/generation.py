import httpx

from src.config import settings

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "Ти асистент мережі спортзалів CrossGYM. Відповідай тільки на основі наданого контексту. "
    "Якщо відповіді немає в контексті — прямо скажи, що не маєш цієї інформації, не вигадуй. "
    "Відповідай тією ж мовою, якою поставлено питання."
)


def build_prompt(question: str, documents: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[{i + 1}] ({doc['source']})\n{doc['content']}" for i, doc in enumerate(documents))
    user_message = f"Контекст:\n{context}\n\nПитання: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def call_openrouter(messages: list[dict]) -> str:
    response = httpx.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={"model": settings.openrouter_model, "messages": messages},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def answer_question(question: str, documents: list[dict]) -> dict:
    messages = build_prompt(question, documents)
    answer = call_openrouter(messages)
    sources = [{"source": doc["source"], "content": doc["content"]} for doc in documents]
    return {"answer": answer, "sources": sources}
