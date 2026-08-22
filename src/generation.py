from typing import Optional

import httpx

from src.config import settings

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "Ти асистент мережі спортзалів CrossGYM у Telegram-чаті. Відповідай тільки на основі "
    "наданого контексту. Якщо відповіді немає в контексті — прямо скажи, що не маєш цієї "
    "інформації, не вигадуй.\n\n"
    "Формат відповіді:\n"
    "- Пиши як живу репліку в месенджері: коротко, природно, без канцеляриту.\n"
    "- НІКОЛИ не використовуй markdown-заголовки (# або ##) — Telegram показує символи "
    "решітки буквально, це виглядає як сміття.\n"
    "- Не розбивай відповідь на розділи з підзаголовками. Списком (через «-») пиши тільки "
    "якщо перелічуєш кілька пунктів (наприклад кількох тренерів) — інакше суцільним текстом.\n"
    "- Не став запитання у відповідь, якщо можеш відповісти напряму з контексту. Питай "
    "уточнення тільки якщо дійсно неможливо зрозуміти що людина має на увазі.\n"
    "- Якщо в історії діалогу вище згадувалась конкретна людина, зал чи тема — і нове "
    "питання явно про неї ж (наприклад «а її інстаграм?», «а там скільки коштує?») — "
    "відповідай про неї, не перепитуй хто мається на увазі.\n"
    "- Ідентифікатори з контексту (номери телефонів, Instagram-акаунти, ціни, адреси) "
    "копіюй у відповідь ДОСЛІВНО, символ у символ. Символи підкреслення (_) в акаунтах "
    "— це частина ніка, а не markdown-курсив: не прибирай і не змінюй їх.\n"
    "- Пиши грамотною стандартною українською мовою, без орфографічних помилок і без "
    "рідкісних чи застарілих словоформ — якщо не впевнений у формі слова, вибирай "
    "простішу й очевидно правильну.\n"
    "- Відповідай тією ж мовою, якою поставлено питання."
)


def build_prompt(question: str, documents: list[dict], history: Optional[list[dict]] = None) -> list[dict]:
    context = "\n\n".join(f"[{i + 1}] ({doc['source']})\n{doc['content']}" for i, doc in enumerate(documents))
    user_message = f"Контекст:\n{context}\n\nПитання: {question}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_message})
    return messages


def call_openrouter(messages: list[dict]) -> str:
    response = httpx.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={"model": settings.openrouter_model, "messages": messages},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def answer_question(question: str, documents: list[dict], history: Optional[list[dict]] = None) -> dict:
    messages = build_prompt(question, documents, history)
    answer = call_openrouter(messages)
    sources = [{"source": doc["source"], "content": doc["content"]} for doc in documents]
    return {"answer": answer, "sources": sources}
