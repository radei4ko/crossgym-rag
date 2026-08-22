import re
from typing import Optional

import httpx

from src.config import settings

# Tokens like Instagram handles ("crossgym_baza_team", "@d__sanya__72") that contain
# an underscore. These are exactly the strings an LLM tends to mangle by treating
# "_" as markdown emphasis syntax and stripping it during generation.
_IDENTIFIER_TOKEN_RE = re.compile(r"@?[A-Za-z0-9][A-Za-z0-9_.]*")

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


def _extract_underscored_identifiers(text: str) -> set[str]:
    """Return tokens in `text` that contain an underscore."""
    return {token for token in _IDENTIFIER_TOKEN_RE.findall(text) if "_" in token}


def _restore_mangled_identifiers(answer: str, documents: list[dict]) -> str:
    """Deterministically repair identifiers whose underscores the model stripped.

    A system-prompt instruction asking the model to leave underscores in
    identifiers untouched is inherently probabilistic — in production it still
    intermittently rendered "crossgym_baza_team" as "crossgymbazateam" (the model
    pattern-matching "_word_" as markdown italics and "cleaning" it). Wording the
    prompt more strongly does not make a probabilistic model deterministic, so
    this guarantees correctness in code instead: for every underscore-containing
    identifier found verbatim in the retrieved context, if the model's answer
    contains the same string with underscores stripped (and not the correct
    string itself), substitute the correct verbatim identifier back in.
    """
    for doc in documents:
        for identifier in _extract_underscored_identifiers(doc.get("content", "")):
            if identifier in answer:
                continue  # model already reproduced it correctly

            bare = identifier.lstrip("@")
            mangled_candidates = {bare.replace("_", "")}
            if identifier.startswith("@"):
                mangled_candidates.add("@" + bare.replace("_", ""))

            for mangled in mangled_candidates:
                stripped_core = mangled.lstrip("@")
                if stripped_core == bare or len(stripped_core) < 3:
                    continue  # nothing was actually stripped, or too short to match safely
                if mangled in answer:
                    answer = answer.replace(mangled, identifier)
                    break
    return answer


_BOLD_MARKER_RE = re.compile(r"\*\*(.+?)\*\*")


def _strip_bold_markers(text: str) -> str:
    """Remove literal "**bold**" markdown wrapping from the answer.

    The Telegram send step uses parse_mode=HTML (see n8n workflow), not Markdown —
    HTML mode never touches underscores, which is what actually fixes the
    identifier-mangling bug (a Markdown parse_mode treats "_word_" as italics and
    eats the underscores; HTML mode has no such rule). But the model still often
    wraps key facts in "**...**", and in HTML mode those asterisks are inert
    literal characters — they'd show up as stray "**" in the chat instead of
    being rendered as bold. Stripped here rather than relying on a prompt
    instruction not to use bold, for the same reason identifiers are restored in
    code: a probabilistic model won't reliably obey a stylistic "don't" forever.
    """
    return _BOLD_MARKER_RE.sub(r"\1", text)


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
    answer = _restore_mangled_identifiers(answer, documents)
    answer = _strip_bold_markers(answer)
    sources = [{"source": doc["source"], "content": doc["content"]} for doc in documents]
    return {"answer": answer, "sources": sources}
