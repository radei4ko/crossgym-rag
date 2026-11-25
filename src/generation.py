import logging
import re
import time
from typing import Optional

import httpx

from src.config import settings
from src.lang import LANG_NAMES, UK, answer_language_ok

logger = logging.getLogger("crossgym.generation")

# Tokens like Instagram handles ("crossgym_baza_team", "@d__sanya__72") that contain
# an underscore. These are exactly the strings an LLM tends to mangle by treating
# "_" as markdown emphasis syntax and stripping it during generation.
_IDENTIFIER_TOKEN_RE = re.compile(r"@?[A-Za-z0-9][A-Za-z0-9_.]*")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Short on purpose. Everything that can be decided in code is NOT in here: reply
# language (src/lang.py), greetings (src/pipeline.py), source numbering (none),
# identifier/markdown clean-up (post-processing below).
SYSTEM_PROMPT = (
    "You are the Telegram assistant of the CrossGYM gym chain in Kryvyi Rih.\n"
    "Answer ONLY from the Context. If the Context says nothing about what the user asks (a pool, "
    "parking, a bar, payment methods ...), reply in one or two sentences that you do not have that "
    "information. \"No information\" is not the same as \"no\": NEVER say \"we don't have X\" or "
    "\"there is no X\" unless the Context states it explicitly. Never invent phone numbers, prices, "
    "addresses or handles; do not suggest a phone number that is not written in the Context.\n\n"
    "Style: a short, natural chat message in plain text. No markdown headings and no bold. "
    "Use \"- \" lists only when listing several items. Do not start with a greeting or a "
    "self-introduction and do not comment on the language: answer the question straight away.\n"
    "Facts: copy names, prices, phone numbers, addresses and @handles exactly as written in the "
    "Context, keeping underscores in handles. Use hall names exactly as they appear in the Context "
    "headers.\n"
    "Lists: when asked who / which / all, go through EVERY entry in the Context and list every "
    "match; do not stop after the first ones. If the user gives only a first name (for example "
    "\"Олександр\"), every entry that starts with or contains that name is a match, even when one "
    "entry is just \"Олександр\" and others are \"Олександр Ляшенко\" or \"Олександр Денисов\": "
    "list ALL of them with hall and @handle and never pick one. The same person (same @handle) can "
    "appear under several halls: list them once and mention all their halls.\n"
    "Halls: if the user did not name a hall and the values differ between halls, give each variant "
    "labelled by hall (merge halls with identical values) instead of asking which hall.\n"
    "Ask a clarifying question only when several different people match equally (for example "
    "several trainers with the same first name), and then list them.\n"
    "Follow-ups: \"he / she / it / there\" refer to the person or hall from the earlier turns.\n"
    "You may add closely related details that are in the Context (conditions, neighbouring options), "
    "but never anything that is not."
)

SOURCE_LABELS = {
    "policy": "Правила запису та скасування",
    "pricing_overview": "Ціни по залах — зведення",
}


def _doc_block(doc: dict) -> str:
    content = doc["content"]
    if content.lstrip().startswith("["):
        return content  # indexed chunks already carry a "[Зал: ... | ...]" header
    return f"[{SOURCE_LABELS.get(doc['source'], doc['source'])}]\n{content}"


def language_directive(lang: str, strict: bool = False) -> str:
    name = LANG_NAMES.get(lang, "Ukrainian")
    text = (
        f"Reply language: {name}. Write the entire reply in {name}. The Context is in Ukrainian: "
        "translate the facts when needed, but keep names, hall names, phone numbers and @handles "
        "exactly as in the Context. Never mention which language you are using."
    )
    if lang == UK:
        text += " Use correct standard Ukrainian."
    else:
        text += " Do not transliterate or translate person names."
    if lang == "en":
        text += (" Write hall names exactly like this: Sotsmisto, Skhidnyi-2, 95 Kvartal, Vechirniy"
                 " (for Соцмісто, Східний-2, 95 квартал, Вечірній).")
    if strict:
        text = f"IMPORTANT: your previous reply was in the wrong language. {text}"
    return text


def build_prompt(
    question: str,
    documents: list[dict],
    history: Optional[list[dict]] = None,
    lang: Optional[str] = None,
    strict: bool = False,
) -> list[dict]:
    context = "\n\n".join(_doc_block(doc) for doc in documents)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    if lang:
        user_message += f"\n\n{language_directive(lang, strict)}"
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


_CITATION_MARKER_RE = re.compile(r"\s?\[\d{1,2}(?:\s*,\s*\d{1,2})*\]")


def _strip_citation_markers(text: str) -> str:
    """Remove leaked "[1]", "[2]"-style source index markers from the answer.

    build_prompt() numbers each context chunk ("[3] (source)\\n...") purely so the
    model can tell chunks apart internally. That numbering is not meant to reach
    the user, but with more chunks in context (match_count raised 5->14) the model
    started imitating an academic citation style and echoing "[2]" inline. A
    prompt instruction not to do this helps but, per this project's established
    pattern (see _strip_bold_markers), a probabilistic model won't reliably obey a
    stylistic "don't" forever — so this is stripped deterministically too.
    """
    return _CITATION_MARKER_RE.sub("", text)


_LANGUAGE_PREAMBLE_RE = re.compile(
    r"^\s*(?:питання\s+(?:російською|англійською|українською)|"
    r"(?:okay|ok|sure|of course)[,!]?\s+(?:i(?:'ll| will)|let me)\s+(?:answer|reply|continue|respond|switch)|"
    r"i(?:'ll| will)\s+(?:answer|reply|continue|respond)|"
    r"(?:добре|хорошо|звичайно|конечно)[,!]?\s+(?:відповідаю|отвечаю|відповім|отвечу)|"
    r"хорошо,\s*отвечу)[^.!\n]*[.!]\s*",
    re.I,
)


def _strip_language_preamble(text: str) -> str:
    """Drop a leading "I will answer in English." style sentence if the model still adds one."""
    stripped = _LANGUAGE_PREAMBLE_RE.sub("", text, count=1)
    return stripped if stripped.strip() else text


def call_openrouter(messages: list[dict]) -> str:
    payload: dict = {
        "model": settings.openrouter_model,
        "messages": messages,
        "max_tokens": settings.openrouter_max_tokens,
        "temperature": 0,  # factual extraction from a context, not creative writing
    }
    if settings.openrouter_reasoning_effort:
        payload["reasoning"] = {"effort": settings.openrouter_reasoning_effort}
    if settings.openrouter_fallback_model and settings.openrouter_fallback_model != settings.openrouter_model:
        payload["models"] = [settings.openrouter_model, settings.openrouter_fallback_model]
    for attempt in range(3):
        response = httpx.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json=payload,
            timeout=30.0,
        )
        if response.status_code in (429, 500, 502, 503) and attempt < 2:
            time.sleep(1.5 * (attempt + 1))  # transient upstream limit/outage: retry briefly
            continue
        break
    response.raise_for_status()
    data = response.json()
    # With a fallback configured OpenRouter may silently answer with another model: make it visible.
    logger.info("served_by=%s requested=%s", data.get("model"), settings.openrouter_model)
    return data["choices"][0]["message"]["content"]


_PHONE_RE = re.compile(r"(?<!\d)(?:\+?38)?[\s\-]?\(?0\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)")


def _phone_key(match: "re.Match") -> str:
    return re.sub(r"\D", "", match.group())[-10:]


def _remove_ungrounded_phones(answer: str, documents: list[dict]) -> str:
    """Drop any sentence that contains a phone number which is not in the retrieved context.

    A model asked to "suggest calling the hall" will happily invent a plausible number when
    the context has none (seen in testing: 097 123 45 67). A wrong phone number is the
    worst failure for a customer-facing bot, so this is enforced in code, not by the prompt."""
    known = {_phone_key(m) for d in documents for m in _PHONE_RE.finditer(d.get("content", ""))}

    def ungrounded(text: str) -> bool:
        return any(_phone_key(m) not in known for m in _PHONE_RE.finditer(text))

    if not ungrounded(answer):
        return answer
    sentences = re.findall(r"[^.!?\n]+[.!?]*[ \t]*\n?|\n", answer)
    kept = "".join(s for s in sentences if not ungrounded(s)).strip()
    return kept or _PHONE_RE.sub("", answer).strip()


def _postprocess(answer: str, documents: list[dict]) -> str:
    answer = _remove_ungrounded_phones(answer, documents)
    answer = _restore_mangled_identifiers(answer, documents)
    answer = _strip_bold_markers(answer)
    answer = _strip_citation_markers(answer)
    return _strip_language_preamble(answer).strip()


def answer_question(
    question: str,
    documents: list[dict],
    history: Optional[list[dict]] = None,
    lang: Optional[str] = None,
) -> dict:
    messages = build_prompt(question, documents, history, lang)
    answer = _postprocess(call_openrouter(messages), documents)
    if lang and not answer_language_ok(answer, lang):
        logger.warning("wrong reply language (wanted %s), retrying once", lang)
        retry = _postprocess(call_openrouter(build_prompt(question, documents, history, lang, strict=True)), documents)
        if answer_language_ok(retry, lang):
            answer = retry
    sources = [{"source": doc["source"], "content": doc["content"]} for doc in documents]
    return {"answer": answer, "sources": sources}
