"""Decide which knowledge-base files a question needs, in plain code.

The whole knowledge base is ~55 small chunks, so semantic scoring has nothing to
"rank": it only drops chunks at random and loses items from lists (3 boxing coaches
spread over 3 chunks). When the question names a hall and/or a topic we load exactly
those files in full and in document order. Hybrid search stays as the fallback for
questions that name neither (e.g. "Розкажи про Катерину Резвін").
"""

import re
from typing import Optional

from src.lang import clean_user_text

CLUB_NAMES = {
    "socmisto": "Соцмісто",
    "skhidnyi": "Східний-2",
    "95kvartal": "95 квартал",
    "vechirniy": "Вечірній",
}

_CLUB_PATTERNS = {
    "socmisto": r"соцміст|соцмист|соцгор|соц\.? ?міст|sotsmist|socmist|sotsmyst|sotsialn",
    "skhidnyi": r"східн|восточн|схидн|skhidn|shidn|vostochn|\beast(?:ern)?\b",
    "95kvartal": r"\b95\b|плаз|plaza|квартал|kvartal|gagarin|гагарін|гагарин",
    "vechirniy": r"вечірн|вечерн|vechirn|vecherni|\bevening\b",
}
_CLUB_RES = {key: re.compile(pattern, re.I) for key, pattern in _CLUB_PATTERNS.items()}

_TOPIC_PATTERNS = {
    "pricing": (
        r"ціна|ціни|цін[іу]|коштує|коштувати|вартіст|вартість|прайс|абонемент|разов|безліміт|"
        r"\bvip\b|доплат|скільки\b|сколько|стоит|стоимост|цена|цены|безлимит|\bprice|\bcost|"
        r"how much|membership|subscription|single visit|drop-?in|\bгрн\b|\buah\b"
    ),
    "trainers": (
        r"тренер|trainer|coach|інструктор|инструктор|instagram|інстаграм|инстаграм|інста\b|"
        r"хто веде|кто ведет|кто ведёт|who (?:teach|run|lead|train|coach)|викладає|"
        r"бокс|boxing|\btrx\b|стретчинг|stretching|crossfit|кроссфіт|кроссфит|кросфіт|"
        r"пауерліфт|powerlift|бікіні|bikini|нутриціолог|нутрициолог|дієтолог|диетолог|масаж|"
        r"массаж|армрес|армреслінг|менс-?фізик|менс-?физик|персональн|дитяч|детск|kids"
    ),
    "locations": (
        r"адрес|де знаходи|де ви\b|де зал|де розташ|где находит|где вы\b|address|where|телефон|"
        r"phone|номер|графік|график|години|часы работы|hours|schedule|open|відкрит|открыт|"
        r"працює зал|режим|до скільки|о котр|во сколько|локаці|location|як дістат|как добрат"
    ),
    "policy": (
        r"скасув|скасуй|відмін|отмен|cancel|правил|rules|policy|штраф|penalt|refund|поверн|"
        r"мережев|не можна відвідувати|в іншому залі|в другом зале|another (?:hall|club|gym)|"
        r"запис[аи]тись|записатися|записаться|how to book|book a"
    ),
}
_TOPIC_RES = {key: re.compile(pattern, re.I) for key, pattern in _TOPIC_PATTERNS.items()}

# question keyword -> tag written into trainer chunks by src/kb.py ("Теги: бокс / boxing; ...")
_SKILL_PATTERNS = [
    (r"бокс|boxing|\bbox\b", "boxing"),
    (r"\btrx\b|петл[іи]", "trx"),
    (r"стретчинг|stretching", "stretching"),
    (r"crossfit|кроссфіт|кроссфит|кросфіт", "crossfit"),
    (r"пауерліфт|powerlift", "powerlifting"),
    (r"бікіні|bikini", "bikini"),
    (r"нутриціолог|нутрициолог|дієтолог|диетолог|nutrition|nutritionist|dietitian", "nutrition"),
    (r"масаж|массаж|massage", "massage"),
    (r"армрес|arm.?wrestl", "arm wrestling"),
    (r"менс-?фіз|менс-?физ|physique", "physique"),
    (r"важк\w+ атлет|тяжк\w+ атлет|weightlift", "weightlifting"),
    (r"бодібілд|бодибилд|bodybuild", "bodybuilding"),
    (r"дитяч|детск|діт[иь]\b|\bkids\b|children", "kids"),
]
_SKILL_RES = [(re.compile(pattern, re.I), tag) for pattern, tag in _SKILL_PATTERNS]

# Words that mean "the thing we were just talking about".
FOLLOW_UP_RE = re.compile(
    r"\b(там|туди|туда|тут|її|його|їх|неї|нього|них|ее|её|его|их|there|his|her|its|their|"
    r"him|them|цей зал|ця тренер|этот зал|that one|the same)\b",
    re.I,
)


def detect_clubs(text: str) -> list[str]:
    found = [(match.start(), key) for key, rx in _CLUB_RES.items() if (match := rx.search(text))]
    return [key for _pos, key in sorted(found)]


def detect_skills(text: str) -> set[str]:
    return {tag for rx, tag in _SKILL_RES if rx.search(text)}


def detect_topics(text: str) -> set[str]:
    topics = {key for key, rx in _TOPIC_RES.items() if rx.search(text)}
    if detect_skills(text):
        topics.add("trainers")  # "box and aerobics" is a trainers question even without the word "trainer"
    return topics


_HANDLE_RE = re.compile(r"Instagram:\s*(@[\w.]+)", re.I)


def narrow_trainers(question: str, documents: list[dict]) -> list[dict]:
    """Keep only trainer chunks that carry the asked-for skill tag, and list a person once.

    "Who teaches boxing" must not hand the model 40 unrelated trainers to leaf through;
    a trainer working in two halls has one chunk per hall and would be listed twice."""
    skills = detect_skills(question)
    narrowed = []
    if skills:
        trainer_chunks = [d for d in documents if d["source"].startswith("trainers_")]
        matching = [d for d in trainer_chunks if any(tag in d["content"].lower() for tag in skills)]
        if matching:
            narrowed = [d for d in documents if not d["source"].startswith("trainers_") or d in matching]
    documents = narrowed or documents
    seen, unique = set(), []
    for doc in documents:
        match = _HANDLE_RE.search(doc["content"]) if doc["source"].startswith("trainers_") else None
        if match and match.group(1).lower() in seen:
            continue
        if match:
            seen.add(match.group(1).lower())
        unique.append(doc)
    return unique


def is_follow_up(question: str) -> bool:
    return bool(FOLLOW_UP_RE.search(question)) or len(question.split()) <= 2


def _club_from_history(history: list[dict]) -> list[str]:
    """Most recent hall(s) mentioned: the user's messages first, then the bot's answers."""
    for role in ("user", "assistant"):
        for turn in reversed(history):
            if turn.get("role") != role:
                continue
            clubs = detect_clubs(clean_user_text(turn.get("content", "")))
            if clubs:
                return clubs
    return []


def _club_sources(topic: str, clubs: list[str]) -> list[str]:
    return [f"{topic}_{club}" for club in clubs]


def select_sources(question: str, history: Optional[list[dict]] = None) -> Optional[list[str]]:
    """Source names (knowledge-base file stems) to load, or None to fall back to search."""
    history = history or []
    clubs = detect_clubs(question)
    topics = detect_topics(question)

    if not clubs and history and is_follow_up(question):
        clubs = _club_from_history(history)

    if not clubs and not topics:
        return None

    sources: list[str] = []
    if not topics:
        # Hall named, topic not ("а в Східному-2?", "розкажи про Соцмісто"): everything about that hall.
        for club in clubs:
            sources += _club_sources("locations", [club]) + _club_sources("pricing", [club])
            sources += _club_sources("trainers", [club])
        return sources + ["policy"]

    for topic in ("locations", "pricing", "trainers"):
        if topic not in topics:
            continue
        if clubs:
            sources += _club_sources(topic, clubs)
        elif topic == "pricing":
            sources.append("pricing_overview")
        else:
            sources += _club_sources(topic, list(CLUB_NAMES))
    if "policy" in topics:
        sources.append("policy")
    return sources or None
