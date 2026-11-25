"""Deterministic reply-language selection (uk / ru / en).

Choosing the reply language used to be left to the LLM through rules in the system
prompt. Those rules conflicted with each other and the surrounding text is almost
entirely Ukrainian, so the model drifted back to Ukrainian on short messages. The
decision is made here in code instead and handed to the model as a single imperative
line at the end of the last user message.
"""

import re
from typing import Optional

UK, RU, EN = "uk", "ru", "en"
LANG_NAMES = {UK: "Ukrainian", RU: "Russian", EN: "English"}

_LETTERS = "A-Za-zА-Яа-яІіЇїЄєҐґЁёЪъЫыЭэ"
_WORD_RE = re.compile(rf"[{_LETTERS}][{_LETTERS}'’ʼ\-]*")
_CYR_RE = re.compile(r"[А-Яа-яІіЇїЄєҐґЁёЪъЫыЭэ]")
_LAT_RE = re.compile(r"[A-Za-z]")
_UK_LETTER_RE = re.compile(r"[іїєґ]")
_RU_LETTER_RE = re.compile(r"[ыэёъ]")

# Function words that exist in only one of the two languages. Words shared by both
# (тренер, зал, абонемент, телефон ...) are deliberately left out.
UK_WORDS = {
    "що", "як", "де", "коли", "скільки", "коштує", "є", "можна", "чи", "мені", "який", "яка",
    "яке", "які", "хто", "цей", "ця", "також", "дякую", "будь", "ласка", "будні", "вихідні",
    "залі", "працює", "працюють", "тренування", "ціна", "ціни", "безлімітний", "разове",
    "запис", "скасувати", "привіт", "добрий", "потрібно", "записатися", "записатись", "вас",
    "ваш", "ваша", "ваші", "мене", "тобі", "тебе", "він", "вона", "його", "її", "їх", "там",
    "тут", "дуже", "зараз", "треба", "хочу", "розкажіть", "підкажіть", "підкажи", "скажи",
    "розкажи", "будь-ласка", "доброго", "вітаю", "адреса", "графік", "години", "відкритий",
}
RU_WORDS = {
    "что", "как", "где", "когда", "сколько", "стоит", "есть", "можно", "ли", "мне", "какой",
    "какая", "какое", "какие", "кто", "этот", "эта", "также", "тоже", "спасибо", "пожалуйста",
    "будни", "выходные", "зале", "работает", "работают", "тренировка", "тренировки",
    "тренировку", "цена", "цены", "безлимитный", "разовое", "разовая", "запись", "отменить",
    "привет", "здравствуйте", "добрый", "нужно", "записаться", "вас", "ваш", "ваша", "ваши",
    "меня", "тебе", "тебя", "он", "она", "его", "ее", "её", "их", "там", "тут", "очень",
    "сейчас", "надо", "хочу", "расскажите", "подскажите", "подскажи", "скажи", "расскажи",
    "доброе", "здравствуй", "адрес", "график", "часы", "открыт", "вечернем", "восточном",
}
_SHARED_WORDS = UK_WORDS & RU_WORDS
UK_WORDS -= _SHARED_WORDS
RU_WORDS -= _SHARED_WORDS

# A lone Latin word is only treated as English when it is a clearly English one.
EN_WORDS = {
    "hello", "hi", "hey", "price", "prices", "coach", "coaches", "trainer", "trainers",
    "schedule", "address", "phone", "hours", "help", "thanks", "thank", "yes", "no", "ok",
    "okay", "pool", "gym", "boxing", "membership",
}
EN_STOP = {
    "the", "is", "are", "at", "and", "we", "you", "can", "in", "of", "to", "from", "for", "on",
    "with", "our", "your", "has", "have", "it", "this", "that", "by", "there", "not", "do",
    "does", "what", "who", "how", "when", "where",
}

# Stems of "<language>" in each language, used for explicit "answer in X" requests.
_LANG_STEMS = {
    EN: ("english", "англійськ", "англійск", "английск", "по-англ", "по-англійськи", "по-английски"),
    RU: ("russian", "русск", "російськ", "по-русск", "по-російськ", "по-рус"),
    UK: ("ukrainian", "українськ", "украинск", "по-украин", "по-українськ", "по-укр"),
}
_REQUEST_FILLER = {
    "please", "pls", "plz", "ok", "okay", "can", "could", "you", "do", "speak", "talk",
    "answer", "reply", "write", "respond", "switch", "continue", "in", "to", "a", "the",
    "and", "only", "now", "from", "на", "по", "чи", "а", "и", "і", "та", "можеш", "можете",
    "можно", "можна", "будь", "ласка", "пожалуйста", "прошу", "мені", "мне", "давай",
    "давайте", "тепер", "теперь", "далі", "дальше", "говори", "говорити", "говорите",
    "відповідай", "відповідь", "відповідайте", "ответь", "отвечай", "отвечайте", "пиши",
    "пишіть", "пишите", "переключись", "переходь", "переходьте", "розмовляти", "общаться",
    "спілкуватись", "спілкуватися", "dalej", "i", "me", "with", "language", "мова", "языке",
    "мовою", "языком", "язык", "словами",
}
_REQUEST_VERB_RE = re.compile(
    r"(speak|talk|answer|reply|write|respond|switch|continue|говор|відповід|ответ|отвеч|пиш|"
    r"переключ|переход|можеш|можете|можно|можна)"
)
_QUESTION_PREFIX_RE = re.compile(r"^.*(?:Питання|Question)\s*:\s*", re.S)


def clean_user_text(text: str) -> str:
    """History written by older code wrapped messages as "Контекст:...Питання: <q>"."""
    return _QUESTION_PREFIX_RE.sub("", text or "").strip()


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _scores(words: list[str], skip_proper: bool) -> tuple[int, int]:
    """(ukrainian_score, russian_score). Capitalised mid-sentence words (hall and
    person names such as "Соцмісто", "Вечірній") are skipped: they carry the hall's
    own spelling, not the language the user is writing in."""
    uk = ru = 0
    for index, word in enumerate(words):
        if skip_proper and index > 0 and word[0].isupper() and not word.isupper():
            continue
        lowered = word.lower().strip("-'’ʼ")
        if lowered in UK_WORDS or _UK_LETTER_RE.search(lowered):
            uk += 1
        if lowered in RU_WORDS or _RU_LETTER_RE.search(lowered):
            ru += 1
    return uk, ru


def detect(text: str) -> Optional[str]:
    """Language of a user message, or None when it is too short or ambiguous."""
    words = _tokens(text)
    if not words:
        return None
    cyr_letters = len(_CYR_RE.findall(text))
    lat_letters = len(_LAT_RE.findall(text))
    latin_words = [w for w in words if _LAT_RE.search(w)]

    if cyr_letters == 0:
        if len(latin_words) >= 2 or (latin_words and latin_words[0].lower() in EN_WORDS):
            return EN
        return None
    if lat_letters > 2 * cyr_letters and len(latin_words) >= 2:
        return EN

    uk, ru = _scores(words, skip_proper=True)
    if uk > ru:
        return UK
    if ru > uk:
        return RU
    return None


def _language_request(text: str) -> tuple[Optional[str], bool]:
    """(requested_language, is_pure) where pure means the message is only the request."""
    """"english?", "talk in english please", "можеш по-російськи?" -> that language.

    A language word inside an ordinary question ("Хто викладає англійську?",
    "Who teaches English?") is not a request to switch."""
    lowered = text.lower().strip()
    words = _tokens(lowered)
    requested = None
    lang_word_indexes = set()
    for index, word in enumerate(words):
        for lang, stems in _LANG_STEMS.items():
            if any(word.startswith(stem) for stem in stems):
                requested = requested or lang
                lang_word_indexes.add(index)
    if not requested:
        return None, False
    remaining = [w for i, w in enumerate(words) if i not in lang_word_indexes and w not in _REQUEST_FILLER]
    if not remaining:
        return requested, True
    if len(words) <= 8 and _REQUEST_VERB_RE.search(lowered):
        return requested, False
    return None, False


def explicit_request(text: str) -> Optional[str]:
    """"english?", "talk in english please", "можеш по-російськи?" -> that language.

    A language word inside an ordinary question ("Хто викладає англійську?",
    "Who teaches English?") is not a request to switch."""
    return _language_request(text)[0]


def pure_language_request(text: str) -> Optional[str]:
    """Only a request to switch ("english?", "по-русски, пожалуйста"), nothing to look up."""
    lang, pure = _language_request(text)
    return lang if pure else None


def resolve_language(question: str, history: Optional[list[dict]] = None) -> str:
    """Walk the user's messages in order; an explicit request or a confident message
    sets the language, an ambiguous one ("box", "ок", "VIP") keeps the previous one.
    Default is Ukrainian."""
    lang: Optional[str] = None
    user_messages = [clean_user_text(t.get("content", "")) for t in (history or []) if t.get("role") == "user"]
    for message in user_messages + [question]:
        requested = explicit_request(message)
        if requested:
            lang = requested
            continue
        detected = detect(message)
        if detected:
            lang = detected
    return lang or UK


def answer_language_ok(text: str, lang: str) -> bool:
    """Cheap check that a generated answer is in the language we asked for."""
    words = _tokens(text)
    if len(words) < 6:
        return True
    cyr = len(_CYR_RE.findall(text))
    lat = len(_LAT_RE.findall(text))
    cyr_ratio = cyr / max(1, cyr + lat)
    stop_ratio = sum(w.lower() in EN_STOP for w in words) / len(words)
    looks_english = stop_ratio >= 0.12 or cyr_ratio < 0.25
    if lang == EN:
        return looks_english
    if looks_english and cyr_ratio < 0.6:
        return False
    uk, ru = _scores(words, skip_proper=False)
    if lang == UK:
        return not ru >= uk + 2
    return not uk >= ru + 2
