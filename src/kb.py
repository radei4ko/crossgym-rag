"""Turn data/kb/*.md into retrieval chunks.

Why not the generic length-based chunker: it cut "## Heading" away from its body, split
one trainer over two chunks, produced byte-identical chunks for halls with the same
price list, and left most chunks without the hall name, so the model had to guess the
hall from a file slug ("skhidnyi" -> "Схидный", "socmisto" -> "Соцгород").

Here every chunk is one logical unit (one trainer, one price section, one hall's
contacts) and starts with a header naming the hall and the topic. Trainer chunks are
also made symmetric: when a trainer's text says they also work in another hall, a
chunk for that hall is generated so "who works at 95 квартал" is answerable from that
hall's own file. The markdown files stay the single source of truth.
"""

import re
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"

CLUB_NAMES = {
    "socmisto": "Соцмісто",
    "skhidnyi": "Східний-2",
    "95kvartal": "95 квартал",
    "vechirniy": "Вечірній",
}
TYPE_LABELS = {
    "locations": "Адреса і графік",
    "pricing": "Ціни",
    "trainers": "Тренери",
}
# Halls with an identical price list (verified by tests/test_kb.py).
PRICE_GROUPS = [("socmisto", "95kvartal"), ("skhidnyi", "vechirniy")]

_HALL_MENTIONS = {
    "95kvartal": re.compile(r"95\s*квартал", re.I),
    "skhidnyi": re.compile(r"Східн(ий|ому|ого)[\s-]*2", re.I),
    "vechirniy": re.compile(r"Вечірн", re.I),
    "socmisto": re.compile(r"Соцміст", re.I),
}
_ALSO_SENTENCE_RE = re.compile(r"\s*Також (?:проводить тренування|працює) в залі [^.\n]*\.", re.I)
_HANDLE_RE = re.compile(r"Instagram:\s*(@[\w.]+)", re.I)

# keyword -> searchable tag (adds English synonyms so English questions can match too)
_TAGS = [
    (r"бокс", "бокс / boxing"),
    (r"пауерліфтинг|жим[уа]? лежачи|станов", "пауерліфтинг / powerlifting"),
    (r"армреслінг|армрестлинг", "армреслінг / arm wrestling"),
    (r"менс-?фізик|менс-?фізік|men's physique", "менс-фізік / men's physique"),
    (r"бікіні", "фітнес-бікіні / fitness bikini"),
    (r"\btrx\b|петл[іи]", "TRX"),
    (r"стретчинг|stretching", "стретчинг / stretching"),
    (r"crossfit", "CrossFit"),
    (r"функціонал|functional", "функціональні тренування / functional training"),
    (r"нутриціолог|дієтолог|харчування", "харчування / nutrition"),
    (r"діт[иь]|дитяч", "діти / kids"),
    (r"тяжк\w+ атлетик|важк\w+ атлетик", "важка атлетика / weightlifting"),
    (r"масаж", "масаж / massage"),
    (r"бодібілдинг", "бодібілдинг / bodybuilding"),
    (r"групов", "групові тренування / group training"),
]


def _header(club: str, topic: str, extra: str = "") -> str:
    return f"[Зал: {CLUB_NAMES[club]} | {topic}{(' | ' + extra) if extra else ''}]"


def _sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(intro_before_first_##, [(title, body), ...])"""
    parts = re.split(r"\n(?=## )", text.strip())
    intro = parts[0] if not parts[0].startswith("## ") else ""
    sections = []
    for part in parts if parts[0].startswith("## ") else parts[1:]:
        title, _, body = part.partition("\n")
        sections.append((title.removeprefix("## ").strip(), body.strip()))
    return intro.strip(), sections


def _tags(text: str) -> str:
    found = []
    for pattern, tag in _TAGS:
        if re.search(pattern, text, re.I) and tag not in found:
            found.append(tag)
    return ("Теги: " + "; ".join(found)) if found else ""


def _other_halls(text: str, own: str) -> list[str]:
    return [club for club, rx in _HALL_MENTIONS.items() if club != own and rx.search(text)]


def _split_stem(stem: str) -> tuple[str, str]:
    topic, _, club = stem.partition("_")
    return topic, club


def _parse_trainers(stem: str, text: str) -> tuple[str, list[dict]]:
    _topic, club = _split_stem(stem)
    intro, sections = _sections(text)
    intro_lines = [ln for ln in intro.splitlines() if ln and not ln.startswith("#")]
    records = []
    for title, body in sections:
        handle = _HANDLE_RE.search(body)
        records.append(
            {
                "name": title,
                "body": body,
                "club": club,
                "handle": handle.group(1).lower() if handle else None,
                "also": _other_halls(body, club),
            }
        )
    return " ".join(intro_lines), records


def _trainer_chunks() -> dict[str, list[str]]:
    """{stem: [chunk, ...]} for every trainers_* file, with cross-hall entries added."""
    parsed = {}
    for path in sorted(KB_DIR.glob("trainers_*.md")):
        intro, records = _parse_trainers(path.stem, path.read_text(encoding="utf-8"))
        parsed[path.stem] = (intro, records)

    halls_by_handle: dict[str, set[str]] = {}
    origin_by_handle: dict[str, dict] = {}
    for _stem, (_intro, records) in parsed.items():
        for rec in records:
            if not rec["handle"]:
                continue
            halls_by_handle.setdefault(rec["handle"], set()).add(rec["club"])
            halls_by_handle[rec["handle"]].update(rec["also"])
            origin_by_handle.setdefault(rec["handle"], rec)

    chunks: dict[str, list[str]] = {}
    for stem, (intro, records) in parsed.items():
        _topic, club = _split_stem(stem)
        out = []
        if intro:
            out.append(f"{_header(club, 'Тренери')}\n{intro}")
        present = {r["handle"] for r in records if r["handle"]}
        for rec in records:
            halls = sorted(halls_by_handle.get(rec["handle"], {club}) if rec["handle"] else {club})
            lines = [f"{_header(club, 'Тренери')}", f"## {rec['name']}", rec["body"]]
            if len(halls) > 1:
                lines.append("Працює в залах: " + ", ".join(CLUB_NAMES[h] for h in halls) + ".")
            tags = _tags(rec["body"])
            if tags:
                lines.append(tags)
            out.append("\n".join(lines))
        # trainers whose own file says they also work here, but who are missing from this hall's file
        for handle, halls in halls_by_handle.items():
            if club in halls and handle not in present:
                src = origin_by_handle[handle]
                body = _ALSO_SENTENCE_RE.sub("", src["body"]).strip()
                lines = [
                    _header(club, "Тренери"),
                    f"## {src['name']}",
                    body,
                    f"Основне місце роботи: зал {CLUB_NAMES[src['club']]}. Працює в залах: "
                    + ", ".join(CLUB_NAMES[h] for h in sorted(halls)) + ".",
                ]
                tags = _tags(src["body"])
                if tags:
                    lines.append(tags)
                out.append("\n".join(lines))
        chunks[stem] = out
    return chunks


def _pricing_chunks(stem: str, text: str) -> list[str]:
    _topic, club = _split_stem(stem)
    _intro, sections = _sections(text)
    chunks = []
    for index, (title, body) in enumerate(sections):
        chunks.append(f"{_header(club, 'Ціни', title)}\n{body}")
    return chunks


def pricing_overview() -> str:
    """One chunk with every hall's price list, grouped by identical price sections."""
    blocks = ["[Ціни по залах — зведення]"]
    for group in PRICE_GROUPS:
        names = " і ".join(CLUB_NAMES[c] for c in group)
        text = (KB_DIR / f"pricing_{group[0]}.md").read_text(encoding="utf-8")
        _intro, sections = _sections(text)
        lines = []
        for title, body in sections:
            body = "\n".join(
                ln for ln in body.splitlines() if not ln.startswith("Абонемент діє в межах одного клубу")
            ).strip()
            lines.append(f"{title}:\n{body}")
        blocks.append(f"Зали {names} (однаковий прайс):\n" + "\n".join(lines))
    blocks.append("Абонемент діє лише в межах одного залу, в якому він куплений; не є мережевим між залами.")
    return "\n\n".join(blocks)


def build_chunks() -> list[dict]:
    """[{"source": stem, "content": text}, ...] in document order, ready for embedding."""
    trainers = _trainer_chunks()
    rows: list[dict] = []
    for path in sorted(KB_DIR.glob("*.md")):
        stem = path.stem
        text = path.read_text(encoding="utf-8")
        topic, club = _split_stem(stem)
        if topic == "trainers":
            contents = trainers[stem]
        elif topic == "pricing":
            contents = _pricing_chunks(stem, text)
        elif topic == "locations":
            contents = [f"{_header(club, 'Адреса і графік')}\n{text.strip()}"]
        else:  # policy.md
            contents = [f"[Правила запису та скасування]\n{text.strip()}"]
        rows += [{"source": stem, "content": c} for c in contents]
    rows.append({"source": "pricing_overview", "content": pricing_overview()})
    return rows
