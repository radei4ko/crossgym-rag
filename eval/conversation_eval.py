"""Multi-turn, multi-language scenarios that the one-shot question eval cannot see.

Every scenario is run k times and counts as passed only if ALL runs pass (pass^k): a
single lucky run proves nothing for a probabilistic system. Scenarios go through
src.pipeline.respond(), i.e. exactly what /ask does (routing, language, model, clean-up).

    python -m eval.conversation_eval            # k=3
    python -m eval.conversation_eval --k 1 --only EN-sticky,RU-price
"""

import argparse
import re
from pathlib import Path

from src.pipeline import respond

KB = Path(__file__).resolve().parent.parent / "data" / "kb"

# ---- language checks: deliberately NOT the code under test (src/lang.py) ----
_EN_STOP = {"the", "is", "are", "at", "and", "we", "you", "can", "in", "of", "to", "for", "on", "with", "has", "have", "it", "this", "there", "not", "our"}
META_RE = re.compile(r"(i will (answer|reply|continue)|i'll (answer|reply)|питання (російською|англійською)|відповідаю (російською|англійською)|отвечаю по-)", re.I)
MUTANTS = ["Схидн", "Соцгород", "Восточн", "Sotsgorod", "Sozhm", "Skhidnyj"]


def _words(text):
    return re.findall(r"[A-Za-zА-Яа-яІіЇїЄєҐґЁё']+", text)


def is_english(text):
    plain = re.sub(r"@[\w.]+", "", text)
    cyr = len(re.findall(r"[А-Яа-яІіЇїЄєҐґЁё]", plain))
    lat = len(re.findall(r"[A-Za-z]", plain))
    return lat >= 1.5 * cyr and lat > 0  # Cyrillic hall/trainer names are kept as written


def is_russian(text):
    low = text.lower()
    ru_marks = len(re.findall(r"[ыэёъ]", low)) + sum(w in low for w in (" стоит", " есть", " также", "тренировк", " зале"))
    uk_lower = len([w for w in re.findall(r"\b[а-яіїєґ'’]*[іїєґ][а-яіїєґ'’]*\b", text) if w[0].islower()])  # not hall names
    if len(text) <= 80:
        return uk_lower == 0 and not re.search(r"[A-Za-z]{4,}", re.sub(r"@[\w.]+", "", text))
    return ru_marks >= 1 and uk_lower <= 1 + ru_marks


def is_ukrainian(text):
    return not re.search(r"[ыэёъ]", text.lower()) and bool(re.search(r"[іїєґ]", text.lower()))


def handles(path):
    return set(h.lower() for h in re.findall(r"Instagram:\s*(@[\w.]+)", (KB / path).read_text(encoding="utf-8")))


def has_all(answer, *needles):
    low = answer.lower()
    return [n for n in needles if n.lower() not in low]


# ---- scenarios: (name, [turn, ...]); a turn is (user_message, check(answer) -> error|None) ----
def need(*needles):
    return lambda a: (f"missing {has_all(a, *needles)}" if has_all(a, *needles) else None)


def lang_is(fn, label, *needles):
    def check(a):
        if not fn(a):
            return f"not {label}: {a[:90]!r}"
        missing = has_all(a, *needles)
        return f"missing {missing}" if missing else None
    return check


def no_meta(a):
    return f"meta phrase: {a[:90]!r}" if META_RE.search(a) else None


def any_of(*checks):
    return lambda a: next((e for e in (c(a) for c in checks) if e is None), None) or "no alternative matched: " + a[:120]


def clarifying_question(a):
    return "asked which hall instead of answering" if re.search(r"який зал|уточніть|which (hall|gym)|какой зал", a.lower()) else None


BOXERS = (["@nik_pro__", "миколай", "mykola", "nikolai"],
          ["@slavatraining_", "ячеслав", "vyacheslav", "viacheslav", "slava"],
          ["@markin_andrey_kr", "маркін", "markin"])


def _all_groups(answer, groups):
    low = answer.lower()
    return [g[0] for g in groups if not any(alt in low for alt in g)]


def boxing_all_three(a):
    missing = _all_groups(a, BOXERS)
    if missing:
        return f"boxing coaches incomplete, missing {missing}: {a[:140]!r}"
    dupes = [h for h in ("@nik_pro__", "@slavatraining_", "@markin_andrey_kr") if a.lower().count(h) > 1]
    return f"person listed twice: {dupes}" if dupes else None


def ambiguous_olexandr(a):
    found = [h for h in ("@d__sanya__72", "@gym_time_your_time", "@sashadenisov6318") if h in a]
    return None if len(found) >= 2 or "?" in a else "picked one Олександр silently"


def english_refusal(a):
    low = a.lower()
    ok = any(m in low for m in ("no information", "don't have", "do not have", "not mentioned", "no info", "doesn't mention", "not in"))
    return None if ok and not low.startswith("yes") else f"not a proper English refusal: {a[:100]!r}"


def short(a):
    return None if len(a) <= 300 else "greeting too long"


def not_greeting_start(a):
    return "starts with a greeting" if re.match(r"\s*(привіт|вітаю|добрий|здравствуй|привет|hello|hi)\b", a.lower()) else None


SKHIDNYI = [["найкус", "@latinas.coach"], ["вінокурова", "@vinokurova_olya_lya"], ["кабанець", "@katerynkafit"],
            ["олександр", "@d__sanya__72"], ["морозова", "@viktorija_trojan"], ["наталія", "@nataliorlovaa7"],
            ["аліна", "@alinostra_0504"], ["едуард", "@eduard.luts"], ["павло", "@pasha__physique"]]


def all_skhidnyi_trainers(a):
    missing = _all_groups(a, SKHIDNYI)
    return None if not missing else f"missing {len(missing)}/9: {missing}"


KB_PHONES = {re.sub(r"\D", "", p)[-10:] for f in KB.glob("locations_*.md") for p in re.findall(r"Телефон:\s*([\d +\-()]+)", f.read_text(encoding="utf-8"))}
PHONE_RE = re.compile(r"(?<!\d)(?:\+?38)?[\s\-]?\(?0\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)")
NO_INFO = ("немає інформації", "не маю", "не вказано", "не містить", "інформації про", "нет информации", "no information", "don't have", "не знайдено", "не згадується")


def honest_refusal(a):
    low = a.lower()
    invented = [m.group() for m in PHONE_RE.finditer(a) if re.sub(r"\D", "", m.group())[-10:] not in KB_PHONES]
    if invented:
        return f"invented phone {invented}"
    if re.search(r"(у нас|у нашому залі|в нас|у нас в залі) (немає|нема) ", low) and not any(m in low for m in NO_INFO):
        return f"asserted absence instead of 'no information': {a[:100]!r}"
    return None if any(m in low for m in NO_INFO) else f"no refusal marker: {a[:100]!r}"


SCENARIOS = {
    "EN-sticky": [
        ("english?", lang_is(is_english, "English")),
        ("did you have trainer who teach box?", lambda a: lang_is(is_english, "English")(a) or boxing_all_three(a)),
        ("box and aerobics", lambda a: no_meta(a) or lang_is(is_english, "English")(a) or boxing_all_three(a)),
        ("what time does Socmisto open on weekdays?", lang_is(is_english, "English", "07:00")),
    ],
    "RU-price": [("Сколько стоит разовое занятие в Вечернем?", lambda a: no_meta(a) or lang_is(is_russian, "Russian", "120")(a))],
    "RU-phone": [("Какой телефон у зала Восточный-2?", lang_is(is_russian, "Russian", "098 655 00 99"))],
    "switch-en-to-ru": [
        ("english?", lang_is(is_english, "English")),
        ("А адрес Соцмиста?", lambda a: lang_is(is_russian, "Russian")(a) or (None if re.search("Тіл?ьги", a) else "address missing")),
    ],
    "EN-boxing-complete": [("Who coaches boxing at Sotsmisto?", lambda a: lang_is(is_english, "English")(a) or boxing_all_three(a))],
    "UK-boxing-complete": [("Хто з тренерів Соцмісто веде бокс?", boxing_all_three)],
    "complete-list-skhidnyi": [("Перелічи всіх тренерів залу Східний-2", all_skhidnyi_trainers)],
    "price-no-hall": [("Скільки коштує разове тренування?", lambda a: clarifying_question(a) or (None if not has_all(a, "200", "120") else f"both tariffs needed: {a[:120]!r}"))],
    "follow-up-uk": [("Хто такий Андрій Маркін?", need("Маркін")), ("а його інстаграм?", need("@markin_andrey_kr"))],
    "follow-up-en": [("Who is Andrii Markin?", lang_is(is_english, "English")), ("what's his instagram?", lambda a: lang_is(is_english, "English")(a) or need("@markin_andrey_kr")(a))],
    "ambiguous-name": [("Який інстаграм у Олександра?", ambiguous_olexandr)],
    "out-of-kb-en": [("Do you have a swimming pool?", lambda a: lang_is(is_english, "English")(a) or english_refusal(a))],
    "no-invented-facts-pool": [("Чи є у вас басейн?", honest_refusal)],
    "no-invented-facts-bar": [("Скільки коштує протеїновий коктейль у барі залу?", honest_refusal)],
    "no-invented-facts-card": [("Чи можна оплатити абонемент карткою на рецепції, чи тільки готівкою?", honest_refusal)],
    "greeting-only": [("Привіт", short)],
    "greeting-plus-question": [("Привіт, скільки коштує безлімітний абонемент на місяць у Соцмісто?", lambda a: not_greeting_start(a) or need("1199")(a))],
    "uk-price": [("Скільки коштує безлімітний абонемент на місяць у 95 кварталі?", lambda a: lang_is(is_ukrainian, "Ukrainian", "1199")(a))],
}


def run_scenario(turns):
    history, errors = [], []
    for message, check in turns:
        answer = respond(message, history)["answer"]
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        error = check(answer) or next((f"hall name garbled: {m}" for m in MUTANTS if m.lower() in answer.lower()), None)
        if error:
            errors.append(f"turn {message!r}: {error}")
            break
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    names = [n for n in SCENARIOS if not args.only or n in args.only.split(",")]
    passed_all = 0
    for name in names:
        failures = []
        for _ in range(args.k):
            failures += run_scenario(SCENARIOS[name])
        ok = not failures
        passed_all += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name} (k={args.k})")
        for failure in failures[:2]:
            print(f"        {failure}")
    print(f"\nScenarios passing {args.k}/{args.k} runs: {passed_all}/{len(names)}")


if __name__ == "__main__":
    main()
