import pytest

from src.routing import detect_clubs, detect_topics, select_sources


def convo(*messages):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": m} for i, m in enumerate(messages)]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Хто з тренерів Соцмісто веде бокс?", ["socmisto"]),
        ("Who coaches boxing at Sotsmisto?", ["socmisto"]),
        ("Какой телефон у зала Восточный-2?", ["skhidnyi"]),
        ("ціни у Східному-2", ["skhidnyi"]),
        ("сколько стоит в Вечернем", ["vechirniy"]),
        ("Яка адреса залу 95 квартал?", ["95kvartal"]),
        ("price at Plaza", ["95kvartal"]),
        ("Соцмісто і Вечірній", ["socmisto", "vechirniy"]),
        ("Скільки коштує абонемент?", []),
    ],
)
def test_detect_clubs(text, expected):
    assert detect_clubs(text) == expected


def test_detect_topics():
    assert detect_topics("Хто веде бокс?") == {"trainers"}
    assert detect_topics("Який телефон залу?") == {"locations"}
    assert "pricing" in detect_topics("сколько стоит разовое")
    assert "policy" in detect_topics("Чи можна скасувати запис?")


def test_hall_and_topic_load_exactly_that_file():
    assert select_sources("Хто з тренерів Соцмісто веде бокс?") == ["trainers_socmisto"]
    assert select_sources("Який телефон залу Східний-2?") == ["locations_skhidnyi"]


def test_price_without_hall_uses_the_overview_not_one_arbitrary_hall():
    assert select_sources("Скільки коштує разове тренування?") == ["pricing_overview"]


def test_trainer_question_without_hall_loads_every_trainers_file():
    sources = select_sources("Який інстаграм у Олександра?")
    assert sources and all(s.startswith("trainers_") for s in sources) and len(sources) == 4


def test_hall_without_topic_loads_everything_about_that_hall():
    sources = select_sources("а в Східному-2?")
    assert {"locations_skhidnyi", "pricing_skhidnyi", "trainers_skhidnyi"} <= set(sources)


def test_follow_up_inherits_the_hall_from_history():
    history = convo("Хто такий Андрій Маркін?", "Тренер з боксу у залі Соцмісто.")
    assert select_sources("а його інстаграм?", history) == ["trainers_socmisto"]


def test_non_follow_up_does_not_inherit_a_hall():
    history = convo("Скільки коштує разове в Соцмісто?", "200 грн")
    assert select_sources("Який телефон залу, де знаходиться ваш зал взагалі?", history) != ["locations_socmisto"]


def test_nothing_recognised_falls_back_to_search():
    assert select_sources("Розкажи про Катерину Резвін") is None
    assert select_sources("Чи є у вас басейн?") is None


def test_skill_word_alone_makes_it_a_trainers_question():
    assert select_sources("box and aerobics") == [f"trainers_{c}" for c in ("socmisto", "skhidnyi", "95kvartal", "vechirniy")]


def test_narrow_trainers_keeps_only_the_asked_skill_and_lists_each_person_once():
    from src.routing import narrow_trainers

    def chunk(source, handle, tags):
        return {"source": source, "content": f"[Зал | Тренери]\n## X\nInstagram: {handle}\nТеги: {tags}"}

    docs = [
        chunk("trainers_95kvartal", "@slavatraining_", "бокс / boxing"),
        chunk("trainers_socmisto", "@slavatraining_", "бокс / boxing"),
        chunk("trainers_socmisto", "@nik_pro__", "бокс / boxing"),
        chunk("trainers_socmisto", "@romansirmanov", "пауерліфтинг / powerlifting"),
        {"source": "policy", "content": "Правила"},
    ]
    kept = narrow_trainers("Who coaches boxing?", docs)
    handles = [d["content"].split("Instagram: ")[1].split("\n")[0] for d in kept if "Instagram" in d["content"]]
    assert handles == ["@slavatraining_", "@nik_pro__"] and kept[-1]["source"] == "policy"


def test_narrow_trainers_without_a_skill_only_removes_duplicates():
    from src.routing import narrow_trainers

    doc = {"source": "trainers_socmisto", "content": "Instagram: @a"}
    assert narrow_trainers("Який інстаграм у Олександра?", [doc, dict(doc, source="trainers_95kvartal")]) == [doc]


def test_unmatched_skill_does_not_empty_the_context():
    from src.routing import narrow_trainers

    docs = [{"source": "trainers_socmisto", "content": "Instagram: @a\nТеги: стретчинг / stretching"}]
    assert narrow_trainers("who teaches boxing", docs) == docs
