import pytest

from src.lang import EN, RU, UK, answer_language_ok, detect, explicit_request, resolve_language


def convo(*messages):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": m} for i, m in enumerate(messages)]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("english?", EN),
        ("talk in english please", EN),
        ("Do you speak English?", EN),
        ("можеш по-російськи?", RU),
        ("ответь по-русски", RU),
        ("українською, будь ласка", UK),
        ("Who teaches English?", None),
        ("Хто викладає англійську?", None),
        ("Скільки коштує абонемент?", None),
    ],
)
def test_explicit_request(text, expected):
    assert explicit_request(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Хто з тренерів Соцмісто веде бокс?", UK),
        ("сколько стоит разовое занятие в Вечернем?", RU),
        ("Какой телефон у зала Восточный-2?", RU),
        ("did you have trainer who teach box?", EN),
        ("box and aerobics", EN),
        ("привіт", UK),
        ("привет", RU),
        ("бокс", None),
        ("VIP", None),
        ("ок", None),
    ],
)
def test_detect(text, expected):
    assert detect(text) == expected


def test_hall_name_spelling_does_not_flip_a_russian_question_to_ukrainian():
    # "Соцмісто" contains "і" but the sentence is Russian
    assert resolve_language("Сколько стоит абонемент в Соцмісто?") == RU


def test_the_reported_bug_english_survives_a_short_ambiguous_message():
    history = convo("привіт", "Привіт!", "english?", "Yes.", "did you have trainer who teach box?", "Yes, we do.")
    assert resolve_language("box and aerobics", history) == EN


def test_ambiguous_message_keeps_the_requested_language():
    assert resolve_language("бокс", convo("english?", "Sure.")) == EN


def test_confident_message_in_another_language_switches():
    assert resolve_language("а скільки коштує абонемент?", convo("english?", "Sure.")) == UK
    assert resolve_language("А адрес Соцмиста?", convo("english?", "Sure.")) == RU


def test_default_is_ukrainian():
    assert resolve_language("VIP") == UK


def test_old_wrapped_history_is_unwrapped():
    history = convo("Контекст:\n[1] ...\n\nПитання: english?", "Sure.")
    assert resolve_language("ok", history) == EN


def test_answer_language_check():
    assert answer_language_ok("Yes, we have boxing coaches at the Socmisto gym. You can contact them by phone.", EN)
    assert not answer_language_ok("Так, у залі Соцмісто є тренери з боксу. Можна записатися за телефоном клубу.", EN)
    assert answer_language_ok("Разовое занятие в зале Вечірній стоит 120 грн, а фитнес — 200 грн.", RU)
    assert not answer_language_ok("Разове тренування у залі Вечірній коштує 120 грн, а фітнес — 200 грн.", RU)
    assert not answer_language_ok("The single visit costs 120 UAH and the monthly pass is available for sale.", UK)
    assert answer_language_ok("Так", UK)  # too short to judge


def test_pure_language_request_vs_real_question():
    from src.lang import pure_language_request

    assert pure_language_request("english?") == EN
    assert pure_language_request("можеш по-російськи?") == RU
    assert pure_language_request("talk in english please") == EN
    assert pure_language_request("Who teaches English?") is None
    assert pure_language_request("можна англійською, скільки коштує абонемент?") is None
