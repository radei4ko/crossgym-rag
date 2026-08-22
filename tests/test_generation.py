from unittest.mock import patch

from src.generation import _restore_mangled_identifiers, _strip_bold_markers, answer_question, build_prompt


def test_build_prompt_includes_context_and_question():
    documents = [{"source": "locations", "content": "Адреса: вул. Сахарова, 17а."}]
    messages = build_prompt("Де знаходиться Східний-2?", documents)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Сахарова" in messages[1]["content"]
    assert "Де знаходиться Східний-2?" in messages[1]["content"]


def test_answer_question_returns_answer_and_sources():
    documents = [{"source": "locations", "content": "Адреса: вул. Сахарова, 17а."}]

    with patch("src.generation.call_openrouter", return_value="Зал Східний-2 на вул. Сахарова, 17а.") as mock_call:
        result = answer_question("Де Східний-2?", documents)

    mock_call.assert_called_once()
    assert result["answer"] == "Зал Східний-2 на вул. Сахарова, 17а."
    assert result["sources"] == [{"source": "locations", "content": "Адреса: вул. Сахарова, 17а."}]


def test_build_prompt_inserts_history_between_system_and_question():
    documents = [{"source": "trainers_skhidnyi", "content": "Катерина Кабанець, Instagram: @katerynkafit"}]
    history = [
        {"role": "user", "content": "Дай контакти тренера катерини"},
        {"role": "assistant", "content": "Катерина Кабанець працює в Східному-2."},
    ]
    messages = build_prompt("У тебе є її інстаграм?", documents, history)

    assert messages[0]["role"] == "system"
    assert messages[1] == history[0]
    assert messages[2] == history[1]
    assert messages[3]["role"] == "user"
    assert "У тебе є її інстаграм?" in messages[3]["content"]


def test_answer_question_passes_history_through():
    documents = [{"source": "trainers_skhidnyi", "content": "Instagram: @katerynkafit"}]
    history = [{"role": "user", "content": "Дай контакти Катерини"}]

    with patch("src.generation.call_openrouter", return_value="@katerynkafit") as mock_call:
        answer_question("А інстаграм?", documents, history=history)

    sent_messages = mock_call.call_args[0][0]
    assert history[0] in sent_messages


def test_answer_question_restores_stripped_instagram_underscores():
    documents = [
        {"source": "locations_socmisto", "content": "Instagram мережі CrossGYM: crossgym_baza_team"}
    ]

    with patch("src.generation.call_openrouter", return_value="Ось інстаграм залу: crossgymbazateam"):
        result = answer_question("Дай інстаграм залу", documents)

    assert "crossgym_baza_team" in result["answer"]
    assert "crossgymbazateam" not in result["answer"]


def test_answer_question_leaves_correct_identifier_untouched():
    documents = [
        {"source": "locations_socmisto", "content": "Instagram мережі CrossGYM: crossgym_baza_team"}
    ]

    with patch("src.generation.call_openrouter", return_value="Ось інстаграм залу: crossgym_baza_team"):
        result = answer_question("Дай інстаграм залу", documents)

    assert result["answer"] == "Ось інстаграм залу: crossgym_baza_team"


def test_restore_mangled_identifiers_handles_at_prefixed_handle():
    documents = [{"source": "trainers_socmisto", "content": "Instagram: @dmitriy_pt.ua"}]

    answer = _restore_mangled_identifiers("Ось: dmitriypt.ua", documents)

    assert "@dmitriy_pt.ua" in answer


def test_restore_mangled_identifiers_handles_double_underscore_handle():
    documents = [{"source": "trainers_skhidnyi", "content": "Instagram: @d__sanya__72"}]

    answer = _restore_mangled_identifiers("Тренер має інстаграм @dsanya72", documents)

    assert "@d__sanya__72" in answer


def test_restore_mangled_identifiers_ignores_unrelated_text():
    documents = [{"source": "locations_socmisto", "content": "Instagram мережі CrossGYM: crossgym_baza_team"}]

    answer = _restore_mangled_identifiers("Зал працює з 07:00 до 21:00.", documents)

    assert answer == "Зал працює з 07:00 до 21:00."


def test_strip_bold_markers_removes_double_asterisks():
    assert _strip_bold_markers("Instagram: **crossgym_baza_team**") == "Instagram: crossgym_baza_team"


def test_strip_bold_markers_leaves_plain_text_untouched():
    assert _strip_bold_markers("Немає форматування тут.") == "Немає форматування тут."


def test_answer_question_strips_bold_around_identifier():
    documents = [{"source": "locations_socmisto", "content": "Instagram мережі CrossGYM: crossgym_baza_team"}]

    with patch("src.generation.call_openrouter", return_value="Instagram: **crossgym_baza_team**"):
        result = answer_question("Дай інстаграм залу", documents)

    assert result["answer"] == "Instagram: crossgym_baza_team"
