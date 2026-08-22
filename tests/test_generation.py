from unittest.mock import patch

from src.generation import answer_question, build_prompt


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
