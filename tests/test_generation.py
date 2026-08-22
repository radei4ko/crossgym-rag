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
