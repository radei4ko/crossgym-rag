from unittest.mock import patch

from src.pipeline import GREETING, NO_INFO, UNAVAILABLE, _sanitise_history, is_smalltalk, respond


def test_greeting_is_answered_without_calling_the_model():
    with patch("src.pipeline.answer_question") as model:
        result = respond("Привіт")
    model.assert_not_called()
    assert result["answer"] == GREETING["uk"]


def test_greeting_language_follows_the_greeting():
    assert respond("Здравствуйте")["answer"] == GREETING["ru"]
    assert respond("hello")["answer"] == GREETING["en"]


def test_greeting_plus_question_is_not_smalltalk():
    assert not is_smalltalk("Привіт, скільки коштує абонемент?")
    assert is_smalltalk("Привіт, хто ти?")


def test_routed_question_passes_language_and_documents_to_the_model():
    docs = [{"source": "trainers_socmisto", "content": "[Зал: Соцмісто | Тренери]\n## Миколай"}]
    with patch("src.pipeline.fetch_sources", return_value=docs), patch(
        "src.pipeline.answer_question", return_value={"answer": "ok", "sources": []}
    ) as model:
        respond("Who coaches boxing at Sotsmisto?")
    args, kwargs = model.call_args
    assert args[1] == docs and kwargs["lang"] == "en"


def test_unrecognised_question_falls_back_to_hybrid_search():
    docs = [{"source": "trainers_95kvartal", "content": "x", "id": 1}]
    with patch("src.pipeline.hybrid_search", return_value=docs) as search, patch(
        "src.pipeline.answer_question", return_value={"answer": "ok", "sources": []}
    ):
        respond("Розкажи про Катерину Резвін")
    search.assert_called_once()


def test_empty_retrieval_gives_a_no_info_answer_in_the_users_language():
    with patch("src.pipeline.hybrid_search", return_value=[]):
        assert respond("Do you have a swimming pool?")["answer"] == NO_INFO["en"]


def test_model_failure_returns_a_polite_message_not_an_exception():
    docs = [{"source": "policy", "content": "x"}]
    with patch("src.pipeline.fetch_sources", return_value=docs), patch(
        "src.pipeline.answer_question", side_effect=RuntimeError("429")
    ):
        assert respond("Чи можна скасувати запис?")["answer"] == UNAVAILABLE["uk"]


def test_history_is_trimmed_to_the_last_two_exchanges_and_unwrapped():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    history[8]["content"] = "Контекст:\n...\n\nПитання: english?"
    clean = _sanitise_history(history)
    assert len(clean) == 4 and clean[-2]["content"] == "english?"


def test_empty_model_answer_becomes_a_polite_no_info_message():
    docs = [{"source": "policy", "content": "x"}]
    with patch("src.pipeline.fetch_sources", return_value=docs), patch(
        "src.pipeline.answer_question", return_value={"answer": " ", "sources": []}
    ):
        assert respond("Чи можна скасувати запис?")["answer"] == NO_INFO["uk"]


def test_pure_language_request_is_acknowledged_without_search_or_model():
    from src.pipeline import LANGUAGE_ACK

    with patch("src.pipeline.hybrid_search") as search, patch("src.pipeline.answer_question") as model:
        assert respond("english?")["answer"] == LANGUAGE_ACK["en"]
        assert respond("можеш по-російськи?")["answer"] == LANGUAGE_ACK["ru"]
    search.assert_not_called()
    model.assert_not_called()
