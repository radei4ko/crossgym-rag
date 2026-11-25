from unittest.mock import patch

from src.generation import (
    _remove_ungrounded_phones,
    _restore_mangled_identifiers,
    _strip_language_preamble,
    _strip_bold_markers,
    _strip_citation_markers,
    answer_question,
    build_prompt,
)


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


def test_strip_citation_markers_removes_bracketed_source_index():
    assert _strip_citation_markers("Тренування коштує 200 грн [2].") == "Тренування коштує 200 грн."


def test_strip_citation_markers_removes_multiple_markers():
    text = "Зал А [1] коштує 200 грн, зал Б [12] коштує 120 грн."
    assert _strip_citation_markers(text) == "Зал А коштує 200 грн, зал Б коштує 120 грн."


def test_strip_citation_markers_removes_comma_separated_group():
    text = "Разове тренування коштує 200 грн [1, 4]."
    assert _strip_citation_markers(text) == "Разове тренування коштує 200 грн."


def test_strip_citation_markers_leaves_plain_text_untouched():
    assert _strip_citation_markers("Немає позначок тут.") == "Немає позначок тут."


def test_answer_question_strips_citation_markers():
    documents = [{"source": "pricing_socmisto", "content": "Разове тренування: 200 грн"}]

    with patch("src.generation.call_openrouter", return_value="Разове тренування коштує 200 грн [2]."):
        result = answer_question("Скільки коштує тренування?", documents)

    assert result["answer"] == "Разове тренування коштує 200 грн."


def test_language_directive_is_the_last_thing_in_the_user_message():
    docs = [{"source": "policy", "content": "Правила"}]
    messages = build_prompt("box and aerobics", docs, lang="en")
    last_paragraph = messages[-1]["content"].split("\n\n")[-1]
    assert last_paragraph.startswith("Reply language: English")
    assert "Sotsmisto" in last_paragraph  # fixed English hall names: no "Sozhmisto"


def test_strict_directive_flags_the_previous_wrong_language():
    docs = [{"source": "policy", "content": "Правила"}]
    messages = build_prompt("q", docs, lang="ru", strict=True)
    assert "wrong language" in messages[-1]["content"] and "Russian" in messages[-1]["content"]


def test_indexed_chunk_keeps_its_own_header_and_unindexed_chunk_gets_a_label():
    docs = [
        {"source": "trainers_socmisto", "content": "[Зал: Соцмісто | Тренери]\n## Миколай"},
        {"source": "policy", "content": "Правила"},
    ]
    content = build_prompt("q", docs)[-1]["content"]
    assert "[Зал: Соцмісто | Тренери]" in content and "[Правила запису та скасування]" in content
    assert "trainers_socmisto" not in content  # the slug must not reach the model


def test_wrong_language_answer_is_retried_once():
    docs = [{"source": "locations_socmisto", "content": "Адреса: вул. Степана Тільги, 16."}]
    wrong = "Зал Соцмісто знаходиться на вулиці Степана Тільги, шістнадцять, у місті Кривий Ріг."
    right = "The Socmisto gym is located at Stepana Tilhy street 16 in Kryvyi Rih, you can visit us."
    with patch("src.generation.call_openrouter", side_effect=[wrong, right]) as call:
        result = answer_question("where is socmisto?", docs, lang="en")
    assert call.call_count == 2 and result["answer"] == right


def test_language_preamble_is_stripped_but_real_answers_are_not():
    assert _strip_language_preamble("Питання російською, тож відповідаю російською.\n\nРазовое — 120 грн.").strip() == "Разовое — 120 грн."
    assert _strip_language_preamble("Okay, I will continue in English.\nHello!").strip() == "Hello!"
    assert _strip_language_preamble("Разове тренування коштує 120 грн.") == "Разове тренування коштує 120 грн."


def test_payload_is_deterministic_and_capped():
    import src.generation as g

    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"model": "m", "choices": [{"message": {"content": "ok"}}]}

    with patch("src.generation.httpx.post", return_value=R()) as post:
        g.call_openrouter([{"role": "user", "content": "x"}])
    body = post.call_args.kwargs["json"]
    assert body["temperature"] == 0 and body["max_tokens"] > 0


def test_invented_phone_number_is_removed_but_real_one_kept():
    docs = [{"source": "locations_socmisto", "content": "Телефон: 068 655 00 99"}]
    answer = "Інформації про паркування немає. Можете уточнити за телефоном: 097 123 45 67."
    assert _remove_ungrounded_phones(answer, docs) == "Інформації про паркування немає."
    real = "Телефон залу: 068 655 00 99."
    assert _remove_ungrounded_phones(real, docs) == real


def test_phone_written_with_country_code_matches_the_context_number():
    docs = [{"source": "locations_socmisto", "content": "Телефон: 068 655 00 99"}]
    assert "+38 068 655 00 99" in _remove_ungrounded_phones("Дзвоніть: +38 068 655 00 99", docs)


def test_answer_that_is_only_an_invented_phone_is_not_left_empty():
    docs = [{"source": "policy", "content": "Правила"}]
    assert _remove_ungrounded_phones("097 123 45 67", docs) == ""  # caller falls back to polite text


def test_prices_are_not_mistaken_for_phones():
    docs = [{"source": "pricing", "content": "Абонемент 1199 грн"}]
    assert _remove_ungrounded_phones("Абонемент коштує 1199 грн, VIP 1800 грн.", docs) == "Абонемент коштує 1199 грн, VIP 1800 грн."
