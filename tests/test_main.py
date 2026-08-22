from src.main import _needs_coreference_resolution


def test_self_contained_question_does_not_need_history():
    assert _needs_coreference_resolution("Дай інстаграм залу") is False


def test_pronoun_follow_up_needs_history():
    assert _needs_coreference_resolution("У тебе є її інстаграм?") is True


def test_pronoun_with_punctuation_is_still_detected():
    assert _needs_coreference_resolution("А там скільки коштує?") is True


def test_named_entity_question_does_not_need_history():
    assert _needs_coreference_resolution("Скільки коштує абонемент у Соцмісто?") is False
