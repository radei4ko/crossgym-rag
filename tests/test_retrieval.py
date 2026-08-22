from src.retrieval import reciprocal_rank_fusion


def test_agreement_boosts_rank():
    vector_ranked = [1, 2, 3]
    trgm_ranked = [2, 1, 3]
    fused = reciprocal_rank_fusion([vector_ranked, trgm_ranked])
    fused_ids = [doc_id for doc_id, _score in fused]
    assert fused_ids[0] in (1, 2)
    assert set(fused_ids) == {1, 2, 3}


def test_doc_missing_from_one_list_still_included():
    vector_ranked = [1, 2]
    trgm_ranked = [3]
    fused = reciprocal_rank_fusion([vector_ranked, trgm_ranked])
    fused_ids = {doc_id for doc_id, _score in fused}
    assert fused_ids == {1, 2, 3}


def test_top_rank_scores_higher_than_bottom_rank():
    fused = reciprocal_rank_fusion([[1, 2, 3]])
    scores = dict(fused)
    assert scores[1] > scores[2] > scores[3]


def test_empty_lists_return_empty_result():
    assert reciprocal_rank_fusion([[], []]) == []
