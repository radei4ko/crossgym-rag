from src.db import get_client
from src.embeddings import embed_query

CANDIDATE_COUNT = 20
RRF_K = 60


def reciprocal_rank_fusion(ranked_id_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Fuse several ranked id lists into one ranking via Reciprocal Rank Fusion.

    Each list is ordered best-first. A doc absent from a list contributes nothing
    from that list, so it's not penalized beyond simply missing that signal.
    """
    scores: dict[int, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def hybrid_search(query_text: str, match_count: int = 5) -> list[dict]:
    """Vector similarity + trigram keyword search, fused with RRF."""
    client = get_client()
    query_vector = embed_query(query_text)

    vector_rows = (
        client.rpc(
            "match_documents_vector",
            {"query_embedding": query_vector, "match_count": CANDIDATE_COUNT},
        )
        .execute()
        .data
    )
    trgm_rows = (
        client.rpc(
            "match_documents_trgm",
            {"query_text": query_text, "match_count": CANDIDATE_COUNT},
        )
        .execute()
        .data
    )

    docs_by_id = {row["id"]: row for row in vector_rows + trgm_rows}
    vector_ids = [row["id"] for row in vector_rows]
    trgm_ids = [row["id"] for row in trgm_rows]

    fused = reciprocal_rank_fusion([vector_ids, trgm_ids])
    top_ids = [doc_id for doc_id, _score in fused[:match_count]]
    return [docs_by_id[doc_id] for doc_id in top_ids]
