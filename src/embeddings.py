from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str], prefix: str) -> list[list[float]]:
    """Embed texts with the multilingual-e5 required prefix ('query: ' or 'passage: ')."""
    prefixed = [f"{prefix} {t}" for t in texts]
    vectors = _model().encode(prefixed, normalize_embeddings=True)
    return vectors.tolist()


def embed_passages(texts: list[str]) -> list[list[float]]:
    return embed(texts, "passage:")


def embed_query(text: str) -> list[float]:
    return embed([text], "query:")[0]
