from collections import defaultdict

from src.db import get_client
from src.embeddings import embed_passages
from src.kb import build_chunks


def ingest() -> None:
    """Re-index the knowledge base without ever leaving the live bot with an empty table:
    new rows are inserted first, and only then are the previous rows of the same source
    deleted (for a few seconds both generations exist, which is harmless)."""
    client = get_client()
    by_source: dict[str, list[str]] = defaultdict(list)
    for row in build_chunks():
        by_source[row["source"]].append(row["content"])

    for source, contents in by_source.items():
        old_ids = [r["id"] for r in client.table("documents").select("id").eq("source", source).execute().data]
        embeddings = embed_passages(contents)
        rows = [{"content": c, "source": source, "embedding": v} for c, v in zip(contents, embeddings)]
        client.table("documents").insert(rows).execute()
        if old_ids:
            client.table("documents").delete().in_("id", old_ids).execute()
        print(f"{source}: {len(rows)} chunks (replaced {len(old_ids)})")


if __name__ == "__main__":
    ingest()
