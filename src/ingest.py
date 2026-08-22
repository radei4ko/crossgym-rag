from pathlib import Path

from src.chunking import chunk_text
from src.db import get_client
from src.embeddings import embed_passages

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"


def ingest() -> None:
    client = get_client()
    for path in sorted(KB_DIR.glob("*.md")):
        source = path.stem
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        embeddings = embed_passages(chunks)

        client.table("documents").delete().eq("source", source).execute()
        rows = [{"content": chunk, "source": source, "embedding": vector} for chunk, vector in zip(chunks, embeddings)]
        client.table("documents").insert(rows).execute()
        print(f"{source}: {len(rows)} chunks")


if __name__ == "__main__":
    ingest()
