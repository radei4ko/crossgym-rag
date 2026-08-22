from src.chunking import chunk_text


def test_short_text_returns_single_chunk():
    text = "Одне коротке речення."
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    assert chunks == [text]


def test_splits_on_paragraph_boundary():
    text = "Перший абзац " + "а" * 300 + ".\n\nДругий абзац " + "б" * 300 + "."
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    assert len(chunks) >= 2
    assert all(len(c) <= 400 + 80 for c in chunks)


def test_overlap_preserves_tail_context():
    text = "А" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    for previous, following in zip(chunks, chunks[1:]):
        assert following.startswith(previous[-50:])


def test_no_empty_chunks():
    text = "Абзац один.\n\n\n\nАбзац два."
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    assert all(chunk.strip() for chunk in chunks)
