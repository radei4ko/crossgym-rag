import argparse
import json
from pathlib import Path

from src.config import settings
from src.generation import answer_question
from src.retrieval import hybrid_search, vector_only_search

QA_PATH = Path(__file__).resolve().parent / "qa_pairs.json"

NEGATION_MARKERS = ["немає", "не маю", "не володію", "не знаю", "не маємо", "не згадується", "не вказано", "не вказан"]
INFO_STEM = "інформ"


def is_refusal(answer: str) -> bool:
    """A negation word plus the 'інформ-' stem, in either order — Ukrainian word order
    varies ("немає інформації" vs "інформації ... немає"), so a fixed phrase match
    produces false negatives on a genuinely correct refusal."""
    lowered = answer.lower()
    return any(marker in lowered for marker in NEGATION_MARKERS) and INFO_STEM in lowered


def _keyword_satisfied(keyword, answer_lower: str) -> bool:
    """A keyword entry can be a string (must appear) or a list of equivalent
    phrasings (any one satisfies it) — the model paraphrases correct answers."""
    if isinstance(keyword, list):
        return any(alt.lower() in answer_lower for alt in keyword)
    return keyword.lower() in answer_lower


def load_qa_pairs() -> list[dict]:
    pairs = json.loads(QA_PATH.read_text(encoding="utf-8"))
    answerable = [p for p in pairs if p["type"] == "answerable"]
    adversarial = [p for p in pairs if p["type"] == "adversarial"]
    return answerable, adversarial


def _recall_at_k(pairs: list[dict], search_fn, match_count: int, label: str) -> float:
    hits = 0
    for pair in pairs:
        results = search_fn(pair["question"], match_count=match_count)
        sources = [doc["source"] for doc in results]
        expected = pair["expected_source"]
        expected_sources = expected if isinstance(expected, list) else [expected]
        hit = any(source in sources for source in expected_sources)
        hits += hit
        status = "OK " if hit else "MISS"
        print(f"[{status}] {pair['question']} -> {sources}")
    recall = hits / len(pairs)
    print(f"\n{label} recall@{match_count}: {hits}/{len(pairs)} = {recall:.0%}")
    return recall


def run_retrieval_eval(match_count: int = settings.match_count) -> None:
    answerable, _ = load_qa_pairs()

    print("=== Hybrid (vector + trigram, RRF) ===")
    hybrid_recall = _recall_at_k(answerable, hybrid_search, match_count, "Hybrid")

    print("\n=== Vector-only baseline ===")
    vector_recall = _recall_at_k(answerable, vector_only_search, match_count, "Vector-only")

    print(f"\nHybrid vs vector-only: {hybrid_recall:.0%} vs {vector_recall:.0%}")


def run_generation_eval(match_count: int = settings.match_count) -> None:
    answerable, adversarial = load_qa_pairs()

    print("=== Faithfulness on answerable questions ===")
    hits = 0
    for pair in answerable:
        documents = hybrid_search(pair["question"], match_count=match_count)
        result = answer_question(pair["question"], documents)
        answer_lower = result["answer"].lower()
        found_all = all(_keyword_satisfied(keyword, answer_lower) for keyword in pair["expected_keywords"])
        hits += found_all
        status = "OK " if found_all else "MISS"
        print(f"[{status}] {pair['question']}\n  answer: {result['answer']}\n  expected: {pair['expected_keywords']}\n")
    faithfulness = hits / len(answerable)
    print(f"Faithfulness (all expected keywords present): {hits}/{len(answerable)} = {faithfulness:.0%}")

    print("\n=== Refusal on adversarial (out-of-KB) questions ===")
    refusals = 0
    for pair in adversarial:
        documents = hybrid_search(pair["question"], match_count=match_count)
        result = answer_question(pair["question"], documents)
        refused = is_refusal(result["answer"])
        refusals += refused
        status = "OK " if refused else "HALLUCINATED"
        print(f"[{status}] {pair['question']}\n  answer: {result['answer']}\n")
    refusal_rate = refusals / len(adversarial)
    print(f"Refusal rate on out-of-KB questions: {refusals}/{len(adversarial)} = {refusal_rate:.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-generation", action="store_true", help="Also run full generation eval (calls OpenRouter)")
    args = parser.parse_args()

    run_retrieval_eval()
    if args.with_generation:
        print()
        run_generation_eval()
