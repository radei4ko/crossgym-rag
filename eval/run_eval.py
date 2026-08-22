import argparse
import json
from pathlib import Path

from src.generation import answer_question
from src.retrieval import hybrid_search

QA_PATH = Path(__file__).resolve().parent / "qa_pairs.json"


def load_qa_pairs() -> list[dict]:
    return json.loads(QA_PATH.read_text(encoding="utf-8"))


def run_retrieval_eval(match_count: int = 5) -> None:
    qa_pairs = load_qa_pairs()
    hits = 0
    for pair in qa_pairs:
        results = hybrid_search(pair["question"], match_count=match_count)
        sources = [doc["source"] for doc in results]
        hit = pair["expected_source"] in sources
        hits += hit
        status = "OK " if hit else "MISS"
        print(f"[{status}] {pair['question']} -> {sources}")

    recall = hits / len(qa_pairs)
    print(f"\nRetrieval recall@{match_count}: {hits}/{len(qa_pairs)} = {recall:.0%}")


def run_generation_eval(match_count: int = 5) -> None:
    qa_pairs = load_qa_pairs()
    hits = 0
    for pair in qa_pairs:
        documents = hybrid_search(pair["question"], match_count=match_count)
        result = answer_question(pair["question"], documents)
        answer = result["answer"]
        found_all = all(keyword.lower() in answer.lower() for keyword in pair["expected_keywords"])
        hits += found_all
        status = "OK " if found_all else "MISS"
        print(f"[{status}] {pair['question']}\n  answer: {answer}\n  expected: {pair['expected_keywords']}\n")

    faithfulness = hits / len(qa_pairs)
    print(f"Faithfulness (all expected keywords present): {hits}/{len(qa_pairs)} = {faithfulness:.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-generation", action="store_true", help="Also run full generation eval (calls OpenRouter)")
    args = parser.parse_args()

    run_retrieval_eval()
    if args.with_generation:
        print()
        run_generation_eval()
