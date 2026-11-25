import re
from collections import Counter

from src.kb import KB_DIR, PRICE_GROUPS, build_chunks


def _prices(stem):
    return sorted(re.findall(r"\d[\d ]*\s*грн", (KB_DIR / f"{stem}.md").read_text(encoding="utf-8")))


def test_halls_in_a_price_group_really_have_identical_prices():
    # pricing_overview merges these halls into one block; this fails if their files diverge
    for first, second in PRICE_GROUPS:
        assert _prices(f"pricing_{first}") == _prices(f"pricing_{second}"), (first, second)


def test_every_chunk_names_its_hall_or_topic():
    for row in build_chunks():
        assert row["content"].startswith("["), row["content"][:60]
        if row["source"].split("_")[0] in ("locations", "pricing", "trainers") and row["source"] != "pricing_overview":
            assert "Зал:" in row["content"].splitlines()[0]


def test_no_two_chunks_are_identical():
    counts = Counter(row["content"] for row in build_chunks())
    assert [c for c, n in counts.items() if n > 1] == []


def test_a_trainer_is_never_split_across_chunks():
    for row in build_chunks():
        if row["source"].startswith("trainers_") and "## " in row["content"]:
            assert row["content"].count("## ") == 1
            assert "Instagram:" in row["content"]


def test_trainers_who_work_in_two_halls_appear_in_both_files():
    rows = build_chunks()
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row["content"])
    hall_95 = "\n".join(by_source["trainers_95kvartal"])
    for handle in ("@dmitriy_pt.ua", "@vladyslav_cross", "@pro_sport_kr", "@alinostra_0504"):
        assert handle in hall_95, handle
    assert "@slavatraining_" in "\n".join(by_source["trainers_socmisto"])


def test_all_boxing_coaches_of_socmisto_carry_the_boxing_tag():
    boxing = [r["content"] for r in build_chunks() if r["source"] == "trainers_socmisto" and "boxing" in r["content"]]
    assert len(boxing) == 3


def test_price_overview_contains_every_price_number_of_the_halls_it_merges():
    overview = next(r["content"] for r in build_chunks() if r["source"] == "pricing_overview")
    for first, _second in PRICE_GROUPS:
        for price in _prices(f"pricing_{first}"):
            assert price in overview, price
