# Design Q&A

Answers to the questions this project is most likely to get pushed on.

## 1. Why Reciprocal Rank Fusion (RRF) instead of a weighted sum of scores?

Cosine distance (pgvector) and trigram similarity (`pg_trgm`) are different metrics on
incomparable scales: cosine similarity for related passages clusters tightly (often
0.75–0.95 in this embedding space), while trigram similarity is spiky — mostly near 0,
with sharp peaks on exact substring overlap. A weighted sum (`0.5*cosine + 0.5*trigram`)
requires normalizing both onto a shared scale, and that normalization is unstable per-query
with a small candidate pool — the min/max of 20 candidates shifts every query, so the same
raw score means something different from one question to the next.

RRF sidesteps this entirely: it only uses each retriever's **rank position**, never the raw
score. `score = sum(1 / (k + rank))` across retrievers. No calibration, no per-query
normalization, robust to wildly different score distributions. It's also the default hybrid
fusion strategy in Elasticsearch, Azure AI Search, and Weaviate — not a novel choice, a
standard one.

**The honest tradeoff, found by this project's own eval:** at `k=1` (top-1 only), hybrid
retrieval actually *loses* to vector-only on this corpus — 90% vs 100% recall (18/20 vs
20/20). Two questions about which trainer covers a topic ("who teaches kids' groups", "who
competes in IFBB bikini") had their correct vector-search top-1 result outranked after
fusion, because a *wrong* chunk scored a strong trigram top-1 (generic words like the club
name or "тренер" match broadly) and RRF's equal per-retriever weighting let that outvote a
right-but-not-emphasized vector match. Because RRF only sees rank, not confidence, a
retriever that's shakily right can be outvoted by one that's confidently wrong. At `k≥2`
both converge to 100% on this corpus — there's room for both signals to appear. This is
disclosed in the README rather than hidden; it's the actual limitation of rank-only fusion,
not a defect in the implementation.

Retuning `RRF_K` doesn't fix it either — swept {1, 5, 10, 30, 60, 100}, identical 18/20 at
every value. That's expected once you look at the actual mechanism: in both failing cases,
the correct doc is rank-1 in vector search but **absent entirely** from trigram's top-20
(never retrieved by that side at all), while the wrong doc appears in *both* retrievers'
candidate lists at moderate rank. Its two-retriever combined score
`1/(k+rank_v) + 1/(k+rank_t)` structurally exceeds the correct doc's single-retriever score
`1/(k+1)` for any positive `k`, because the extra term is always positive — so no value of
`k` changes which one wins. RRF here effectively rewards "found by both retrievers,
mediocre" over "found by exactly one retriever, top-ranked." That's not a bug to patch, it's
what independent-list rank voting does by construction.

**So why is hybrid retrieval still worth it, given it loses on this eval?** Because the 20
natural-language questions aren't the case hybrid retrieval is *for*. A second, deliberately
different eval set — 6 bare exact-match strings a user might paste verbatim (a phone number,
a price, an Instagram handle plus one word) — shows vector-only failing completely: **0/6**,
because a string like `"068 655 00 99"` or `"549 грн"` carries no semantic content for an
embedding model to latch onto. Hybrid scores **6/6** on the same set, because trigram search
matches the exact substring regardless of semantics. That's the concrete trade being made:
give up a little precision on natural-language top-1 ranking, in exchange for not silently
failing every time someone pastes a number or a handle instead of asking a question. Both
numbers are real and reproducible (`eval/run_eval.py`, `answerable` vs `exact_match` question
types) — this isn't a case of picking whichever eval flatters the design choice after the
fact; both were measured, and both are reported.

## 2. What does recall@k actually measure — and what does it not measure?

`recall@k`: for a question, is the chunk that contains the correct answer present *anywhere*
in the top-k retrieved results? It's a **retrieval-only** metric, computed before generation
runs at all.

What it does **not** tell you:
- **Whether the model used the chunk correctly.** The model can receive the right chunk in
  its top-5 and still answer wrong, ignore it, or misattribute it — this project's own eval
  found exactly that (see the IFBB question in `eval/qa_pairs.json`: the correct chunk was
  retrieved, but generation sometimes only reports on the other location's trainer with the
  same specialty). That's why generation-time **faithfulness** (does the answer contain the
  right facts) and **refusal rate** (does the model correctly say "I don't know" when the
  answer genuinely isn't in the KB) are measured separately in `eval/run_eval.py
  --with-generation`, decoupled from retrieval.
- **Position within the top-k.** Recall@5 treats rank 1 and rank 5 identically. For that,
  you'd want Mean Reciprocal Rank (MRR) or NDCG — not implemented here because with ~20
  total chunks the difference between "found at rank 1" and "found at rank 5" barely matters
  in practice; it would matter on a corpus large enough that rank within the top-k affects
  which chunks the LLM actually reads carefully.

## 3. Why `intfloat/multilingual-e5-small` specifically?

Three constraints, in order of how binding they were:
1. **The knowledge base is Ukrainian.** Most embedding models are English-centric and
   degrade sharply on Slavic morphology. The e5 multilingual family is trained across 100+
   languages with published multilingual retrieval benchmarks (MIRACL/MTEB) covering
   Ukrainian/Russian.
2. **Had to run locally, no per-request cost.** No embedding API dependency or per-query
   billing for a portfolio-scale project — e5-small is ~118M parameters, runs comfortably on
   CPU with no GPU.
3. **Size/latency vs. quality.** The "small" variant (384-dim) is fast enough for real-time
   ingestion and query without a GPU, at some retrieval-quality cost versus e5-base/large.
   For a corpus this size that cost isn't visible in the eval; it's a real tradeoff to
   revisit if the corpus grows into the thousands of chunks.

One easy-to-miss detail actually implemented: e5 models require literal `"query: "` /
`"passage: "` text prefixes for correct asymmetric retrieval (see `src/embeddings.py`) —
skipping this silently degrades results without erroring, which is exactly the kind of bug
that survives code review and only shows up as "the embeddings are a bit worse than
expected."

## 4. Why no reranker?

A cross-encoder reranker (e.g. bge-reranker, Cohere Rerank) re-scores the top-N candidates
by jointly encoding query+passage together — much more accurate than independent
embeddings, but it can't be precomputed or indexed; it runs at query time over every
candidate pair, so cost scales with candidate count.

Not used here, deliberately:
- **The candidate pool is ~20 chunks total.** A reranker's job is separating "good" from
  "great" inside a large, noisy candidate set. With a pool this small, RRF-fused top-k
  already **is** the relevant set almost every time — there's nothing left for a reranker to
  clean up.
- **Extra model, extra latency, extra cost** aren't justified until retrieval precision
  measurably suffers at the current scale. The actual trigger to add one is the corpus
  growing large enough that the top-20 candidates start containing genuinely irrelevant
  chunks the current fusion can't filter — not "best practice says to."
