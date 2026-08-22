# CrossGYM RAG Assistant

A retrieval-augmented generation (RAG) system that answers questions about a gym chain
(locations, pricing, trainers, booking policy) from its real knowledge base — exposed as a
REST API and, live, as a Telegram bot. Built as a portfolio project to demonstrate
production-style RAG engineering: hybrid retrieval, citation-backed answers, and an
evaluation harness that measures retrieval and generation separately — not just a prompt
wrapped in an API.

The knowledge base is real: two CrossGYM locations (Sotsmisto, Skhidnyi-2) in Kryvyi Rih,
Ukraine, with their actual pricing and trainer rosters. Content is in Ukrainian; this README
and the code are in English.

## Live demo

Talk to it on Telegram: **[@crossgym_chatbot](https://t.me/crossgym_chatbot)** — ask it about pricing,
trainers, locations, or booking policy in Ukrainian. Ask it something outside its knowledge
base and it will tell you it doesn't know, rather than guess.

<!-- 2-3 real conversation screenshots go here -->

## Results (measured, not asserted)

Run yourself: `python3 -m eval.run_eval --with-generation` (needs `OPENROUTER_API_KEY`;
retrieval-only numbers need no LLM calls).

| Metric | Score | What it means |
|---|---|---|
| Hybrid retrieval recall@5 | **20/20 (100%)** | correct chunk is in the top 5 for every answerable eval question |
| Vector-only recall@5 (baseline) | **20/20 (100%)** | ties hybrid at k=5 — see caveat below |
| Hybrid retrieval recall@1 | **18/20 (90%)** | hybrid *loses* to vector-only (100%) at top-1 — see below |
| Faithfulness (generated answer contains the right facts) | **18/20 (90%)** | measured on the LLM's output, not on retrieval |
| Refusal rate on out-of-KB questions | **5/5 (100%)** | model never hallucinated an answer to a question genuinely absent from the KB |

**Honest caveat on the headline recall@5 numbers:** with a corpus this small (~20 chunks),
both hybrid and vector-only retrieval saturate to 100% at k=5 — there just isn't enough
noise in the candidate pool for k=5 to matter yet. The more interesting number is
**recall@1**, where hybrid actually *regresses* against vector-only: RRF's rank-only fusion
let a confidently-wrong trigram match outvote a correctly-ranked vector match on 2 of 20
questions. That's a real limitation of Reciprocal Rank Fusion, not a bug — full breakdown in
[`docs/DESIGN_QA.md`](docs/DESIGN_QA.md#1-why-reciprocal-rank-fusion-rrf-instead-of-a-weighted-sum-of-scores).

**The two faithfulness misses are also worth naming, not hiding:** one is an eval-harness
artifact (the model gave a correct paraphrase that a literal keyword match didn't catch,
since fixed by accepting equivalent phrasings — see `eval/qa_pairs.json`); the other is a
genuine generation-time attribution slip on a question where two different locations' KB
entries share a keyword ("IFBB fitness-bikini") — the retrieved chunk was there, but the
model sometimes reports on only one location. Root-caused in
[`docs/DESIGN_QA.md`](docs/DESIGN_QA.md#2-what-does-recallk-actually-measure--and-what-does-it-not-measure).

Design rationale for hybrid retrieval, RRF vs. weighted fusion, the embedding model choice,
and why there's no reranker: [`docs/DESIGN_QA.md`](docs/DESIGN_QA.md).

## Why these design choices

**Hybrid retrieval (vector + keyword), not vector-only.** Pure embedding search misses exact
matches on names, prices, and Instagram handles — short, low-context strings that don't embed
well semantically. Postgres trigram search (`pg_trgm`) catches those; vector search catches
paraphrased questions. Both run as independent ranked lists and get fused with
**Reciprocal Rank Fusion (RRF)**, a rank-based combination that doesn't require normalizing
incomparable similarity scores (cosine distance vs. trigram similarity) onto the same scale —
a common source of bugs when combining heterogeneous retrievers naively. (See the measured
recall@1 regression above for the honest tradeoff this makes.)

**RRF implemented in Python, not SQL.** The two Postgres RPC functions
(`match_documents_vector`, `match_documents_trgm`) each do one simple, well-indexed ranked
query. Fusion logic lives in `src/retrieval.py` as a pure function
(`reciprocal_rank_fusion`) with no I/O — easy to unit test and easy to read, instead of being
buried in a PL/pgSQL function.

**Local multilingual embeddings, not an embedding API.** The knowledge base is Ukrainian.
`intfloat/multilingual-e5-small` runs locally via `sentence-transformers`, avoiding a paid
API dependency and per-request latency/cost for a portfolio-scale project. Note the e5 model
family requires `"query: "` / `"passage: "` prefixes on input text — a real, easy-to-miss
detail (see `src/embeddings.py`).

**Generation is provider-agnostic via OpenRouter.** The model is a config value
(`OPENROUTER_MODEL`), not hardcoded, so swapping models doesn't touch code.

**Citations by construction.** Every answer is generated strictly from the retrieved chunks,
and the API returns those chunks (`sources`) alongside the answer, so a caller can verify
what the model actually saw.

**Short-term conversation memory for the Telegram layer.** A follow-up like "what's her
Instagram?" has no entity name in it — retrieval alone can't resolve it. The last few turns
of the conversation are kept per chat and sent alongside the question, and the previous
assistant reply is folded into the retrieval query itself so the right chunk gets fetched
even when the current question doesn't name the entity.

## Architecture

```
 Telegram user
      │
      ▼
 Telegram Bot API
      │  webhook
      ▼
 n8n workflow ──────────────────────────────────────────────┐
 ┌─────────────────┐   ┌────────────────┐   ┌─────────────┐ │
 │ Telegram Trigger │──►│  Load History   │──►│  HTTP POST   │ │
 │                  │   │ (per-chat, n8n  │   │  /ask        │ │
 │                  │   │  static data)   │   │              │ │
 └─────────────────┘   └────────────────┘   └──────┬───────┘ │
                                                     │         │
                        ┌────────────────┐          │         │
                        │  Save History   │◄─────────┘         │
                        │ (append Q + A)  │                    │
                        └────────┬────────┘                    │
                                 ▼                              │
                        ┌────────────────┐                     │
                        │  Send Answer    │─────────────────────┘
                        │ (+ quick-reply  │
                        │  keyboard)      │
                        └────────────────┘

                                 ▲
                                 │ same request also reachable directly
                                 │
                     ┌─────────────────────┐
   POST /ask ──────► │   FastAPI (main.py)  │
   (question,        └──────────┬──────────┘
    history)                    │
                                 ▼
                    ┌────────────────────────┐
                    │  hybrid_search(query)   │   src/retrieval.py
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
     match_documents_vector           match_documents_trgm
     (pgvector cosine, RPC)           (pg_trgm similarity, RPC)
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                  reciprocal_rank_fusion(...)      (pure fn, unit-tested)
                                 │
                                 ▼
                    top-k documents (with sources)
                                 │
                                 ▼
                  build_prompt + call_openrouter    src/generation.py
                                 │
                                 ▼
                  { answer, sources }  ◄──────────  returned to caller


   Offline (once per KB change):
   data/kb/*.md ──► chunk_text() ──► embed_passages() ──► upsert into `documents`
                  src/chunking.py    src/embeddings.py         src/ingest.py
```

## Project layout

```
crossgym-rag/
├── data/kb/*.md      # knowledge base (Ukrainian): locations, pricing, trainers, policy
├── docs/
│   └── DESIGN_QA.md    # design rationale: RRF vs weighted fusion, recall@k, embeddings, reranker
├── src/
│   ├── config.py     # env-based settings
│   ├── chunking.py    # pure text chunking, paragraph-aware
│   ├── embeddings.py  # sentence-transformers wrapper (multilingual e5)
│   ├── db.py           # Supabase client factory
│   ├── ingest.py       # loads data/kb, chunks, embeds, upserts to Supabase
│   ├── retrieval.py    # hybrid search + RRF fusion + vector-only baseline
│   ├── generation.py   # prompt building (+ conversation history) + OpenRouter call
│   └── main.py          # FastAPI app: POST /ask, GET /health
├── eval/
│   ├── qa_pairs.json    # 20 answerable + 5 adversarial (out-of-KB) Q&A pairs
│   └── run_eval.py       # retrieval recall@k (hybrid vs vector-only), faithfulness, refusal rate
├── tests/                 # pytest: chunking, RRF fusion, generation (mocked)
└── Dockerfile
```

## Setup (local)

1. Create a Python 3.12 virtual environment and install dependencies:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in the blanks:
   - `SUPABASE_SERVICE_ROLE_KEY` — from your Supabase project's dashboard (Settings → API).
     Never commit this; it's a secret, server-side-only key.
   - `OPENROUTER_API_KEY` — from https://openrouter.ai
   - `API_KEY` — any string; it's the bearer key for this API's own `/ask` endpoint.
3. Ingest the knowledge base (embeds and uploads `data/kb/*.md` to Supabase):
   ```
   python3 -m src.ingest
   ```
4. Run the API:
   ```
   uvicorn src.main:app --reload
   ```
5. Ask a question:
   ```
   curl -X POST localhost:8000/ask \
     -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
     -d '{"question": "Скільки коштує разове тренування в Соцмісто?"}'
   ```

## Testing and evaluation

```
python3 -m pytest                            # unit tests: chunking, RRF fusion, generation (mocked)
python3 -m eval.run_eval                       # retrieval recall@k: hybrid vs vector-only baseline
python3 -m eval.run_eval --with-generation     # + faithfulness on answerable Qs, refusal rate on adversarial Qs
```

## Deployment (how this is actually running)

The API runs as one more container in the same Docker Compose stack as the Telegram
automation (n8n) and a Caddy reverse proxy, on a self-managed VPS:

```
services:
  crossgym-rag:
    build: ./crossgym-rag
    restart: unless-stopped
    env_file: ./crossgym-rag/.env
    networks: [web]
```

Caddy terminates TLS and reverse-proxies the public domain to the container on the internal
Docker network; n8n reaches it the same way any other client would — a plain HTTPS POST to
`/ask` with the API key in a header. The n8n workflow itself holds no application logic
beyond the Telegram plumbing (webhook trigger → load/save per-chat history → call the API →
reply) — all retrieval, fusion, and generation logic lives in this repo, not in the
workflow.

For local development or a single-container deployment elsewhere:
```
docker build -t crossgym-rag .
docker run -p 8000:8000 --env-file .env crossgym-rag
```
