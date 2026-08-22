# CrossGYM RAG Assistant

A retrieval-augmented generation (RAG) API that answers questions about a gym chain
(locations, pricing, trainers, booking policy) using its real knowledge base. Built as a
portfolio project to demonstrate production-style RAG engineering: hybrid retrieval,
citation-backed answers, and an evaluation harness — not just a prompt wrapped in an API.

The knowledge base is real: two CrossGYM locations (Sotsmisto, Skhidnyi-2) in Kryvyi Rih,
Ukraine, with their actual pricing and trainer rosters. Content is in Ukrainian; this README
and the code are in English.

## Why these design choices

**Hybrid retrieval (vector + keyword), not vector-only.** Pure embedding search misses exact
matches on names, prices, and Instagram handles — short, low-context strings that don't embed
well semantically. Postgres trigram search (`pg_trgm`) catches those; vector search catches
paraphrased questions. Both run as independent ranked lists and get fused with
**Reciprocal Rank Fusion (RRF)**, a rank-based combination that doesn't require normalizing
incomparable similarity scores (cosine distance vs. trigram similarity) onto the same scale —
a common source of bugs when combining heterogeneous retrievers naively.

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

## Architecture

```
                     ┌─────────────────────┐
   POST /ask ──────► │   FastAPI (main.py)  │
   (question)        └──────────┬──────────┘
                                 │
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
├── src/
│   ├── config.py     # env-based settings
│   ├── chunking.py    # pure text chunking, paragraph-aware
│   ├── embeddings.py  # sentence-transformers wrapper (multilingual e5)
│   ├── db.py           # Supabase client factory
│   ├── ingest.py       # loads data/kb, chunks, embeds, upserts to Supabase
│   ├── retrieval.py    # hybrid search + RRF fusion
│   ├── generation.py   # prompt building + OpenRouter call
│   └── main.py          # FastAPI app: POST /ask, GET /health
├── eval/
│   ├── qa_pairs.json    # 18 real Q&A pairs over the knowledge base
│   └── run_eval.py       # retrieval recall@k, optional generation faithfulness check
└── tests/                 # pytest: chunking, RRF fusion, generation (mocked)
```

## Setup

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
python3 -m pytest                  # unit tests: chunking, RRF fusion, generation (mocked)
python3 -m eval.run_eval             # retrieval recall@5 against 18 real Q&A pairs
python3 -m eval.run_eval --with-generation   # also checks generated answers (calls OpenRouter)
```

## Deployment

```
docker build -t crossgym-rag .
docker run -p 8000:8000 --env-file .env crossgym-rag
```
