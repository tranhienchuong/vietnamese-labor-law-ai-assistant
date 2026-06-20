# Vietnamese Labor Law GraphRAG QA

Vietnamese Labor Law GraphRAG QA is a scoped legal information assistant for
question answering over a fixed Vietnamese labor-law corpus. It combines Qdrant
hybrid dense/sparse retrieval with a Neo4j legal knowledge graph, then generates
answers only from retrieved context and validates citations deterministically.

This project is not a legal advice tool. It does not claim legal correctness
beyond the indexed corpus. It is intended to help locate, assemble, and verify
information from the scoped documents used in the thesis report:
*Graph-Augmented Retrieval-Augmented Generation for Vietnamese Labor Law Question
Answering*.

## Scope

The official corpus contains exactly these six document groups:

| Document ID | Document |
| --- | --- |
| `45-2019-qh14` | Labor Code 2019 |
| `92-2015-qh13-labor-only` | Civil Procedure Code 2015, labor-only subset |
| `nghi-dinh-135-2020-nd-cp` | Decree 135/2020/ND-CP |
| `nghi-dinh-145-2020-nd-cp` | Decree 145/2020/ND-CP |
| `thong-tu-09-2020-tt-bldtbxh` | Circular 09/2020/TT-BLDTBXH |
| `thong-tu-10-2020-tt-bldtbxh` | Circular 10/2020/TT-BLDTBXH |

Questions outside this indexed corpus should return an insufficient-context
response instead of unsupported legal conclusions.

## Architecture

```text
User query
  -> query routing / intent gating
  -> Qdrant hybrid dense+sparse seed retrieval
  -> Neo4j legal graph expansion
  -> coordinate/reference fallback for explicit article references
  -> deduplication, reranking, and context budgeting
  -> grounded answer generation
  -> deterministic citation validation
  -> final answer or insufficient-context response
```

The final context assembly follows the thesis formula:

```text
C_final = Rerank(S_k ∪ C_graph ∪ C_fallback)
```

Where `S_k` is the Qdrant seed set, `C_graph` is the Neo4j expansion context,
and `C_fallback` is the direct coordinate/reference fallback context.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the implementation mapping.

## Pipeline

The official offline pipeline is:

1. Legal document preprocessing and Unicode/text cleaning.
2. Hierarchy-aware legal chunking by document, article, clause, point, and appendix/table.
3. Semantic preservation through `retrieval_text`, `citation_text`, and structured metadata.
4. Metadata enrichment for document type, normative rank, topics, actors, and issue types.
5. Cross-reference extraction and resolution.
6. Qdrant hybrid dense/sparse indexing.
7. Neo4j legal graph construction.
8. Retrieval ablation and deterministic end-to-end evaluation.

## Current Official Artifacts

The current official generated artifacts are:

- `artifacts/chunks/legal_chunks_enriched.jsonl`
- `artifacts/index/current.json`
- `artifacts/graph/legal_graph_build_summary.json`
- `artifacts/evaluation/golden_benchmark_100_extended.jsonl`

Older outputs may be retained under `archive/legacy/` for auditability, but they
are not the source of truth for the thesis-aligned runtime.

## Getting Started

Install backend dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Configure environment variables using `.env.example` as a template. For local
retrieval without external services, `artifacts/index/current.json` can point to
local Qdrant artifacts. For Qdrant Cloud, set `QDRANT_URL`, `QDRANT_API_KEY`, and
`QDRANT_COLLECTION`.

Start Neo4j when rebuilding or using graph expansion:

```powershell
docker compose -f docker-compose.neo4j.yml up -d
```

Start the backend:

```powershell
.venv\Scripts\python.exe -m uvicorn vn_labor_law_ai_assistant.api:app --reload --host 0.0.0.0 --port 8000
```

Run the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Default local URLs:

- Backend API: `http://localhost:8000`
- Backend docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:5173`

## Supabase Google Auth Deployment

The production web architecture is:

- Frontend: Vite + React + TypeScript on Vercel
- Auth: Supabase Auth with Google OAuth
- App data: Supabase Postgres
- Backend: existing Python/FastAPI app on Render
- Retrieval: Qdrant and the existing index artifacts

The frontend authenticates with Supabase and sends the Supabase access token to
FastAPI:

```http
Authorization: Bearer <supabase_access_token>
```

FastAPI keeps local email/password auth by default. Set `AUTH_PROVIDER=supabase`
only in environments that should accept Supabase Auth tokens.

Supabase Postgres stores production app data when `APP_DATA_BACKEND=supabase`:

- `profiles`
- `conversations`
- `messages`
- `chat_traces`

Apply the schema in `supabase/migrations/001_app_schema.sql` before enabling the
production backend. No migration from the old SQLite users table is required.
When `APP_DATA_BACKEND=supabase`, `artifacts/app.db` is not used for production
app data. Qdrant and `artifacts/index` are still used for retrieval and should
not be deleted.

Frontend environment variables:

```text
VITE_SUPABASE_URL=https://your-project-ref.supabase.co
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_BASE_URL=https://your-render-backend.onrender.com
```

Backend Render environment variables:

```text
APP_ENV=production
AUTH_PROVIDER=supabase
APP_DATA_BACKEND=supabase
AUTH_SEED_DEFAULT_USERS=0
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
SUPABASE_DB_URL=postgresql://postgres.your-project-ref:password@aws-0-region.pooler.supabase.com:6543/postgres
ADMIN_EMAILS=your@email.com,another@email.com
CORS_ALLOW_ORIGINS=https://trolyluatlaodong.live,https://your-vercel-domain.vercel.app
```

Keep existing backend secrets such as `GROQ_API_KEY`, `QDRANT_URL`,
`QDRANT_API_KEY`, and `AUTH_SECRET` configured on Render. Do not commit Google
client secrets, Supabase service role keys, database passwords, or API keys.

Supabase URL configuration:

```text
Site URL:
https://trolyluatlaodong.live

Redirect URLs:
https://trolyluatlaodong.live/auth/callback
http://localhost:5173/auth/callback
```

Google OAuth should use the Supabase callback URL as the authorized redirect URI:

```text
https://<project-ref>.supabase.co/auth/v1/callback
```

## Rebuild Commands

Validate curated text:

```powershell
.venv\Scripts\python.exe scripts\validate_curated_legal_texts.py
```

Build hierarchy-aware chunks:

```powershell
.venv\Scripts\python.exe scripts\build_legal_chunks.py
```

Enrich chunk metadata:

```powershell
.venv\Scripts\python.exe scripts\enrich_legal_chunks.py
```

Build cross-reference edges:

```powershell
.venv\Scripts\python.exe scripts\build_reference_edges.py
```

Build the Qdrant hybrid index:

```powershell
.venv\Scripts\python.exe scripts\build_index.py `
  --chunk-file artifacts\chunks\legal_chunks_enriched.jsonl `
  --collection-name vietnamese_labor_law_chunks
```

Build the Neo4j legal graph:

```powershell
.venv\Scripts\python.exe scripts\build_legal_graph.py `
  --index-path artifacts\index `
  --chunks-path artifacts\chunks\legal_chunks_enriched.jsonl `
  --reference-edges-path artifacts\graph\reference_edges.jsonl `
  --reset
```

Full reproducibility details are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Ask From CLI

Retrieve context only:

```powershell
.venv\Scripts\python.exe scripts\ask.py --retrieve-only "Nguoi 14 tuoi co duoc lam viec khong?"
```

Generate an answer using the configured provider:

```powershell
.venv\Scripts\python.exe scripts\ask.py --provider groq --model qwen/qwen3-32b "Hop dong lao dong can co nhung noi dung gi?"
```

## Evaluation

The thesis-aligned final evaluation is deterministic. It reports:

- Recall@10
- Required Citation Coverage
- Forbidden Citation Violation Rate
- adjusted end-to-end pass rate
- answer pass rate
- citation validation pass rate
- quality validation pass rate
- out-of-corpus QA pass rate

Run retrieval ablation on the 100-query benchmark:

```powershell
.venv\Scripts\python.exe scripts\ablation_retrieval_100.py `
  --benchmark-path artifacts\evaluation\golden_benchmark_100_extended.jsonl `
  --output-prefix ablation_retrieval_100_final_candidate
```

Run deterministic end-to-end evaluation:

```powershell
.venv\Scripts\python.exe scripts\evaluate_end_to_end_rag.py `
  --benchmark-path artifacts\evaluation\golden_benchmark_100_extended.jsonl `
  --output-prefix end_to_end_100_final_candidate
```

Compute adjusted split metrics for 94 in-corpus queries and 6 out-of-corpus
refusal tests:

```powershell
.venv\Scripts\python.exe scripts\compute_100_split_metrics.py `
  --results-path artifacts\evaluation\end_to_end_100_final_candidate_results.json `
  --output-json artifacts\evaluation\benchmark_100_scope_guard_split_metrics.json `
  --failed-cases-csv artifacts\evaluation\benchmark_100_scope_guard_adjusted_failed_cases.csv
```

Exploratory LLM-as-Judge or RAGAS experiments, if retained, are not the final
evaluation method for this project.

## Tests

Backend checks:

```powershell
.venv\Scripts\ruff.exe check .
.venv\Scripts\mypy.exe src/vn_labor_law_ai_assistant/core src/vn_labor_law_ai_assistant/api src/vn_labor_law_ai_assistant/auth src/vn_labor_law_ai_assistant/db
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Frontend checks:

```powershell
cd frontend
npm install
npm run lint
npm run build
```

## Project Structure

```text
artifacts/        Generated chunks, indexes, graph summaries, and evaluation outputs
corpus/           Raw, cleaned, and curated legal source texts
docs/             Architecture, reproducibility, and CI notes
frontend/         Vite React user interface for Vercel
scripts/          Pipeline, graph, indexing, QA, and evaluation commands
src/              Python backend package
tests/            Unit and integration-style tests with service mocks where possible
thesis/           Thesis source/report artifacts
archive/legacy/   Superseded exploratory artifacts retained for auditability
```

## Limitations

- The assistant is limited to the six indexed document groups.
- It provides legal information from retrieved sources, not professional legal advice.
- Deterministic citation validation checks grounding against retrieved context, not full legal correctness.
- Out-of-corpus behavior depends on scoped query routing and insufficient-context guards.
- External laws, later amendments, or updated regulations require corpus and index rebuilds before use.
