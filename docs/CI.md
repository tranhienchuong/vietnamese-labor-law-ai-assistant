# CI Checks

The repository has three GitHub Actions workflows:

- `backend.yml`: Python dependency install, Ruff, scoped mypy, and unit tests.
- `frontend.yml`: npm install, Next.js lint, and production build.
- `benchmark.yml`: manual retrieval benchmark workflow. It requires an indexed corpus and may require provider secrets for answer-generation runs.

## Backend

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\ruff.exe check .
.venv\Scripts\mypy.exe src/vn_labor_law_ai_assistant/core src/vn_labor_law_ai_assistant/api src/vn_labor_law_ai_assistant/auth src/vn_labor_law_ai_assistant/db
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

CI uses safe test defaults:

```env
APP_ENV=test
AUTH_SECRET=test-secret-for-ci
AUTH_SEED_DEFAULT_USERS=0
APP_DB_PATH=artifacts/test-app.db
GROQ_API_KEY=dummy
QDRANT_API_KEY=dummy
QUERY_ROUTER_ENABLED=0
RERANKER_MODEL=
```

## Frontend

```powershell
cd frontend
npm ci --include=dev
npm run lint
npm run build
npm audit
```

`BACKEND_URL` should point to the backend base URL, for example:

```env
BACKEND_URL=http://localhost:8000
```

## Benchmark

The benchmark workflow is manual because it depends on the current index artifacts and optional provider credentials. Retrieval-only runs require a valid `artifacts/index/current.json` and either local Qdrant artifacts or a configured `QDRANT_URL`.

```powershell
.venv\Scripts\python.exe scripts\run_benchmark.py --limit 20 --output-dir eval\results
```

Answer-generation benchmark runs require an LLM provider key, for example `GROQ_API_KEY`.
