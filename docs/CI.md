# CI and Benchmark Checks

This project uses separate GitHub Actions workflows for backend checks, frontend checks, and manual RAG benchmark runs.

## Backend Checks

Run locally from the repository root:

```shell
python -m pip install -e ".[dev]"
ruff check .
mypy src/vn_labor_law_ai_assistant/core src/vn_labor_law_ai_assistant/api src/vn_labor_law_ai_assistant/auth src/vn_labor_law_ai_assistant/db
python -m unittest discover -s tests -v
```

The backend workflow runs the same commands on Python 3.10 and 3.11. It uses test-safe environment variables and dummy API keys, so pull request checks do not call Groq, Azure OpenAI, or Qdrant.

The initial Ruff gate is intentionally conservative: it checks `E` and `F` classes while ignoring legacy line-length and script import-order findings. TODO: expand Ruff coverage after a dedicated formatting cleanup.

The initial Mypy gate is also scoped to the core API/auth/db modules because a full `mypy src` currently reports broad legacy dynamic-payload typing issues. TODO: expand Mypy coverage module by module.

## Frontend Checks

Run locally:

```shell
cd frontend
npm install
npm run lint
npm run build
```

The frontend workflow uses Node 20 and `npm ci` with `frontend/package-lock.json`.

## Benchmark

Run a small local benchmark:

```shell
python scripts/run_benchmark.py --provider groq --model qwen/qwen3-32b --limit 10
```

For retrieval-only benchmark runs, omit `--model`:

```shell
python scripts/run_benchmark.py --limit 10
```

Benchmark runs can require API keys when answer generation or LLM-as-a-judge scoring is enabled. The GitHub Actions benchmark workflow is manual-only and defaults to retrieval-only mode, so it does not spend API credits on pull requests. When generation is enabled, the workflow expects the matching provider secrets to be configured.

The benchmark also needs a reachable vector index. If the checked-out `artifacts/index/current.json` points to local Qdrant files that are not present, configure the `QDRANT_URL` and optional `QDRANT_API_KEY` repository secrets before running the manual workflow.

Benchmark outputs are uploaded as workflow artifacts from:

- `eval/results/*.jsonl`
- `eval/results/*.csv`
