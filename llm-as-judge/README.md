# RAGAS Evaluation with TheSparkDaily Judge

This folder contains the standalone RAGAS evaluation pipeline for the Vietnamese labor-law RAG system. The evaluation code is kept outside the production RAG code so it can be run for experiments without changing retrieval or generation behavior.

## Current Project Layout

- RAG pipeline: `src/vn_labor_law_ai_assistant/rag/retrieval/*` and `src/vn_labor_law_ai_assistant/rag/answering/*`
- Runtime config: `src/vn_labor_law_ai_assistant/core/config.py`
- App LLM: `src/vn_labor_law_ai_assistant/llm.py`
- Embeddings: `src/vn_labor_law_ai_assistant/embeddings.py`
- Benchmark data: `eval/data/*.jsonl`
- Python: `>=3.10`
- Dependencies: `pyproject.toml`

## Install

Run this once from the repository root so both the app package and evaluation dependencies are importable:

```bash
python -m pip install -e ".[dev]"
```

Then run evaluation commands from `llm-as-judge`:

```bash
cd llm-as-judge
```

The supported namespace is `ragas_eval`.

## Configure TheSparkDaily

Set these variables in your shell or in the repository `.env` file:

```bash
export THESPARKDAILY_API_KEY="your_api_key"
export THESPARKDAILY_BASE_URL="https://api.thesparkdaily.com/v1"
export THESPARKDAILY_JUDGE_MODEL="gpt-5.4-pro"
export THESPARKDAILY_FAST_JUDGE_MODEL="gpt-5.4-nano"
```

PowerShell:

```powershell
$env:THESPARKDAILY_API_KEY="your_api_key"
```

API keys are read from environment variables and are not printed.

## RAGAS Embeddings

`answer_relevancy` and `answer_correctness` need embeddings. This pipeline passes an explicit embedding adapter to RAGAS instead of relying on defaults.

The adapter uses the project's existing embedding config:

- `EMBEDDING_PROVIDER=sentence_transformers`: loads `DENSE_MODEL`
- `EMBEDDING_PROVIDER=custom_http`: calls the configured `EMBEDDING_API_URL`

Optional RAGAS overrides:

```bash
export RAGAS_EMBEDDING_MODEL="keepitreal/vietnamese-sbert"
export RAGAS_EMBEDDING_DEVICE="cpu"
export RAGAS_EMBEDDING_BATCH_SIZE="32"
```

## Dataset Format

Evaluation runs require `response` and, for RAGAS modes, `retrieved_contexts`:

```json
{
  "id": "LBR_001",
  "user_input": "Cau hoi luat lao dong",
  "response": "Cau tra loi do RAG system sinh ra",
  "retrieved_contexts": ["chunk 1", "chunk 2"],
  "reference": "Cau tra loi chuan",
  "reference_contexts": ["chunk luat chuan"],
  "gold_citation": "Dieu/Khoan luat chuan"
}
```

Legacy field mappings:

- `question` -> `user_input`
- `answer`, `gold_answer`, `gold_answer_full` -> `reference`
- `generated_answer` -> `response`
- `contexts` -> `retrieved_contexts`

## Validate Dataset

Validate a benchmark without requiring generated responses:

```bash
python -m ragas_eval.dataset_loader \
  --input ../eval/data/golden_benchmark_100_answered_v2.jsonl \
  --validate-only
```

Validate that a dataset is ready for RAGAS scoring:

```bash
python -m ragas_eval.dataset_loader \
  --input path/to/benchmark_with_responses.jsonl \
  --require-response \
  --require-contexts
```

## Dry Run

Dry-run validates the selected rows and makes no API calls:

```bash
python -m ragas_eval.run_ragas_eval \
  --input ../eval/data/golden_benchmark_100_answered_v2.jsonl \
  --limit 2 \
  --mode full \
  --dry-run
```

If the selected dataset rows do not have `response` or `retrieved_contexts`, this command exits with schema errors that name the sample IDs and missing fields.

## Run Evaluation

Fast mode uses the fast judge model for lighter RAGAS metrics:

```bash
python -m ragas_eval.run_ragas_eval \
  --input path/to/benchmark_with_responses.jsonl \
  --output outputs/ragas_results_fast.csv \
  --limit 10 \
  --mode fast
```

Accurate mode runs only the custom legal judge with the accurate model:

```bash
python -m ragas_eval.run_ragas_eval \
  --input path/to/benchmark_with_responses.jsonl \
  --output outputs/ragas_results_legal.csv \
  --mode accurate
```

Full mode runs RAGAS metrics with the fast judge model and the legal judge with the accurate model:

```bash
python -m ragas_eval.run_ragas_eval \
  --input path/to/benchmark_with_responses.jsonl \
  --output outputs/ragas_results_full.csv \
  --mode full \
  --judge-model gpt-5.4-pro
```

## Outputs

By default, results and summaries are written under:

```text
llm-as-judge/outputs/
```

Typical files:

- `ragas_results_YYYYMMDD_HHMM.csv`
- `ragas_summary_YYYYMMDD_HHMM.json`
- `ragas_summary_YYYYMMDD_HHMM.md`

Regenerate a summary:

```bash
python -m ragas_eval.summarize_results \
  --input outputs/ragas_results_full.csv \
  --dataset path/to/benchmark_with_responses.jsonl \
  --judge-model gpt-5.4-pro
```

## API Smoke Test

`test_api.py` uses the same `THESPARKDAILY_*` config as the evaluator:

```bash
python test_api.py
```
