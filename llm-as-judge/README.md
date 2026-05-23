# RAGAS Evaluation with TheSparkDaily Judge

This folder contains the RAGAS evaluation pipeline for the Vietnamese labor-law RAG system. It is intentionally separate from the production retrieval and generation code under `src/vn_labor_law_ai_assistant`.

## Project Notes

- RAG pipeline: `src/vn_labor_law_ai_assistant/rag/retrieval/*` and `src/vn_labor_law_ai_assistant/rag/answering/*`.
- Retrieval config: `src/vn_labor_law_ai_assistant/core/config.py`, plus index metadata under `artifacts/index`.
- Generation config: current app LLM code is `src/vn_labor_law_ai_assistant/llm.py` and uses Groq.
- Embedding config: `src/vn_labor_law_ai_assistant/embeddings.py`, with `sentence_transformers` or `custom_http`.
- Benchmark data: current JSONL files live in `eval/data`.
- Python version: `pyproject.toml` requires Python `>=3.10`.
- Dependency manager: `pyproject.toml` with setuptools.

## Install

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

## Configure TheSparkDaily

Set the API key in your environment or in the repo `.env` file:

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

## Dataset Format

Preferred JSONL sample:

```json
{
  "id": "LBR_001",
  "user_input": "Câu hỏi luật lao động",
  "response": "Câu trả lời do RAG system sinh ra",
  "retrieved_contexts": ["chunk 1", "chunk 2"],
  "reference": "Câu trả lời chuẩn",
  "reference_contexts": ["chunk luật chuẩn"],
  "gold_citation": "Điều/Khoản luật chuẩn"
}
```

The loader also maps older fields:

- `question` -> `user_input`
- `answer`, `gold_answer`, `gold_answer_full` -> `reference`
- `generated_answer` -> `response`
- `contexts` -> `retrieved_contexts`

## Validate Benchmark

From `llm-as-judge`:

```bash
python -m ragas_eval.dataset_loader \
  --input ../eval/data/golden_benchmark_100_answered_v2.jsonl \
  --validate-only
```

The compatibility namespace also works from `llm-as-judge`:

```bash
python -m evaluation.ragas_eval.dataset_loader \
  --input ../eval/data/golden_benchmark_100_answered_v2.jsonl \
  --validate-only
```

## Dry Run

Use this before setting an API key to confirm that an evaluation-ready dataset has `response` and `retrieved_contexts`:

```bash
python -m ragas_eval.run_ragas_eval \
  --input path/to/benchmark_with_responses.jsonl \
  --limit 10 \
  --mode full \
  --dry-run
```

## Run Evaluation

Fast mode uses the fast judge model for cheaper RAGAS metrics:

```bash
python -m ragas_eval.run_ragas_eval \
  --input path/to/benchmark_with_responses.jsonl \
  --output outputs/ragas_results.csv \
  --limit 10 \
  --mode fast
```

Accurate mode runs only the custom legal judge:

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

## Summarize Results

`run_ragas_eval.py` writes summary JSON and Markdown automatically. You can regenerate them:

```bash
python -m ragas_eval.summarize_results \
  --input outputs/ragas_results_full.csv \
  --dataset path/to/benchmark_with_responses.jsonl \
  --judge-model gpt-5.4-pro
```

The Markdown summary includes average metrics, legal judge metrics, error-type distribution, and the 10 lowest-scoring samples.
