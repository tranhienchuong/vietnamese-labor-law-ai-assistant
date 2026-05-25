# Vietnamese Labor Law AI Assistant

RAG chatbot for answering Vietnamese labor-law questions with legal citations. The current scope focuses on employment contract termination, notice periods, severance allowance and related provisions in Vietnamese labor law.

## Architecture

### System Architecture

```text
Frontend (Next.js)
      |
      v
FastAPI Backend
      |
      +--> Hybrid Retriever
      |       |
      |       +--> Qdrant vector database
      |       +--> SQLite record store
      |
      +--> Groq LLM
              |
              +--> Grounded answer with citations
```

### RAG Pipeline

```text
Raw / cleaned legal documents
      |
      v
Corpus builder
      |
      v
Chunking + legal metadata
      |
      v
Dense embedding + sparse BM25/PyVi encoding
      |
      v
Qdrant hybrid index
      |
      v
Query routing + hybrid retrieval + optional reranking
      |
      v
Prompt assembly with citation guardrails
      |
      v
Groq answer generation
```

## Technology:

* Source control: [![GitHub][Github-logo]][Github-url]
* Backend: [![Python][Python-logo]][Python-url] [![FastAPI][FastAPI-logo]][FastAPI-url]
* Frontend: [![Next.js][Next-logo]][Next-url] [![React][React-logo]][React-url] [![TypeScript][TypeScript-logo]][TypeScript-url] [![Tailwind CSS][Tailwind-logo]][Tailwind-url]
* LLM Provider: [![Groq][Groq-logo]][Groq-url]
* Vector Database: [![Qdrant][Qdrant-logo]][Qdrant-url]
* Embeddings / NLP: [![HuggingFace][HuggingFace-logo]][HuggingFace-url] [![SentenceTransformers][SentenceTransformers-logo]][SentenceTransformers-url]
* Data processing: [![PyMuPDF][PyMuPDF-logo]][PyMuPDF-url] [![SQLite][SQLite-logo]][SQLite-url]
* Testing: [![unittest][Unittest-logo]][Unittest-url]

# Table of contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Getting Started](#getting-started)
- [API Endpoints](#api-endpoints)
- [Evaluation](#evaluation)
- [CI and Benchmark Checks](#ci-and-benchmark-checks)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

## Overview

This project builds a Retrieval Augmented Generation (RAG) assistant for Vietnamese labor-law questions. It indexes legal documents by article, clause and point, retrieves relevant legal context, then asks an LLM to answer only from the retrieved evidence.

The system is designed for a legal case study around employment contract termination. It prioritizes citation quality, grounded answers and measurable evaluation over open-ended legal advice.

The project no longer uses a local LLM runtime. Generation is handled through Groq API, while embeddings can be generated locally with Sentence Transformers or through a custom HTTP embedding API.

## Demo

Run the backend locally:

```powershell
.venv\Scripts\python.exe -m uvicorn vn_labor_law_ai_assistant.api:app --host 0.0.0.0 --port 8000
```

Run the frontend locally:

```powershell
cd frontend
npm run dev
```

Default local URLs:

- Backend API: http://localhost:8000
- Backend docs: http://localhost:8000/docs
- Frontend: http://localhost:3000

Example question:

```text
Trợ cấp thôi việc tính thế nào theo Điều 46?
```

## Features

- [X] Build corpus from raw or cleaned Vietnamese legal documents.
- [X] Split documents into legal chunks with article/clause/point metadata.
- [X] Hybrid retrieval with dense vectors, sparse vectors and RRF.
- [X] Qdrant-backed vector index with payload text for runtime retrieval.
- [X] Small-to-big context assembly through parent chunks.
- [X] Optional semantic reranking with cross-encoder models.
- [X] Groq-based answer generation with citation guardrails.
- [X] FastAPI chat endpoint for backend integration.
- [X] Next.js chat interface.
- [X] Benchmark import, evaluation and LLM-as-a-judge scoring.
- [ ] Add production monitoring dashboard.
- [X] Add CI checks and manual benchmark workflow.

## Getting Started

### Clone the repository

```shell
git clone <your-repository-url>
cd vietnamese-labor-law-ai-assistant
```

### Install backend dependencies

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e .
```

### Configure environment variables

Create a `.env` file in the repository root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3-32b

# Optional: Groq model for LLM-as-a-judge
BENCHMARK_JUDGE_PROVIDER=groq
BENCHMARK_JUDGE_MODEL=openai/gpt-oss-120b

QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=vietnamese_labor_law_chunks

EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_API_URL=
EMBEDDING_API_TOKEN=
EMBEDDING_API_TIMEOUT_SECONDS=60

RERANKER_MODEL=
RERANKER_TOP_N=24
RETRIEVER_RECORD_SOURCE=sqlite
INDEX_PATH=artifacts/index
CORS_ALLOW_ORIGINS=*
```

### Build corpus

```powershell
.venv\Scripts\python.exe scripts\build_corpus.py --curated-text corpus\cleaned\du_lieu_cham_dut_hop_dong_lao_dong.txt corpus\cleaned\nghi-dinh-145-2020-nd-cp.txt
```

### Build hybrid index

```powershell
.venv\Scripts\python.exe scripts\build_index.py --dense-model keepitreal/vietnamese-sbert
```

### Ask from CLI

Retrieve context only:

```powershell
.venv\Scripts\python.exe scripts\ask.py --retrieve-only "tro cap thoi viec tinh the nao theo Dieu 46?"
```

Generate an answer with Groq:

```powershell
.venv\Scripts\python.exe scripts\ask.py --provider groq --model openai/gpt-oss-120b "Khi nào người lao động được nghỉ việc mà không cần báo trước?"
```

### Start up your backend

```powershell
.venv\Scripts\python.exe -m uvicorn vn_labor_law_ai_assistant.api:app --reload --host 0.0.0.0 --port 8000
```

### Access your frontend

Install frontend dependencies:

```powershell
cd frontend
npm install
```

Create `frontend/.env.local` when the frontend should call the backend:

```env
BACKEND_URL=http://localhost:8000
```

Start the frontend:

```powershell
npm run dev
```

## API Endpoints

Available endpoints for backend:

| Methods | Functionality |
|:--------|:--------------|
| GET `/` | Health-style root status |
| GET `/health` | Backend health check |
| POST `/chat` | RAG chat endpoint with retrieval, optional retrieve-only mode and answer generation |

Example `/chat` request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Trợ cấp thôi việc tính thế nào theo Điều 46?"
    }
  ],
  "provider": "groq",
  "model": "qwen/qwen3-32b",
  "topK": 8,
  "maxContexts": 6,
  "includeCitations": true
}
```

## Evaluation

Import benchmark data from Excel:

```powershell
.venv\Scripts\python.exe scripts\import_benchmark.py path\to\golden_benchmark_template.xlsx
```

Run retrieval and answer-quality benchmark:

```powershell
.venv\Scripts\python.exe scripts\run_benchmark.py --provider groq --model qwen/qwen3-32b --limit 10
```

Use a Groq model as the LLM-as-a-judge:

```powershell
.venv\Scripts\python.exe scripts\run_benchmark.py --provider groq --model qwen/qwen3-32b --judge-provider groq --judge-model openai/gpt-oss-120b --limit 10
```

Run unit tests:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Benchmark outputs are written to `eval/results/*.jsonl` and `eval/results/*.csv`.

### Recommended Evaluation Flow

Use `scripts/run_benchmark.py` as the system runner, then use RAGAS/TheSparkDaily
as the final evaluator. The recommended path disables the legacy benchmark judge
while exporting a RAGAS-ready input file:

Step 1: generate answers and prompt contexts from the live RAG system.

```powershell
$env:LEGAL_GRAPH_ENABLED="false"

.venv\Scripts\python.exe scripts\run_benchmark.py `
  --benchmark-path llm-as-judge\evaluation\ragas_eval\labor_law_benchmark_pilot_30_eval_template.jsonl `
  --provider groq `
  --model qwen/qwen3-32b `
  --no-judge `
  --export-ragas `
  --ragas-output-path eval\results\pilot_30_vector_only_ragas_input.jsonl
```

Step 2: score the exported file with RAGAS/TheSparkDaily.

```powershell
cd llm-as-judge

..\.venv\Scripts\python.exe -m ragas_eval.run_ragas_eval `
  --input ..\eval\results\pilot_30_vector_only_ragas_input.jsonl `
  --mode full
```

Step 3: compare vector-only against graph-augmented retrieval by rerunning Step 1
with Neo4j graph expansion enabled, then scoring the second export with Step 2.

```powershell
$env:LEGAL_GRAPH_ENABLED="true"
$env:LEGAL_GRAPH_BACKEND="neo4j"

.venv\Scripts\python.exe scripts\run_benchmark.py `
  --benchmark-path llm-as-judge\evaluation\ragas_eval\labor_law_benchmark_pilot_30_eval_template.jsonl `
  --provider groq `
  --model qwen/qwen3-32b `
  --no-judge `
  --export-ragas `
  --ragas-output-path eval\results\pilot_30_vector_graph_ragas_input.jsonl
```

## CI and Benchmark Checks

Automation details are documented in [docs/CI.md](docs/CI.md).

Backend checks:

```powershell
python -m pip install -e ".[dev]"
ruff check .
mypy src/vn_labor_law_ai_assistant/core src/vn_labor_law_ai_assistant/api src/vn_labor_law_ai_assistant/auth src/vn_labor_law_ai_assistant/db
python -m unittest discover -s tests -v
```

Frontend checks:

```powershell
cd frontend
npm install
npm run lint
npm run build
```

Benchmark workflow runs manually in GitHub Actions and defaults to retrieval-only mode. Answer-generation benchmark runs may require provider API key secrets.

## Project Structure

```bash
├── corpus/
│   ├── raw/
│   ├── cleaned/
│   ├── chunks/
│   └── metadata/
│
├── eval/
│   ├── data/
│   └── results/
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── lib/
│   ├── package.json
│   └── next.config.mjs
│
├── my-embedding-api/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── scripts/
│   ├── ask.py
│   ├── analyze_benchmark_failures.py
│   ├── build_corpus.py
│   ├── build_index.py
│   ├── import_benchmark.py
│   └── run_benchmark.py
│
├── src/
│   └── vn_labor_law_ai_assistant/
│       ├── answering.py
│       ├── api.py
│       ├── config.py
│       ├── corpus_pipeline.py
│       ├── embeddings.py
│       ├── evaluation.py
│       ├── indexing.py
│       ├── llm.py
│       └── retriever.py
│
├── tests/
├── pyproject.toml
├── README.md
└── test_groq.py
```

## Acknowledgements

- Vietnamese labor-law source documents used for the RAG case study.
- Open-source Python, FastAPI, Qdrant, Hugging Face and Next.js ecosystems.

<!-- MARKDOWN LINKS & IMAGES -->
[Github-logo]: https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white
[Github-url]: https://github.com/

[Python-logo]: https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white
[Python-url]: https://www.python.org/

[FastAPI-logo]: https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white
[FastAPI-url]: https://fastapi.tiangolo.com/

[Next-logo]: https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/

[React-logo]: https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB
[React-url]: https://react.dev/

[TypeScript-logo]: https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white
[TypeScript-url]: https://www.typescriptlang.org/

[Tailwind-logo]: https://img.shields.io/badge/Tailwind_CSS-38B2AC?logo=tailwindcss&logoColor=white
[Tailwind-url]: https://tailwindcss.com/

[Groq-logo]: https://img.shields.io/badge/Groq-F55036?logo=groq&logoColor=white
[Groq-url]: https://groq.com/

[Qdrant-logo]: https://img.shields.io/badge/Qdrant-DC244C?logo=qdrant&logoColor=white
[Qdrant-url]: https://qdrant.tech/

[HuggingFace-logo]: https://img.shields.io/badge/HuggingFace-000000?logo=huggingface&logoColor=yellow
[HuggingFace-url]: https://huggingface.co/

[SentenceTransformers-logo]: https://img.shields.io/badge/SentenceTransformers-2B6CB0?logo=huggingface&logoColor=white
[SentenceTransformers-url]: https://www.sbert.net/

[PyMuPDF-logo]: https://img.shields.io/badge/PyMuPDF-306998?logo=python&logoColor=white
[PyMuPDF-url]: https://pymupdf.readthedocs.io/

[SQLite-logo]: https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white
[SQLite-url]: https://www.sqlite.org/

[Unittest-logo]: https://img.shields.io/badge/unittest-3776AB?logo=python&logoColor=white
[Unittest-url]: https://docs.python.org/3/library/unittest.html
