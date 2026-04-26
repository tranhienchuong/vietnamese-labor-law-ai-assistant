# AI Assistant for Vietnamese Labor Law

Tro ly AI hoi dap phap luat lao dong Viet Nam, truoc mat tap trung vao chu de `cham dut hop dong lao dong` cho case study.

## Muc tieu giai doan 2 thang

- Xay dung MVP RAG local, uu tien tinh chinh xac va kha nang trich dan.
- Chi tra loi trong pham vi du lieu phap ly da nap vao he thong.
- Ho tro case study ve:
  - don phuong cham dut hop dong lao dong;
  - thoi han bao truoc;
  - tro cap thoi viec, tro cap mat viec;
  - cac truong hop dac thu theo Nghi dinh 145/2020/ND-CP.

## Trang thai hien tai

- Da co du lieu goc trong [corpus/raw](C:/Workspace/vietnamese-labor-law-ai-assistant/corpus/raw).
- Da co smoke test cho Ollama trong [test_ollama.py](C:/Workspace/vietnamese-labor-law-ai-assistant/test_ollama.py).
- Da co smoke test cho Groq trong [test_groq.py](C:/Workspace/vietnamese-labor-law-ai-assistant/test_groq.py).
- Da them pipeline xay dung corpus o [scripts/build_corpus.py](C:/Workspace/vietnamese-labor-law-ai-assistant/scripts/build_corpus.py).

## Cau truc du an

```text
corpus/
  raw/        # PDF goc
  cleaned/    # text da chuan hoa
  chunks/     # JSONL chunks cho RAG
  metadata/   # metadata tung van ban va manifest tong
docs/
  project_scope.md
eval/
  data/       # benchmark JSONL da import
  results/    # ket qua benchmark local
scripts/
  ask.py
  analyze_benchmark_failures.py
  build_corpus.py
  build_index.py
  import_benchmark.py
  run_benchmark.py
src/
  vn_labor_law_ai_assistant/
    answering.py
    config.py
    evaluation.py
    indexing.py
    llm.py
    retriever.py
tests/
  test_answering.py
  test_corpus_pipeline.py
  test_evaluation.py
  test_indexing.py
  test_llm.py
  test_retriever.py
```

## Cach chay

```powershell
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe scripts\build_corpus.py --curated-text corpus\cleaned\du_lieu_cham_dut_hop_dong_lao_dong.txt corpus\cleaned\nghi-dinh-145-2020-nd-cp.txt
.venv\Scripts\python.exe scripts\build_index.py --dense-model keepitreal/vietnamese-sbert
.venv\Scripts\python.exe scripts\import_benchmark.py C:\Users\tranh\Downloads\golden_benchmark_template.xlsx
.venv\Scripts\python.exe scripts\run_benchmark.py --limit 10
.venv\Scripts\python.exe scripts\run_benchmark.py --provider ollama --model qwen3:4b --limit 10
.venv\Scripts\python.exe scripts\run_benchmark.py --provider ollama --model qwen3:4b --reranker-model BAAI/bge-reranker-v2-m3 --limit 10
.venv\Scripts\python.exe scripts\run_benchmark.py --provider ollama --model qwen3:4b --no-judge --limit 10
.venv\Scripts\python.exe scripts\run_benchmark.py --provider groq --model qwen/qwen3-32b --limit 10
.venv\Scripts\python.exe scripts\analyze_benchmark_failures.py eval\results\benchmark_ollama-qwen3-4b_20260419_175749.csv --failure-type retrieval_miss --limit 5
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\ask.py --retrieve-only "tro cap thoi viec tinh the nao theo Dieu 46?"
.venv\Scripts\python.exe scripts\ask.py --retrieve-only --reranker-model BAAI/bge-reranker-v2-m3 "tro cap thoi viec tinh the nao theo Dieu 46?"
.venv\Scripts\python.exe scripts\ask.py --provider ollama --model qwen3:4b "tro cap thoi viec tinh the nao theo Dieu 46?"
.venv\Scripts\python.exe scripts\ask.py --provider groq --model qwen/qwen3-32b "tro cap thoi viec tinh the nao theo Dieu 46?"
.venv\Scripts\python.exe test_ollama.py
.venv\Scripts\python.exe test_groq.py
```

Bien moi truong:

- Sua file `.env` o root project de dat bien moi truong local.
- `OLLAMA_MODEL`: model mac dinh khi dung provider `ollama`.
- `GROQ_MODEL`: model mac dinh khi dung provider `groq`.
- `GROQ_API_KEY`: API key de goi Groq.
- `LLM_PROVIDER`: provider mac dinh cho `scripts/ask.py` va `scripts/run_benchmark.py`.
- `BENCHMARK_JUDGE_PROVIDER`: provider mac dinh cho LLM judge trong `scripts/run_benchmark.py` khi co `--model` va khong dung `--no-judge`. Mac dinh la `groq`.
- `BENCHMARK_JUDGE_MODEL`: model mac dinh cho LLM judge. Neu de trong va judge provider la `groq`, script se dung `openai/gpt-oss-120b`.
- `RERANKER_MODEL`: model cross-encoder reranker tuy chon, vi du `BAAI/bge-reranker-v2-m3`. De trong neu muon tat semantic reranking.
- `RERANKER_TOP_N`: so candidate top dau duoc dua qua reranker. Mac dinh la `24`.
- `EMBEDDING_PROVIDER`: `sentence_transformers` de encode local, hoac `custom_http` de goi embedding API rieng.
- `EMBEDDING_API_URL`: endpoint HTTP khi dung `EMBEDDING_PROVIDER=custom_http`, vi du `https://your-space.hf.space/v1/embeddings`.
- `EMBEDDING_API_TOKEN`: optional Bearer token neu embedding API/Space de private.
- `EMBEDDING_API_TIMEOUT_SECONDS`: timeout cho embedding API. Mac dinh la `60`.
- `QDRANT_URL`: URL Qdrant server/cloud. Neu de trong, he thong fallback ve Qdrant embedded local.
- `QDRANT_API_KEY`: API key cho Qdrant Cloud.
- `QDRANT_COLLECTION`: ten collection Qdrant dung khi build/query.
- `RETRIEVER_RECORD_SOURCE`: `sqlite` cho mode local cu, hoac `qdrant_payload` de runtime doc full text truc tiep tu Qdrant payload.
- `INDEX_PATH`: duong dan manifest index cho backend API. Mac dinh la `artifacts/index`.
- Khi doi model embedding runtime, phai rebuild lai Qdrant index bang dung `DENSE_MODEL` do.

## Dau ra cua pipeline

Lenh build corpus se:

- doc tat ca file PDF trong `corpus/raw`;
- co the nap them cac file text da lam sach qua `--curated-text`;
- neu `curated_text` trung `document_id` voi PDF goc thi ban da lam sach se duoc uu tien, khong bi ghi de boi du lieu tho;
- phat hien van ban nao co the extract text, van ban nao can OCR;
- ghi text da lam sach vao `corpus/cleaned`;
- tach thanh chunks JSONL trong `corpus/chunks`;
- ghi metadata vao `corpus/metadata`.

## Dau ra indexing

Lenh build hybrid index se:

- doc toan bo chunk JSONL trong `corpus/chunks`;
- tao `dense_text` va `sparse_text` cho tung chunk;
- sinh dense embedding bang sentence-transformers;
- word-segment tieng Viet bang PyVi va tao sparse vector co trong so BM25;
- luu dense + sparse vectors trong cung mot Qdrant collection local hoac Qdrant server/cloud neu co `QDRANT_URL`;
- luu full text, dense_text, sparse_text va citation_text vao Qdrant payload de backend co the dung `RETRIEVER_RECORD_SOURCE=qdrant_payload`;
- luu canonical records vao `artifacts/index/builds/build_<timestamp>/records.db`;
- cap nhat con tro `artifacts/index/current.json` sau khi build thanh cong.

Backend API FastAPI:

```powershell
.venv\Scripts\python.exe -m uvicorn vn_labor_law_ai_assistant.api:app --host 0.0.0.0 --port 8000
```

Lenh `scripts/ask.py` se:

- route cau hoi thanh cac metadata heuristic nhu `actor`, `topic`, `issue_type`, `dieu/khoan/diem`;
- chay hybrid search `dense + sparse + RRF` trong Qdrant;
- co the chay them semantic re-ranker cross-encoder neu truyen `--reranker-model`;
- dedup small-to-big context qua `parent_chunk_id` bang SQLite;
- uu tien giu nguyen tung block context va loai bot block diem thap theo budget token/char thay vi cat cut ngang dieu khoan;
- gui context da loc cho provider LLM duoc chon (`ollama` hoac `groq`) voi guardrails citation;
- bo sung few-shot prompting de model format cau tra loi on dinh hon;
- chi chap nhan `legal_basis` nam trong danh sach `citation_text` da retrieve.

## Evaluation

Lenh `scripts/import_benchmark.py` se:

- doc workbook benchmark `.xlsx`;
- tim dong header thuc te trong sheet `golden_benchmark`;
- convert 100 cau hoi ve JSONL repo-native trong `eval/data/`.

Lenh `scripts/run_benchmark.py` se:

- tai benchmark JSONL da import;
- chay retriever hien tai tren tung cau hoi;
- co the bat semantic re-ranker cross-encoder bang `--reranker-model` de thu nghiem giam false negative o retrieval;
- ghi cot `retrieval_hit_at_<top_k>` va cac citation retrieve duoc;
- neu co `--model`, chay them generation qua provider da chon va luu output de review;
- mac dinh se bat them LLM-as-a-judge bang provider/model rieng de cham `answer_correct`, legal reasoning, missing-information handling, groundedness va format;
- judge co them rubric `groundedness_score_1_5` de phat manh cau tra loi vuot qua evidence va citation duoc cap;
- judge cham them `legal_issue_classification_correct`, `legal_reasoning_score_1_5`, `missing_information_score_0_2`, `citation_supports_answer` va `hallucination_types`;
- citation duoc tach thanh `citation_document_correct` va `citation_provision_correct`; cot `citation_article_correct` va `citation_correct` duoc giu nhu alias tuong thich;
- `abstention_correct` se phat ca under-abstention va over-abstention: case can thieu du kien ma model khong abstain la `no`, case khong can abstain ma model lai tu choi cung la `no`;
- `skill_tag` duoc gan tu dong de nhom loi theo ky nang nhu `legal_classification`, `missing_fact_handling`, `procedure_checking`, `remedy_calculation`, `rule_lookup`;
- `final_score_10` duoc tinh bang cong thuc ban tu dong tu issue classification, answer correctness, citation provision + support, missing-information handling hoac legal reasoning, groundedness va clarity/format;
- neu muon chi sinh cau tra loi ma khong cham bang judge, them `--no-judge`;
- ghi ket qua ra `eval/results/*.jsonl` va `eval/results/*.csv`, trong do ten file co kem `provider:model` da duoc slugify de de tach tung run.

De so sanh model, chay `scripts/run_benchmark.py` rieng cho tung provider/model va doi chieu cac file ket qua trong `eval/results/`.

## Luu y runtime

- Qdrant local khoa thu muc du lieu theo tung process. Khong nen chay hai lenh `ask.py` hoac `build_index.py` song song cung mot build.
- Neu can truy cap dong thoi, nen chuyen sang Qdrant server mode thay vi local embedded mode.

## Luu y du lieu hien tai

- `du_lieu_cham_dut_hop_dong_lao_dong.txt` dang la curated source cho Bo luat Lao dong 2019.
- `nghi-dinh-145-2020-nd-cp.txt` dang la curated source cho Nghi dinh 145/2020/ND-CP.
- Khi build voi hai file cleaned tren, manifest hien tai se ra `2 ready`, `0 needs_ocr`.

## Tai lieu quan trong

- Scope va KPI cho tuan 1-2: [docs/project_scope.md](C:/Workspace/vietnamese-labor-law-ai-assistant/docs/project_scope.md)
- Roadmap 8 tuan: [ROADMAP.md](C:/Workspace/vietnamese-labor-law-ai-assistant/ROADMAP.md)

## Buoc tiep theo

- danh gia retrieval chat luong tren bo cau hoi thuc te;
- tinh chinh heuristic query router de giam false negative voi cau hoi hoi thoai tu nhien;
- bo sung re-ranking va evaluation metrics cho dense / sparse / hybrid;
- them giao dien demo nho hoac API cho luong hoi dap.
