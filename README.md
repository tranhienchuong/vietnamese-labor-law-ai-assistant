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
.venv\Scripts\python.exe scripts\import_benchmark.py C:\Users\tranh\Downloads\golden_benchmark_100_answered_v1.xlsx
.venv\Scripts\python.exe scripts\run_benchmark.py --limit 10
.venv\Scripts\python.exe scripts\run_benchmark.py --provider ollama --model qwen3:4b --limit 10
.venv\Scripts\python.exe scripts\run_benchmark.py --provider groq --model openai/gpt-oss-20b --limit 10
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\ask.py --retrieve-only "tro cap thoi viec tinh the nao theo Dieu 46?"
.venv\Scripts\python.exe scripts\ask.py --provider ollama --model qwen3:4b "tro cap thoi viec tinh the nao theo Dieu 46?"
.venv\Scripts\python.exe scripts\ask.py --provider groq --model openai/gpt-oss-20b "tro cap thoi viec tinh the nao theo Dieu 46?"
.venv\Scripts\python.exe test_ollama.py
.venv\Scripts\python.exe test_groq.py
```

Bien moi truong:

- Sua file `.env` o root project de dat bien moi truong local.
- `OLLAMA_MODEL`: model mac dinh khi dung provider `ollama`.
- `GROQ_MODEL`: model mac dinh khi dung provider `groq`.
- `GROQ_API_KEY`: API key de goi Groq.
- `LLM_PROVIDER`: provider mac dinh cho `scripts/ask.py` va `scripts/run_benchmark.py`.

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
- luu dense + sparse vectors trong cung mot Qdrant collection local;
- luu canonical records vao `artifacts/index/builds/build_<timestamp>/records.db`;
- cap nhat con tro `artifacts/index/current.json` sau khi build thanh cong.

Lenh `scripts/ask.py` se:

- route cau hoi thanh cac metadata heuristic nhu `actor`, `topic`, `issue_type`, `dieu/khoan/diem`;
- chay hybrid search `dense + sparse + RRF` trong Qdrant;
- dedup small-to-big context qua `parent_chunk_id` bang SQLite;
- gui context da loc cho provider LLM duoc chon (`ollama` hoac `groq`) voi guardrails citation;
- chi chap nhan `legal_basis` nam trong danh sach `citation_text` da retrieve.

## Evaluation

Lenh `scripts/import_benchmark.py` se:

- doc workbook benchmark `.xlsx`;
- tim dong header thuc te trong sheet `golden_benchmark`;
- convert 100 cau hoi ve JSONL repo-native trong `eval/data/`.

Lenh `scripts/run_benchmark.py` se:

- tai benchmark JSONL da import;
- chay retriever hien tai tren tung cau hoi;
- ghi `retrieval_hit_at_5` va cac citation retrieve duoc;
- neu co `--model`, chay them generation qua provider da chon va luu output de review;
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
