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
scripts/
  build_corpus.py
src/
  vn_labor_law_ai_assistant/
tests/
  test_corpus_pipeline.py
```

## Cach chay

```powershell
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe scripts\build_corpus.py --curated-text corpus\cleaned\du_lieu_cham_dut_hop_dong_lao_dong.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe test_ollama.py
```

## Dau ra cua pipeline

Lenh build corpus se:

- doc tat ca file PDF trong `corpus/raw`;
- co the nap them cac file text da lam sach qua `--curated-text`;
- phat hien van ban nao co the extract text, van ban nao can OCR;
- ghi text da lam sach vao `corpus/cleaned`;
- tach thanh chunks JSONL trong `corpus/chunks`;
- ghi metadata vao `corpus/metadata`.

## Luu y du lieu hien tai

- `Nghi dinh 145/2020/ND-CP` la PDF text-based va da extract duoc.
- `VanBanGoc_BO LUAT 45 QH14.pdf` hien duoc phat hien la PDF scan/image, chua OCR duoc trong may hien tai, nen pipeline se gan co `needs_ocr`.

## Tai lieu quan trong

- Scope va KPI cho tuan 1-2: [docs/project_scope.md](C:/Workspace/vietnamese-labor-law-ai-assistant/docs/project_scope.md)
- Roadmap 8 tuan: [ROADMAP.md](C:/Workspace/vietnamese-labor-law-ai-assistant/ROADMAP.md)

## Buoc tiep theo sau tuan 1-2

- bo sung ban OCR hoac ban text-based cua Bo luat Lao dong 2019;
- tao vector index va truy hoi co trich dan;
- xay bo evaluation cho cac tinh huong cham dut hop dong lao dong.
