# Plan: Qdrant Cloud Level 3 - Store Vector + Full Text Payload

**Project:** Vietnamese Labor Law AI Assistant  
**Goal:** Chuyển hệ thống RAG sang kiến trúc cloud-friendly, trong đó Qdrant Cloud lưu cả vector, metadata và full text/context cần thiết để backend không còn phụ thuộc vào `records.db` khi truy vấn runtime.

---

## 1. Mục tiêu của Level 3

Hiện tại hệ thống dùng kiến trúc hybrid local:

```text
User question
  -> encode dense query
  -> encode sparse query
  -> search local Qdrant
  -> nhận chunk_id
  -> đọc records.db để lấy full text
  -> build prompt
  -> gọi LLM
  -> trả lời
```

Level 3 sẽ đổi thành:

```text
User question
  -> backend cloud
  -> encode dense query
  -> encode sparse query
  -> search Qdrant Cloud
  -> nhận full text + metadata trực tiếp từ payload
  -> build prompt
  -> gọi Groq/OpenAI/Gemini
  -> trả lời frontend
```

Mục tiêu chính:

- Không cần Qdrant embedded local ở runtime.
- Không cần `records.db` ở runtime.
- Lưu full text/context trong Qdrant Cloud payload.
- Backend có thể deploy lên Render/Railway/Fly.io.
- Máy cá nhân yếu vẫn có thể làm việc vì indexing có thể chạy trên Google Colab hoặc GitHub Actions.

---

## 2. Kiến trúc sau khi hoàn thành

```text
Frontend: Vercel
  |
  | POST /api/chat
  v
Backend: Render/Railway FastAPI
  |
  | encode query
  v
Qdrant Cloud
  | stores:
  | - dense vector
  | - sparse vector
  | - full text
  | - citation_text
  | - metadata
  v
Backend builds prompt
  |
  v
Groq API / Other LLM Provider
  |
  v
Frontend displays answer + citations
```

---

## 3. Những file chính cần sửa

```text
src/vn_labor_law_ai_assistant/indexing.py
src/vn_labor_law_ai_assistant/retriever.py
src/vn_labor_law_ai_assistant/api.py       # nếu chưa có thì tạo mới
scripts/build_index.py                     # nếu cần thêm flag upload cloud
pyproject.toml                             # thêm fastapi/uvicorn nếu chưa có
.env.example                               # thêm biến môi trường cloud
```

Có thể tạo thêm file mới:

```text
scripts/upload_qdrant_cloud.py
scripts/build_cloud_index.py
```

---

## 4. Environment Variables cần có

Backend runtime:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3-32b

QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=vietnamese_labor_law_chunks

RERANKER_MODEL=
RERANKER_TOP_N=24
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_API_URL=
EMBEDDING_API_TOKEN=
EMBEDDING_API_TIMEOUT_SECONDS=60
```

Build/indexing environment:

```env
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=vietnamese_labor_law_chunks
DENSE_MODEL=keepitreal/vietnamese-sbert
EMBEDDING_PROVIDER=sentence_transformers
```

If runtime embeddings come from the Hugging Face Space in `my-embedding-api`, set:

```env
EMBEDDING_PROVIDER=custom_http
EMBEDDING_API_URL=https://your-username-my-embedding-api.hf.space/v1/embeddings
EMBEDDING_API_TOKEN=
DENSE_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Then rebuild and upload the Qdrant collection with the same dense model used by the Space.

Frontend Vercel:

```env
BACKEND_URL=https://your-backend.onrender.com
```

---

## 5. Thay đổi 1: Lưu full text vào Qdrant payload

### 5.1 Hiện trạng

Trong `indexing.py`, khi upsert point vào Qdrant, payload hiện chủ yếu là metadata:

```python
payload=record.payload
```

Nhưng các field quan trọng như `text`, `dense_text`, `sparse_text` đang được lưu ở SQLite `records.db`.

### 5.2 Mục tiêu

Qdrant payload cần lưu đủ dữ liệu để retriever build prompt mà không cần SQLite.

Payload mới nên có:

```python
payload={
    **record.payload,
    "text": record.text,
    "dense_text": record.dense_text,
    "sparse_text": record.sparse_text,
    "citation_text": record.citation_text,
    "parent_chunk_id": record.parent_chunk_id,
}
```

### 5.3 Việc cần làm

Trong hàm `build_qdrant_collection`, sửa đoạn tạo `PointStruct`.

Từ:

```python
models.PointStruct(
    id=make_qdrant_point_id(record.chunk_id),
    vector={
        dense_vector_name: list(dense_vector),
        sparse_vector_name: models.SparseVector(
            indices=sparse_vector.indices,
            values=sparse_vector.values,
        ),
    },
    payload=record.payload,
)
```

Thành:

```python
models.PointStruct(
    id=make_qdrant_point_id(record.chunk_id),
    vector={
        dense_vector_name: list(dense_vector),
        sparse_vector_name: models.SparseVector(
            indices=sparse_vector.indices,
            values=sparse_vector.values,
        ),
    },
    payload={
        **record.payload,
        "text": record.text,
        "dense_text": record.dense_text,
        "sparse_text": record.sparse_text,
        "citation_text": record.citation_text,
        "parent_chunk_id": record.parent_chunk_id,
    },
)
```

---

## 6. Thay đổi 2: Hỗ trợ Qdrant Cloud client

### 6.1 Hiện trạng

Code hiện tại tạo Qdrant local embedded bằng `path`:

```python
client = qdrant_client_cls(path=str(qdrant_path))
```

### 6.2 Mục tiêu

Nếu có `QDRANT_URL`, dùng Qdrant Cloud. Nếu không có, fallback về local embedded để vẫn test local được.

### 6.3 Tạo helper function

Có thể thêm function này vào `indexing.py` hoặc module mới `qdrant_utils.py`:

```python
import os
from pathlib import Path


def build_qdrant_client(qdrant_client_cls, qdrant_path: Path | None = None):
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()

    if qdrant_url:
        return qdrant_client_cls(
            url=qdrant_url,
            api_key=qdrant_api_key or None,
        )

    if qdrant_path is None:
        raise ValueError("qdrant_path is required when QDRANT_URL is not set.")

    return qdrant_client_cls(path=str(qdrant_path))
```

Sau đó thay trong build index:

```python
client = build_qdrant_client(qdrant_client_cls, qdrant_path)
```

Và trong retriever:

```python
self._qdrant = build_qdrant_client(qdrant_client_cls, self._qdrant_path)
```

---

## 7. Thay đổi 3: Retriever không cần `records.db` ở runtime

### 7.1 Hiện trạng

Retriever hiện đang dùng SQLite:

```python
self._sqlite = sqlite3.connect(self._records_db_path)
```

và `_fetch_records()` đọc text từ bảng `records`.

### 7.2 Mục tiêu

Khi dùng Qdrant Cloud payload đầy đủ, retriever tạo `RetrievedRecord` trực tiếp từ `hit.payload`.

### 7.3 Thêm mode runtime

Thêm biến môi trường:

```env
RETRIEVER_RECORD_SOURCE=qdrant_payload
```

Các giá trị:

```text
sqlite          # mode cũ
qdrant_payload  # mode mới Level 3
```

### 7.4 Thêm function tạo record từ payload

Trong `retriever.py`:

```python
def record_from_qdrant_payload(payload: dict[str, object]) -> RetrievedRecord:
    return RetrievedRecord(
        chunk_id=str(payload["chunk_id"]),
        parent_chunk_id=str(payload.get("parent_chunk_id")) if payload.get("parent_chunk_id") else None,
        citation_text=str(payload.get("citation_text") or ""),
        text=str(payload.get("text") or ""),
        dense_text=str(payload.get("dense_text") or ""),
        sparse_text=str(payload.get("sparse_text") or ""),
        payload=payload,
    )
```

### 7.5 Sửa `_fetch_records`

Ý tưởng:

```python
def _fetch_records_from_hits(self, hits: Sequence[SearchHit]) -> dict[str, RetrievedRecord]:
    if self._record_source == "qdrant_payload":
        records = {}
        for hit in hits:
            record = record_from_qdrant_payload(hit.payload)
            records[record.chunk_id] = record
        return records

    return self._fetch_records([hit.chunk_id for hit in hits])
```

Sau đó trong các chỗ đang gọi:

```python
self._fetch_records(chunk_ids)
```

chuyển sang:

```python
self._fetch_records_from_hits(hits)
```

---

## 8. Thay đổi 4: Xử lý parent-child context không dùng SQLite

### 8.1 Vấn đề

Hiện tại hệ thống có logic small-to-big context qua `parent_chunk_id`. Nếu dùng SQLite, retriever có thể lấy parent record từ `records.db`.

Nếu bỏ SQLite, cần đảm bảo Qdrant payload có đủ parent context hoặc có cách lấy parent từ Qdrant.

### 8.2 Cách đơn giản nhất

Trong quá trình build payload, lưu luôn text đủ lớn vào mỗi point.

Có 2 lựa chọn:

#### Option A: Mỗi point lưu chính text của chunk

Dễ làm nhất, nhưng small-to-big có thể yếu hơn.

```python
"text": record.text
```

#### Option B: Mỗi point lưu thêm parent_text

Tốt hơn cho chất lượng answer.

Payload nên có thêm:

```python
"parent_text": parent_record.text if parent_record else record.text,
"parent_citation_text": parent_record.citation_text if parent_record else record.citation_text,
```

Sau đó khi build context, ưu tiên dùng `parent_text` nếu có.

### 8.3 Khuyến nghị

Bắt đầu với Option A để hệ thống chạy được trước. Sau đó nâng lên Option B nếu retrieval thiếu context.

---

## 9. Thay đổi 5: Manifest cloud-friendly

### 9.1 Hiện trạng

`current.json` đang lưu:

```json
{
  "qdrant_path": ".../qdrant",
  "records_db_path": ".../records.db",
  "sparse_encoder_path": ".../sparse_encoder.json"
}
```

### 9.2 Mục tiêu

Manifest mới vẫn giữ thông tin local fallback, nhưng thêm thông tin cloud:

```json
{
  "collection_name": "vietnamese_labor_law_chunks",
  "qdrant_storage": "cloud",
  "record_source": "qdrant_payload",
  "dense_model_name": "keepitreal/vietnamese-sbert",
  "dense_vector_name": "dense",
  "sparse_vector_name": "sparse",
  "sparse_encoder_path": "artifacts/index/builds/build_xxx/sparse_encoder.json"
}
```

### 9.3 Lưu ý

Ngay cả Level 3 vẫn cần `sparse_encoder.json` ở runtime nếu còn dùng sparse search, vì backend cần encode sparse query.

Có 3 cách xử lý `sparse_encoder.json`:

```text
1. Commit vào repo nếu file nhỏ
2. Đặt trong backend image/container
3. Tải từ cloud storage khi backend start
```

Với demo, chọn cách 1.

---

## 10. Build index trên Google Colab

Vì máy local yếu, dùng Colab để build và upload Qdrant Cloud.

### 10.1 Các bước Colab

```bash
git clone https://github.com/tranhienchuong/vietnamese-labor-law-ai-assistant.git
cd vietnamese-labor-law-ai-assistant
pip install -e .
```

Set env:

```python
import os
os.environ["QDRANT_URL"] = "https://your-cluster-url.qdrant.io"
os.environ["QDRANT_API_KEY"] = "your_qdrant_api_key"
os.environ["QDRANT_COLLECTION"] = "vietnamese_labor_law_chunks"
```

Build corpus nếu cần:

```bash
python scripts/build_corpus.py --curated-text corpus/cleaned/du_lieu_cham_dut_hop_dong_lao_dong.txt corpus/cleaned/nghi-dinh-145-2020-nd-cp.txt
```

Build index và upload cloud:

```bash
python scripts/build_index.py --dense-model keepitreal/vietnamese-sbert
```

Sau khi build xong, lấy các file cần giữ:

```text
artifacts/index/current.json
artifacts/index/builds/build_xxx/sparse_encoder.json
```

Nếu vẫn giữ fallback/local test, giữ thêm:

```text
artifacts/index/builds/build_xxx/index_manifest.json
artifacts/index/builds/build_xxx/records.jsonl
```

---

## 11. Backend API sau Level 3

Backend FastAPI nên có:

```text
GET /
POST /chat
GET /health
```

Request từ frontend:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Công ty đơn phương chấm dứt hợp đồng trái luật thì phải bồi thường gì?"
    }
  ],
  "mode": "legal_qa",
  "language": "vi",
  "includeCitations": true
}
```

Response text stream hoặc plain text:

```text
Theo Bộ luật Lao động 2019, ...

Cơ sở pháp lý:
- Bộ luật số 45/2019/QH14, Điều 41, khoản ...
```

Frontend hiện tại đọc text stream, nên backend có thể bắt đầu bằng `PlainTextResponse`. Sau này nâng lên `StreamingResponse`.

---

## 12. Deploy backend

### 12.1 Render settings

```text
Service Type: Web Service
Runtime: Python
Root Directory: root repo
Build Command: pip install -e .
Start Command: uvicorn vn_labor_law_ai_assistant.api:app --host 0.0.0.0 --port $PORT
```

### 12.2 Environment Variables trên Render

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3-32b

QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=vietnamese_labor_law_chunks
RETRIEVER_RECORD_SOURCE=qdrant_payload

INDEX_PATH=artifacts/index
RERANKER_MODEL=
RERANKER_TOP_N=24
```

---

## 13. Deploy frontend

Frontend đã có Next.js trong `frontend/`.

Vercel settings:

```text
Root Directory: frontend
Framework Preset: Next.js
Build Command: npm run build
Install Command: npm install
Output Directory: .next
```

Environment Variables trên Vercel:

```env
BACKEND_URL=https://your-backend.onrender.com
```

Sau khi thêm env, cần redeploy frontend.

---

## 14. Checklist triển khai

### Phase 1: Chuẩn bị

- [ ] Tạo Qdrant Cloud cluster.
- [ ] Tạo collection name chuẩn: `vietnamese_labor_law_chunks`.
- [ ] Tạo API key Qdrant Cloud.
- [ ] Tạo Groq API key.
- [ ] Thêm `.env.example` cho các biến môi trường mới.

### Phase 2: Sửa indexing

- [ ] Sửa `build_qdrant_collection` để payload chứa full text.
- [ ] Thêm helper `build_qdrant_client`.
- [ ] Hỗ trợ `QDRANT_URL` và `QDRANT_API_KEY`.
- [ ] Đảm bảo local fallback vẫn chạy nếu không có `QDRANT_URL`.
- [ ] Build thử index trên local hoặc Colab.
- [ ] Kiểm tra Qdrant Cloud có points và payload đầy đủ.

### Phase 3: Sửa retriever

- [ ] Thêm `RETRIEVER_RECORD_SOURCE=qdrant_payload`.
- [ ] Thêm `record_from_qdrant_payload`.
- [ ] Sửa logic fetch records để dùng payload thay vì SQLite.
- [ ] Đảm bảo query vẫn encode dense/sparse được.
- [ ] Test retrieve-only không dùng `records.db`.

### Phase 4: API backend

- [ ] Thêm FastAPI nếu chưa có.
- [ ] Tạo `src/vn_labor_law_ai_assistant/api.py`.
- [ ] Tạo endpoint `GET /`.
- [ ] Tạo endpoint `POST /chat`.
- [ ] Test local với `uvicorn`.
- [ ] Test frontend local gọi backend local.

### Phase 5: Cloud build/index

- [ ] Chạy build trên Google Colab.
- [ ] Upload points lên Qdrant Cloud.
- [ ] Tải hoặc commit `current.json`.
- [ ] Tải hoặc commit `sparse_encoder.json`.
- [ ] Không commit secret/API key.

### Phase 6: Deploy

- [ ] Deploy backend lên Render/Railway.
- [ ] Set env backend.
- [ ] Test `GET /` backend public.
- [ ] Test `POST /chat` backend public.
- [ ] Set `BACKEND_URL` trên Vercel.
- [ ] Redeploy frontend.
- [ ] Test `/chat` public.

---

## 15. Test cases bắt buộc

### Test 1: Health check

```bash
curl https://your-backend.onrender.com/
```

Expected:

```json
{"status":"ok"}
```

### Test 2: Chat API

```bash
curl -X POST https://your-backend.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Công ty đơn phương chấm dứt hợp đồng trái luật thì phải bồi thường gì?"
      }
    ],
    "mode": "legal_qa",
    "language": "vi",
    "includeCitations": true
  }'
```

Expected:

```text
Có câu trả lời tiếng Việt.
Có cơ sở pháp lý.
Không báo lỗi records.db.
Không báo lỗi Qdrant local lock.
```

### Test 3: Qdrant payload

Kiểm tra một point trong Qdrant Cloud phải có:

```json
{
  "chunk_id": "...",
  "citation_text": "...",
  "text": "...",
  "dense_text": "...",
  "sparse_text": "..."
}
```

### Test 4: Frontend public

Mở:

```text
https://your-frontend.vercel.app/chat
```

Hỏi:

```text
Người lao động hợp đồng không xác định thời hạn muốn nghỉ việc thì phải báo trước bao lâu?
```

Expected:

```text
Frontend nhận câu trả lời từ backend thật, không phải demo fallback.
```

---

## 16. Rủi ro và cách xử lý

### Rủi ro 1: Qdrant payload quá lớn

Nếu payload quá lớn, Qdrant search có thể chậm hoặc tốn dung lượng.

Cách xử lý:

- Chỉ lưu `text` đã chunk, không lưu toàn bộ document.
- Giới hạn chunk size hợp lý.
- Không lưu dữ liệu trùng lặp quá nhiều.
- Sau này chuyển full text sang Supabase/Postgres nếu corpus lớn.

### Rủi ro 2: Backend vẫn cần model embedding

Ngay cả khi dùng Qdrant Cloud, backend vẫn phải encode câu hỏi bằng sentence-transformers.

Cách xử lý:

- Dùng backend cloud có RAM đủ.
- Cache model khi app start.
- Dùng model embedding nhẹ hơn nếu cần.
- Sau này dùng embedding API cloud để không load model local.

### Rủi ro 3: Sparse encoder vẫn cần file local

Nếu còn hybrid search, backend vẫn cần `sparse_encoder.json`.

Cách xử lý:

- Commit `sparse_encoder.json` nếu nhỏ.
- Hoặc tải từ cloud storage khi backend start.
- Hoặc tạm tắt sparse search, chỉ dùng dense search cho MVP.

### Rủi ro 4: Render free tier sleep

Render free có thể sleep sau một thời gian không dùng.

Cách xử lý:

- Chấp nhận cold start cho demo.
- Hoặc dùng Railway/Fly.io/VPS.
- Hoặc dùng cron ping nếu được phép.

---

## 17. Lộ trình đề xuất

### Ngày 1

- Sửa `indexing.py` để payload chứa full text.
- Thêm Qdrant Cloud client helper.
- Test build local nhỏ hoặc Colab.

### Ngày 2

- Sửa `retriever.py` để đọc records từ Qdrant payload.
- Test `retrieve-only` không cần `records.db`.

### Ngày 3

- Tạo FastAPI backend `/chat`.
- Test frontend local gọi backend local.

### Ngày 4

- Build/upload index lên Qdrant Cloud bằng Colab.
- Deploy backend lên Render/Railway.

### Ngày 5

- Set `BACKEND_URL` trên Vercel.
- Test public end-to-end.
- Ghi lại lỗi và benchmark lại vài câu hỏi chính.

---

## 18. Acceptance Criteria

Level 3 được coi là hoàn thành khi:

- [ ] Qdrant Cloud có collection chứa dense vector, sparse vector và full text payload.
- [ ] Backend có thể trả lời mà không cần mở `records.db`.
- [ ] Backend dùng `QDRANT_URL` và `QDRANT_API_KEY`.
- [ ] Backend deploy được lên cloud.
- [ ] Frontend Vercel gọi được backend public qua `BACKEND_URL`.
- [ ] Câu trả lời có nội dung và cơ sở pháp lý.
- [ ] Không còn lỗi Qdrant local embedded lock.
- [ ] Không cần máy cá nhân mạnh để chạy demo public.

---

## 19. Quyết định kỹ thuật tạm thời

Để đi nhanh, chọn các quyết định sau:

```text
Qdrant Cloud: lưu vector + full text payload
records.db: không dùng ở runtime
sparse_encoder.json: vẫn giữ ở backend hoặc commit nếu nhỏ
embedding model: vẫn chạy trong backend cloud
LLM: Groq
frontend: Vercel
backend: Render/Railway
index build: Google Colab
```

Sau khi demo ổn, có thể tối ưu tiếp:

```text
- chuyển embedding query sang cloud API
- chuyển sparse_encoder.json sang cloud storage
- thêm streaming thật từ backend
- thêm citation JSON thay vì plain text
- thêm logging và evaluation dashboard
```

---

## 20. Kết luận

Level 3 là hướng phù hợp nếu máy local yếu và muốn demo public ổn định hơn. Ý tưởng cốt lõi là: **Qdrant Cloud không chỉ lưu vector, mà lưu luôn full text/context trong payload**. Khi đó backend cloud có thể search và build prompt trực tiếp từ Qdrant, giảm phụ thuộc vào file SQLite local.

Thứ tự ưu tiên:

```text
1. Payload đầy đủ trong Qdrant Cloud
2. Retriever đọc từ Qdrant payload
3. Backend FastAPI public
4. Frontend Vercel nối BACKEND_URL
5. Build index bằng Colab thay vì máy local
```
