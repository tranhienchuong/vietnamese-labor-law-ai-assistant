# Task cho AI Agent: Sửa lỗi Qdrant ReadTimeout khi chạy benchmark RAG

## Bối cảnh

Project: `vietnamese-labor-law-ai-assistant`

Khi chạy:

```bash
python scripts/run_benchmark.py
```

hệ thống bị lỗi:

```text
httpx.ReadTimeout: The read operation timed out
qdrant_client.http.exceptions.ResponseHandlingException: The read operation timed out
```

Traceback cho thấy lỗi xảy ra ở luồng:

```text
scripts/run_benchmark.py
→ retriever.retrieve()
→ context_assembler.assemble_contexts()
→ context_assembler.add_article_sibling_contexts()
→ record_store.fetch_article_siblings()
→ record_store.fetch_records_by_reference()
→ self.qdrant.scroll()
```

Điều này cho thấy timeout hiện tại không nằm ở embedding model, mà nằm ở bước gọi Qdrant `scroll` để lấy thêm sibling contexts sau khi retrieval chính đã hoàn thành.

---

## Mục tiêu

Hãy sửa code để bước lấy sibling contexts từ Qdrant:

1. Không bị timeout khi benchmark chạy nhiều câu hỏi.
2. Không gọi Qdrant `scroll` lặp lại quá nhiều lần theo kiểu N+1.
3. Không trả vector về nếu chỉ cần payload/context text.
4. Có timeout/retry hợp lý cho Qdrant.
5. Có cache hoặc batching để giảm số lần gọi Qdrant.
6. Có script tạo payload index cho các field dùng để filter.
7. Có log thời gian cho các bước Qdrant chậm để dễ debug sau này.

---

## Các file cần kiểm tra và sửa

Ưu tiên kiểm tra các file sau:

```text
src/vn_labor_law_ai_assistant/rag/retrieval/context_assembler.py
src/vn_labor_law_ai_assistant/rag/retrieval/record_store.py
src/vn_labor_law_ai_assistant/rag/retrieval/retriever.py
scripts/run_benchmark.py
```

Ngoài ra hãy tìm nơi khởi tạo `QdrantClient`, ví dụ:

```text
QdrantClient(...)
```

Có thể nằm trong config, factory, vector store hoặc retriever setup.

---

## Yêu cầu kỹ thuật chi tiết

### 1. Tăng timeout cho QdrantClient

Tìm nơi khởi tạo `QdrantClient` và thêm timeout có thể cấu hình bằng env var.

Ví dụ:

```python
import os
from qdrant_client import QdrantClient

QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "120"))

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=QDRANT_TIMEOUT,
)
```

Nếu project đang dùng `host`, `port`, `path`, `prefer_grpc` hoặc local Qdrant thì giữ nguyên cấu hình cũ, chỉ bổ sung `timeout`.

---

### 2. Đảm bảo `scroll` không trả vector

Trong mọi chỗ gọi:

```python
self.qdrant.scroll(...)
```

nếu chỉ cần metadata/text/payload thì phải set:

```python
with_payload=True
with_vectors=False
```

Ví dụ:

```python
points, next_page = self.qdrant.scroll(
    collection_name=self.collection_name,
    scroll_filter=scroll_filter,
    limit=limit,
    with_payload=True,
    with_vectors=False,
)
```

Không được lấy vector về khi chỉ dùng để assemble context.

---

### 3. Giảm N+1 query khi fetch sibling contexts

Hiện tại có khả năng code đang làm kiểu:

```text
for hit in hits:
    fetch_article_siblings(hit)
```

Nếu top-k có 10 hits, benchmark có 100 questions, số lần gọi Qdrant có thể tăng rất nhanh.

Hãy refactor để:

1. Gom các hit theo khóa sibling, ví dụ:
   - `article_id`
   - hoặc `document_id + article_number`
   - hoặc `law_id + article_number`
   - hoặc field tương đương trong payload hiện tại.

2. Dedupe key trước khi fetch.

3. Cache kết quả sibling theo key trong một lần retrieve.

Ví dụ logic:

```python
sibling_cache = {}

for hit in hits:
    key = make_article_key(hit.payload)

    if key not in sibling_cache:
        sibling_cache[key] = record_store.fetch_article_siblings_by_key(key)

    siblings = sibling_cache[key]
```

Nếu có 10 hits nhưng chỉ thuộc 2 điều luật thì chỉ được gọi Qdrant khoảng 2 lần, không phải 10 lần.

---

### 4. Thêm retry/backoff cho Qdrant scroll

Tạo helper trong `record_store.py`, ví dụ:

```python
import time
import logging
from qdrant_client.http.exceptions import ResponseHandlingException

logger = logging.getLogger(__name__)

def _scroll_with_retry(self, *, max_retries: int = 3, base_sleep: float = 1.0, **kwargs):
    last_exc = None

    for attempt in range(max_retries):
        try:
            return self.qdrant.scroll(**kwargs)
        except ResponseHandlingException as exc:
            last_exc = exc
            sleep_s = base_sleep * (2 ** attempt)
            logger.warning(
                "Qdrant scroll failed, attempt=%s/%s, sleep=%.1fs, error=%s",
                attempt + 1,
                max_retries,
                sleep_s,
                exc,
            )
            time.sleep(sleep_s)

    raise last_exc
```

Sau đó thay các lệnh `self.qdrant.scroll(...)` bằng `_scroll_with_retry(...)`.

Không swallow exception âm thầm. Nếu retry hết vẫn lỗi thì raise lỗi để benchmark biết.

---

### 5. Thêm giới hạn sibling context

Không nên lấy toàn bộ điều luật nếu điều quá dài.

Thêm config/env var, ví dụ:

```python
SIBLING_CONTEXT_LIMIT = int(os.getenv("SIBLING_CONTEXT_LIMIT", "8"))
ENABLE_ARTICLE_SIBLING_CONTEXTS = os.getenv("ENABLE_ARTICLE_SIBLING_CONTEXTS", "true").lower() == "true"
```

Khi benchmark/debug, có thể tắt sibling contexts bằng:

```bash
set ENABLE_ARTICLE_SIBLING_CONTEXTS=false
```

hoặc trên Linux/macOS:

```bash
export ENABLE_ARTICLE_SIBLING_CONTEXTS=false
```

Nếu project đã có config system riêng thì dùng theo style hiện tại của project.

---

### 6. Tạo payload index cho Qdrant

Tạo script mới:

```text
scripts/create_qdrant_payload_indexes.py
```

Script này cần tạo index cho các field đang dùng trong filter sibling.

Các field ứng viên cần kiểm tra trong payload thật:

```text
article_id
article_number
document_id
law_id
reference
chunk_type
metadata.article_id
metadata.article_number
metadata.document_id
metadata.law_id
metadata.reference
metadata.chunk_type
```

Chỉ tạo index cho field thực sự tồn tại / thực sự dùng trong filter.

Ví dụ:

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models

client.create_payload_index(
    collection_name=collection_name,
    field_name="metadata.article_id",
    field_schema=models.PayloadSchemaType.KEYWORD,
    wait=True,
)
```

Script phải idempotent: nếu index đã tồn tại thì không làm benchmark fail.

---

### 7. Thêm logging timing cho Qdrant scroll

Ở `record_store.py`, log thời gian mỗi lần scroll:

```python
import time
import logging

logger = logging.getLogger(__name__)

start = time.perf_counter()
points, next_page = ...
elapsed = time.perf_counter() - start

logger.info(
    "Qdrant scroll done: collection=%s, points=%s, elapsed=%.3fs",
    self.collection_name,
    len(points),
    elapsed,
)
```

Nếu `elapsed > 5s`, log warning.

---

### 8. Không sửa lung tung phần embedding

Lỗi hiện tại không nằm ở Hugging Face embedding API. Không đổi embedding model, không đổi chunking, không đổi vector dimension, trừ khi có lý do rõ ràng.

---




## Checklist nghiệm thu sau khi AI Agent sửa

Chạy lần lượt:

```bash
python scripts/create_qdrant_payload_indexes.py
```

Sau đó chạy smoke test nếu project có:

```bash
python scripts/run_benchmark.py
```

Hoặc chạy benchmark ít câu trước nếu có option limit:

```bash
python scripts/run_benchmark.py --limit 10
```

Quan sát log:

```text
Qdrant scroll done
elapsed=...
```

Cần kiểm tra:

- Không còn `httpx.ReadTimeout`.
- Không còn `ResponseHandlingException: The read operation timed out`.
- Số lần gọi `scroll` không tăng tuyến tính theo `top_k * số câu hỏi`.
- Thời gian benchmark giảm hoặc ít nhất ổn định hơn.
- Nếu tắt sibling contexts thì retrieval vẫn chạy được.

---

## Gợi ý debug nếu vẫn timeout

Nếu vẫn timeout sau các thay đổi trên:

1. Kiểm tra Qdrant đang chạy local, Docker hay cloud.
2. Kiểm tra collection có bao nhiêu points.
3. Kiểm tra filter đang dùng đúng field đã được index chưa.
4. Kiểm tra payload field nằm ở root hay trong `metadata.*`.
5. Giảm `SIBLING_CONTEXT_LIMIT`.
6. Tắt sibling contexts để xác nhận bottleneck.
7. Chuyển `scroll` sang query theo point IDs nếu đã có danh sách IDs cụ thể.
8. Nếu Qdrant chạy cloud/free tier, cân nhắc nâng tier hoặc chạy local trong Docker khi benchmark.
