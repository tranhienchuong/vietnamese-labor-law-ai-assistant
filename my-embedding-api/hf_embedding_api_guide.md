# Hướng dẫn tạo Free Embedding API trên Hugging Face Spaces

## Mục tiêu

Tạo một API embedding miễn phí hoặc chi phí thấp bằng **Hugging Face Spaces + FastAPI + sentence-transformers**.

API này sẽ nhận text đầu vào và trả về vector embedding. Sau đó backend RAG của bạn có thể gọi API này thay vì tự load model embedding trên Render. Cách này giúp giảm RAM cho backend chính.

---

## Kiến trúc tổng quan

```text
Frontend Vercel
        ↓
Backend FastAPI chính trên Render
        ↓
Gọi Embedding API riêng trên Hugging Face Spaces
        ↓
Nhận vector embedding
        ↓
Query Qdrant Cloud
        ↓
Lấy context pháp lý
        ↓
Gọi Groq LLM
        ↓
Trả câu trả lời về frontend
```

Embedding API trên Hugging Face Spaces chỉ có nhiệm vụ:

```text
Text → Vector embedding
```

Nó không trả lời câu hỏi, không gọi Groq, không query Qdrant. Nó chỉ biến văn bản thành vector.

---

## Bước 1: Chuẩn bị mã nguồn trên máy tính

Tạo một thư mục mới trên máy tính, ví dụ:

```text
my-embedding-api/
```

Bên trong thư mục này, tạo 3 file bắt buộc:

```text
my-embedding-api/
├── requirements.txt
├── main.py
└── Dockerfile
```

Có thể tạo thêm file `README.md` nếu muốn cấu hình metadata cho Hugging Face Spaces.

---

## File 1: `requirements.txt`

File này báo cho Hugging Face Spaces biết cần cài đặt các thư viện Python nào.

```txt
fastapi
uvicorn
sentence-transformers
pydantic
torch
```

### Giải thích

```text
fastapi
```

Dùng để tạo API.

```text
uvicorn
```

Dùng để chạy FastAPI server.

```text
sentence-transformers
```

Dùng để load model embedding.

```text
pydantic
```

Dùng để định nghĩa cấu trúc dữ liệu request và response.

```text
torch
```

Là backend cần thiết cho sentence-transformers.

---

## File 2: `main.py`

Đây là file chính của API.

API này sẽ có:

```text
GET /
```

Dùng để kiểm tra server còn sống không.

```text
POST /v1/embeddings
```

Dùng để tạo embedding từ text.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
import torch

# 1. Khởi tạo FastAPI app
app = FastAPI(title="Free Embedding API")

# 2. Load model vào RAM khi server khởi động
print("Đang load model embedding...")

try:
    # Ưu tiên dùng CUDA nếu có.
    # HF Spaces free thường dùng CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model đa ngôn ngữ, hỗ trợ tiếng Việt, tương đối nhẹ.
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name, device=device)

    print(f"Đã load model thành công trên thiết bị: {device}")

except Exception as e:
    print(f"Lỗi load model: {e}")
    model = None


# 3. Định nghĩa cấu trúc dữ liệu đầu vào
class EmbedRequest(BaseModel):
    input: Union[str, List[str]]


# 4. Định nghĩa cấu trúc dữ liệu đầu ra
class EmbedResponse(BaseModel):
    embedding: Union[List[float], List[List[float]]]
    model: str


# 5. Endpoint kiểm tra health check
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "API Embedding đang chạy trên Hugging Face Spaces!"
    }


# 6. Endpoint chính để tạo embedding
@app.post("/v1/embeddings", response_model=EmbedResponse)
async def get_embeddings(request: EmbedRequest):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model chưa được load thành công."
        )

    try:
        sentences = request.input

        if isinstance(sentences, str):
            sentences = [sentences]

        if not sentences:
            raise HTTPException(
                status_code=400,
                detail="Input không được để trống."
            )

        embeddings = model.encode(
            sentences,
            normalize_embeddings=True
        )

        embeddings_list = embeddings.tolist()

        if isinstance(request.input, str):
            embeddings_list = embeddings_list[0]

        return EmbedResponse(
            embedding=embeddings_list,
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi server nội bộ: {str(e)}"
        )
```

---

## Vì sao chọn model này?

Model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Ưu điểm:

```text
- Hỗ trợ nhiều ngôn ngữ, trong đó có tiếng Việt.
- Nhẹ hơn nhiều model embedding lớn.
- Tốc độ khá nhanh.
- Phù hợp cho demo RAG.
- Ít tốn RAM hơn các model lớn như bge-m3.
```

Nhược điểm:

```text
- Chất lượng có thể kém hơn các model retrieval chuyên biệt.
- Nếu cần chất lượng cao hơn, có thể thử BAAI/bge-m3, nhưng model đó nặng hơn.
```

---

## File 3: `Dockerfile`

Hugging Face Spaces dạng Docker sẽ đọc file này để build container.

```dockerfile
# Dùng image Python chính thức, bản nhẹ
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào image
COPY . .

# Expose port mặc định của Hugging Face Spaces
EXPOSE 7860

# Chạy FastAPI bằng uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## File khuyến nghị thêm: `README.md`

Nên tạo thêm file `README.md` để Hugging Face Spaces biết đây là Docker Space và app chạy ở port 7860.

```md
---
title: Free Embedding API
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Free Embedding API

FastAPI service for generating sentence embeddings using sentence-transformers.
```

Cấu trúc thư mục sau khi thêm `README.md`:

```text
my-embedding-api/
├── requirements.txt
├── main.py
├── Dockerfile
└── README.md
```

---

## Bước 2: Test local trước khi deploy

Mở terminal trong thư mục `my-embedding-api`.

Cài thư viện:

```bash
pip install -r requirements.txt
```

Chạy API local:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

Mở trình duyệt:

```text
http://localhost:7860/
```

Nếu thấy:

```json
{
  "status": "ok",
  "message": "API Embedding đang chạy trên Hugging Face Spaces!"
}
```

là API đã chạy.

---

## Bước 3: Test endpoint embedding local

Dùng PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:7860/v1/embeddings" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "input": "Công ty đơn phương chấm dứt hợp đồng lao động trái pháp luật thì phải bồi thường gì?"
  }'
```

Kết quả mong đợi:

```json
{
  "embedding": [0.0123, -0.0456, ...],
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
```

Nếu truyền list nhiều câu:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:7860/v1/embeddings" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "input": [
      "Người lao động nghỉ việc phải báo trước bao lâu?",
      "Trợ cấp thôi việc được tính như thế nào?"
    ]
  }'
```

Kết quả sẽ trả về list vector:

```json
{
  "embedding": [
    [0.0123, -0.0456, ...],
    [0.0789, 0.0111, ...]
  ],
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
```

---

## Bước 4: Tạo Hugging Face Space

Vào Hugging Face:

```text
https://huggingface.co/spaces
```

Tạo Space mới:

```text
New Space
```

Cấu hình:

```text
Space name: my-embedding-api
License: tùy chọn
SDK: Docker
Hardware: CPU basic/free
Visibility: Public hoặc Private
```

Sau đó tạo Space.

---

## Bước 5: Upload code lên Hugging Face Space

Có 2 cách.

### Cách 1: Upload trực tiếp trên web

Vào Space vừa tạo, chọn tab **Files** rồi upload các file:

```text
requirements.txt
main.py
Dockerfile
README.md
```

Sau đó Hugging Face sẽ tự build lại Space.

### Cách 2: Push bằng Git

Clone Space về máy:

```bash
git clone https://huggingface.co/spaces/your-username/my-embedding-api
cd my-embedding-api
```

Copy 4 file vào thư mục đó:

```text
requirements.txt
main.py
Dockerfile
README.md
```

Commit và push:

```bash
git add .
git commit -m "Add FastAPI embedding service"
git push
```

---

## Bước 6: Chờ Hugging Face build

Sau khi push hoặc upload file, Hugging Face sẽ build Docker image.

Vào tab:

```text
Logs
```

Nếu thành công, bạn sẽ thấy server chạy trên port 7860.

Khi Space chạy xong, API sẽ có URL dạng:

```text
https://your-username-my-embedding-api.hf.space
```

Endpoint embedding sẽ là:

```text
https://your-username-my-embedding-api.hf.space/v1/embeddings
```

---

## Bước 7: Test API trên Hugging Face

Dùng PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "https://your-username-my-embedding-api.hf.space/v1/embeddings" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "input": "Công ty đơn phương chấm dứt hợp đồng lao động trái pháp luật thì phải bồi thường gì?"
  }'
```

Nếu API trả về vector, Space đã hoạt động.

---

## Bước 8: Tích hợp với backend RAG chính

Sau khi API embedding chạy được, backend chính của bạn cần gọi URL:

```text
https://your-username-my-embedding-api.hf.space/v1/embeddings
```

Nên thêm env vào backend chính:

```env
EMBEDDING_PROVIDER=custom_http
EMBEDDING_API_URL=https://your-username-my-embedding-api.hf.space/v1/embeddings
DENSE_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Ý tưởng hoạt động:

1. Backend chính nhận câu hỏi.
2. Backend chính gửi câu hỏi sang `EMBEDDING_API_URL`.
3. Hugging Face Space trả về vector.
4. Backend chính dùng vector query Qdrant Cloud.
5. Backend chính lấy context.
6. Backend chính gọi Groq để trả lời.

---

## Bước 9: Rebuild lại Qdrant index

Bắt buộc phải rebuild index bằng cùng model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Vì nếu Qdrant collection được build bằng model cũ nhưng runtime dùng model mới, search sẽ sai.

Quy trình đúng:

```text
1. Build lại index bằng model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
2. Upload lại vector lên Qdrant Cloud.
3. Đảm bảo Qdrant payload vẫn chứa full text.
4. Commit lại metadata index tối thiểu.
5. Set backend env trỏ đến embedding API mới.
6. Redeploy backend.
```

---

## Bước 10: Những lỗi thường gặp

### 1. Space build rất lâu

Nguyên nhân:

```text
sentence-transformers và torch khá nặng.
```

Cách xử lý:

```text
- Chờ thêm.
- Kiểm tra Logs.
- Dùng model nhẹ hơn.
```

### 2. Space bị sleep

Hugging Face Spaces free có thể sleep nếu không dùng một thời gian.

Khi backend gọi lần đầu, API có thể chậm vì phải wake up và load model.

### 3. Lần đầu gọi rất lâu

Lần đầu gọi API có thể mất thời gian vì model cần được load vào RAM.

Sau khi model đã load, các lần sau sẽ nhanh hơn.

### 4. Backend chính vẫn lỗi RAM

Nếu backend chính vẫn load sentence-transformers, nghĩa là bạn chưa sửa backend chính để gọi embedding API ngoài.

Backend chính phải không tự load model embedding nữa.

### 5. Vector search sai

Nguyên nhân thường gặp:

```text
Build index dùng model A nhưng runtime dùng model B.
```

Cách xử lý:

```text
Rebuild index bằng đúng model mà runtime đang dùng.
```

---

## Bước 11: Lưu ý về production

Cách này phù hợp cho:

```text
- Demo sinh viên
- MVP nhỏ
- Test RAG
- Tránh lỗi RAM trên Render
```

Nhưng chưa phải production mạnh vì:

```text
- Hugging Face Spaces free có thể sleep.
- Lần đầu gọi có thể chậm.
- Rate limit có thể xảy ra.
- Không đảm bảo uptime tuyệt đối.
```

Nếu cần production ổn định hơn, cân nhắc:

```text
- Render/Railway gói RAM cao hơn
- Cloud Run
- Modal
- Replicate
- OpenAI/Gemini/Cohere embedding API
```

---

## Tóm tắt

Bạn tạo một Hugging Face Space riêng chỉ để làm embedding API.

Nó nhận:

```json
{
  "input": "text cần embedding"
}
```

và trả:

```json
{
  "embedding": [0.0123, -0.0456, "..."],
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
```

Sau đó backend chính gọi API này thay vì tự load model embedding. Cách này giúp giảm RAM trên Render và có thể giúp project deploy được với hạ tầng yếu hơn.
