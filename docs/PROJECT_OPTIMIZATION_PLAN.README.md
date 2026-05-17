# Project Optimization Plan

> Branch đề xuất: `codex/user-admin-auth-phase1`  
> Project: Vietnamese Labor Law AI Assistant  
> Goal: tối ưu lại cấu trúc code theo hướng dễ mở rộng, dễ debug RAG, an toàn hơn cho auth/admin, và dễ triển khai production.

---

## 1. Bối cảnh hiện tại

Dự án hiện đã có đầy đủ các khối chính:

- Backend FastAPI cho chat, auth, conversation và admin API.
- Frontend Next.js cho landing page, chat UI, login, admin dashboard và settings page.
- RAG pipeline gồm corpus builder, chunking theo metadata pháp luật, dense/sparse indexing, Qdrant hybrid retrieval, reranking và LLM answer generation.
- Auth phase 1 gồm user/admin, session token, HTTP-only cookie phía frontend và SQLite app database.
- Evaluation/benchmark cho retrieval và answer quality.

Tuy nhiên, code đang bắt đầu phình ở một số file lớn:

- `src/vn_labor_law_ai_assistant/api.py`: đang gom app setup, auth dependencies, login/logout, conversation API, admin API và chat endpoint.
- `src/vn_labor_law_ai_assistant/auth_store.py`: đang gom security, token, user repository, session repository, conversation repository và message repository.
- `src/vn_labor_law_ai_assistant/retriever.py`: đang gom query routing, Qdrant search, record fetching, forced references, fallback hits, scoring, reranking và context assembly.
- `src/vn_labor_law_ai_assistant/answering.py`: đang gom prompt, schema, parser, citation guard và một số answer override hard-code.
- Frontend có nhiều API proxy route lặp logic auth headers, token check và backend forwarding.

---

## 2. Mục tiêu tối ưu

### 2.1. Mục tiêu kiến trúc

- Giữ mô hình **modular monolith**, chưa tách microservice.
- Tách rõ các layer:
  - API layer
  - Service layer
  - Repository/database layer
  - RAG/retrieval layer
  - LLM provider layer
  - Frontend feature layer
- Giữ behavior hiện tại để tránh làm vỡ chatbot.
- Refactor theo từng PR nhỏ, có test bao quanh.

### 2.2. Mục tiêu chất lượng RAG

- Dễ debug câu trả lời sai: biết lỗi do router, retriever, reranker, context assembly hay LLM.
- Log được retrieval trace cho mỗi lượt chat.
- Có benchmark tự động để so sánh trước/sau refactor.
- Giảm hard-code rule trong Python, ưu tiên YAML/config.

### 2.3. Mục tiêu production

- Không dùng default secret/password ở môi trường production.
- Auth rõ trách nhiệm, dễ đổi SQLite sang Postgres sau này.
- Admin dashboard dùng dữ liệu thật thay vì mock constants.
- CI kiểm tra backend/frontend trước khi merge.

---

## 3. Cấu trúc thư mục mục tiêu

Đề xuất cấu trúc backend:

```text
src/vn_labor_law_ai_assistant/
  api/
    app.py
    deps.py
    routes/
      health.py
      auth.py
      chat.py
      conversations.py
      admin.py

  core/
    config.py
    security.py
    errors.py
    logging.py

  db/
    sqlite.py
    migrations.py

  auth/
    models.py
    schemas.py
    repository.py
    service.py

  conversations/
    schemas.py
    repository.py
    service.py

  rag/
    service.py
    retrieval/
      retriever.py
      query_encoder.py
      qdrant_search.py
      record_store.py
      reference_expander.py
      context_assembler.py
      scoring.py
      semantic_reranker.py
    answering/
      prompt.py
      schema.py
      parser.py
      citation_guard.py
      overrides.py

  llm/
    base.py
    groq_provider.py
    azure_provider.py
    factory.py

  corpus/
    pipeline.py
    chunking.py
    metadata.py
```

Đề xuất cấu trúc frontend:

```text
frontend/src/
  app/
    api/
    admin/
    chat/
    login/

  features/
    auth/
      auth-provider.tsx
      login-form.tsx
      use-auth.ts
    chat/
      chat-interface.tsx
      chat-input.tsx
      message-list.tsx
      use-chat-session.ts
    admin/
      dashboard.tsx
      settings-form.tsx

  lib/
    api/
      client.ts
      auth.ts
      chat.ts
      conversations.ts
      admin.ts
    constants.ts
    types.ts
    utils.ts
```

---

## 4. Roadmap theo PR

## PR 1 — Backend API Modularization

### Mục tiêu

Tách `api.py` thành nhiều router nhỏ mà không đổi logic nghiệp vụ.

### Việc cần làm

- [ ] Tạo `api/app.py` để khởi tạo FastAPI app, CORS và include routers.
- [ ] Tạo `api/deps.py` cho dependency:
  - `get_auth_store()`
  - `get_retriever()`
  - `require_current_user()`
  - `require_admin_user()`
- [ ] Tạo route modules:
  - `api/routes/health.py`
  - `api/routes/auth.py`
  - `api/routes/conversations.py`
  - `api/routes/chat.py`
  - `api/routes/admin.py`
- [ ] Giữ backward-compatible import nếu cần:
  - `vn_labor_law_ai_assistant.api:app` vẫn chạy được.
- [ ] Thêm Pydantic schemas:
  - `LoginRequest`
  - `ChatRequest`
  - `ConversationCreateRequest`
  - `AuthUserResponse`
  - `ConversationResponse`

### Tiêu chí hoàn thành

- [ ] Backend vẫn chạy bằng lệnh cũ:

```powershell
.venv\Scripts\python.exe -m uvicorn vn_labor_law_ai_assistant.api:app --reload --host 0.0.0.0 --port 8000
```

- [ ] Các endpoint cũ vẫn hoạt động:
  - `GET /`
  - `GET /health`
  - `POST /auth/login`
  - `POST /auth/logout`
  - `GET /auth/me`
  - `GET /conversations`
  - `POST /conversations`
  - `GET /conversations/{conversation_id}`
  - `POST /chat`
  - `GET /admin/stats`
- [ ] Unit tests hiện tại pass.

### Rủi ro

- Sai import path khiến Uvicorn không tìm thấy `app`.
- Pydantic alias như `conversationId`, `topK`, `maxContexts` bị mismatch với frontend.

---

## PR 2 — Auth Store Refactor

### Mục tiêu

Tách `auth_store.py` thành security, repository và service để dễ bảo trì.

### Việc cần làm

- [ ] Tạo `core/security.py`:
  - `hash_password()`
  - `verify_password()`
  - `create_access_token()`
  - `decode_and_verify_token()`
  - `token_hash()`
- [ ] Tạo `db/sqlite.py`:
  - connection factory
  - row factory
  - foreign key pragma
- [ ] Tạo `auth/repository.py`:
  - `UserRepository`
  - `SessionRepository`
- [ ] Tạo `conversations/repository.py`:
  - `ConversationRepository`
  - `MessageRepository`
- [ ] Tạo `auth/service.py`:
  - `authenticate_user()`
  - `create_session()`
  - `revoke_session()`
  - `get_user_by_token()`
- [ ] Tạo `conversations/service.py`:
  - `list_conversations()`
  - `get_conversation()`
  - `create_conversation()`
  - `ensure_conversation_for_question()`
  - `append_message()`

### Security checklist

- [ ] `AUTH_SECRET` phải bắt buộc ở non-dev environment.
- [ ] Không fallback production secret sang `dev-only-change-me...`.
- [ ] `AUTH_SEED_DEFAULT_USERS` mặc định nên là `0` trong production.
- [ ] Default admin/user password chỉ dùng local dev.
- [ ] Login failure message không leak thông tin user tồn tại hay không.
- [ ] Token/session revoke vẫn hoạt động.

### Tiêu chí hoàn thành

- [ ] Login/logout/me vẫn hoạt động.
- [ ] Chat vẫn tự tạo conversation khi chưa có `conversationId`.
- [ ] User không đọc được conversation của user khác.
- [ ] Admin check role vẫn hoạt động.
- [ ] Test auth/conversation được bổ sung.

---

## PR 3 — Settings và Config Cleanup

### Mục tiêu

Thay config thủ công bằng typed settings.

### Việc cần làm

- [ ] Thêm dependency `pydantic-settings`.
- [ ] Tạo `core/config.py` với `Settings`.
- [ ] Gom env vars:
  - LLM provider/model
  - Qdrant config
  - Retriever config
  - Auth config
  - App DB config
  - CORS config
- [ ] Thay `os.getenv(...)` rải rác bằng `settings` object.
- [ ] Giữ `load_repo_env()` tạm thời nếu cần backward compatibility, nhưng không để nó là cơ chế chính lâu dài.

### Tiêu chí hoàn thành

- [ ] App fail sớm nếu thiếu config bắt buộc.
- [ ] `.env.example` được cập nhật rõ dev/prod.
- [ ] Tests không phụ thuộc vào `.env` thật.

---

## PR 4 — RAG Retriever Decomposition

### Mục tiêu

Tách `HybridRetriever` thành các component nhỏ nhưng giữ public API `retrieve()`.

### Việc cần làm

- [ ] Tạo `rag/retrieval/query_encoder.py`:
  - dense query encoding
  - sparse query encoding
- [ ] Tạo `rag/retrieval/qdrant_search.py`:
  - build filters
  - query Qdrant
  - fusion/RRF search
- [ ] Tạo `rag/retrieval/record_store.py`:
  - SQLite record source
  - Qdrant payload record source
  - fetch by chunk ids
  - fetch by legal reference
- [ ] Tạo `rag/retrieval/reference_expander.py`:
  - forced reference records
  - reference fallback hits
  - article/clause/point fallback
- [ ] Tạo `rag/retrieval/context_assembler.py`:
  - parent chunk expansion
  - sibling context expansion
  - small-to-big context assembly
- [ ] Tạo `rag/retrieval/scoring.py`:
  - boost flags
  - boost rule matching
  - heuristic score adjustment
- [ ] Tạo `rag/retrieval/semantic_reranker.py`:
  - cross-encoder reranking
  - RRF fusion with heuristic rank

### Tiêu chí hoàn thành

- [ ] `HybridRetriever.retrieve()` vẫn trả `RetrievalResult` như cũ.
- [ ] Existing tests pass.
- [ ] Thêm unit tests cho từng component mới.
- [ ] Benchmark retrieval không giảm so với trước.

### Gợi ý test mới

- [ ] Query có explicit `Điều 35` phải pin đúng article.
- [ ] Query có `khoản 1 điểm a` phải lấy đúng clause/point.
- [ ] Query hỏi liệt kê phải expand sibling context.
- [ ] Query tính trợ cấp thôi việc phải ưu tiên Điều 46.
- [ ] Query về nghị định phải ưu tiên document implementation nếu có hint.

---

## PR 5 — Answering Module Cleanup

### Mục tiêu

Tách `answering.py` thành prompt/schema/parser/citation guard/override.

### Việc cần làm

- [ ] Tạo `rag/answering/prompt.py`:
  - `SYSTEM_PROMPT`
  - few-shot examples
  - prompt builder
- [ ] Tạo `rag/answering/schema.py`:
  - JSON schema
  - Pydantic model nếu cần
- [ ] Tạo `rag/answering/parser.py`:
  - extract JSON
  - parse answer payload
- [ ] Tạo `rag/answering/citation_guard.py`:
  - allowed citation builder
  - citation canonicalization
  - evidence quote validation
- [ ] Tạo `rag/answering/overrides.py`:
  - contextual overrides hiện tại
- [ ] Chuyển override hard-code sang YAML ở bước sau.

### Tiêu chí hoàn thành

- [ ] Output answer format không đổi.
- [ ] Citation guard vẫn loại citation không nằm trong context.
- [ ] Evidence quote phải thật sự xuất hiện trong context.
- [ ] Tests cho parser/citation guard pass.

---

## PR 6 — Rule YAML Consolidation

### Mục tiêu

Giảm hard-code rule trong Python, gom routing/scoring/override rule vào YAML.

### Việc cần làm

- [ ] Mở rộng `rules/routing_config.yaml` hoặc tách thêm:
  - `rules/routing_config.yaml`
  - `rules/scoring_config.yaml`
  - `rules/answer_overrides.yaml`
- [ ] Đưa các override phổ biến vào config:
  - loại hợp đồng lao động
  - báo trước 45 ngày
  - báo trước 30 ngày
  - báo trước 03 ngày làm việc
  - thử việc 60 ngày
  - lương thử việc 85%
  - làm thêm ngày nghỉ hằng tuần 200%
- [ ] Rule loader validate schema khi startup.

### Tiêu chí hoàn thành

- [ ] Thêm/sửa rule không cần sửa Python code.
- [ ] Rule invalid phải báo lỗi dễ hiểu.
- [ ] Benchmark các câu override hiện tại không giảm chất lượng.

---

## PR 7 — Frontend API Client và Auth Guard Cleanup

### Mục tiêu

Giảm fetch logic lặp lại trong frontend, gom theo feature.

### Việc cần làm

- [ ] Tạo `frontend/src/lib/api/client.ts`:
  - wrapper fetch
  - error handling
  - JSON/text response handling
- [ ] Tạo API modules:
  - `lib/api/auth.ts`
  - `lib/api/chat.ts`
  - `lib/api/conversations.ts`
  - `lib/api/admin.ts`
- [ ] Refactor `AuthProvider` dùng `authApi`.
- [ ] Refactor `ChatInterface` dùng `chatApi`/`conversationApi`.
- [ ] Giữ HTTP-only cookie, không expose token ra client JS.
- [ ] Tối ưu middleware auth check:
  - Option A: cache role ngắn hạn
  - Option B: verify JWT role claim cục bộ nếu chuyển token chuẩn JWT

### Tiêu chí hoàn thành

- [ ] Login/logout vẫn hoạt động.
- [ ] `/chat` redirect đúng nếu chưa login.
- [ ] `/admin` chỉ admin vào được.
- [ ] Conversation list/load vẫn hoạt động.
- [ ] Chat vẫn nhận `X-Conversation-Id`.

---

## PR 8 — Admin Dashboard dùng dữ liệu thật

### Mục tiêu

Biến admin dashboard từ mock UI thành monitoring dashboard tối thiểu.

### Backend cần thêm

- [ ] `GET /admin/stats` trả:
  - total users
  - total conversations
  - total messages
  - active sessions
  - index collection name
  - record source
  - reranker enabled
  - query router enabled
- [ ] `GET /admin/health` trả:
  - app status
  - database status
  - qdrant status
  - llm config status, không trả secret
- [ ] `GET /admin/retrieval-config` trả config đang dùng:
  - `QDRANT_COLLECTION`
  - `RETRIEVER_RECORD_SOURCE`
  - `INDEX_PATH`
  - `RERANKER_MODEL`
  - `QUERY_ROUTER_ENABLED`

### Frontend cần sửa

- [ ] `admin/page.tsx` fetch dữ liệu thật.
- [ ] Bỏ `ADMIN_METRICS` mock hoặc chỉ dùng làm fallback.
- [ ] Thêm loading/error state.
- [ ] `admin/settings/page.tsx` không hiển thị nút “Lưu cấu hình” nếu chưa có API update thật.

### Tiêu chí hoàn thành

- [ ] Admin thấy trạng thái thật của hệ thống.
- [ ] Không leak API key, auth secret, token.
- [ ] User thường không gọi được admin API.

---

## PR 9 — Observability và Retrieval Trace

### Mục tiêu

Mỗi câu trả lời sai phải có dữ liệu để debug.

### Việc cần làm

- [ ] Tạo bảng `chat_traces` hoặc lưu metadata vào `messages.metadata_json`.
- [ ] Lưu cho mỗi lượt chat:
  - query
  - user_id
  - conversation_id
  - provider/model
  - latency
  - intent metadata
  - retrieved chunk ids
  - matched citations
  - selected context ids
  - insufficient_context
  - error nếu có
- [ ] Thêm `X-Request-Id` cho API response.
- [ ] Log structured JSON thay vì print rời rạc.
- [ ] Admin xem được trace gần đây.

### Tiêu chí hoàn thành

- [ ] Có thể mở một câu trả lời và biết retriever đã lấy chunk nào.
- [ ] Có thể phân loại lỗi:
  - router sai
  - retrieval sai
  - context thiếu
  - LLM hallucinate
  - citation guard loại hết citation

---

## PR 10 — Benchmark và CI/CD

### Mục tiêu

Tự động phát hiện regression sau mỗi thay đổi.

### Backend CI

- [ ] Thêm dev dependencies:

```toml
[project.optional-dependencies]
dev = [
  "pytest",
  "pytest-cov",
  "httpx",
  "ruff",
  "mypy",
  "types-PyYAML"
]
```

- [ ] GitHub Actions backend:
  - install python
  - install `.[dev]`
  - `ruff check`
  - `mypy src`
  - `pytest`

### Frontend CI

- [ ] GitHub Actions frontend:
  - `npm ci`
  - `npm run lint`
  - `npm run build`

### Benchmark CI hoặc manual workflow

- [ ] Add workflow chạy benchmark thủ công.
- [ ] Upload artifact `eval/results/*.csv`.
- [ ] So sánh metrics trước/sau:
  - retrieval hit rate
  - citation precision
  - answer score
  - insufficient context rate

### Tiêu chí hoàn thành

- [ ] PR không pass test thì không merge.
- [ ] Có số liệu trước/sau cho RAG changes.

---

## 5. Thứ tự ưu tiên khuyến nghị

Làm theo thứ tự sau để ít rủi ro nhất:

1. **PR 1 — Backend API Modularization**  
   Tách route trước, không đổi logic.

2. **PR 2 — Auth Store Refactor**  
   Vì branch hiện tại là auth/admin phase 1, nên ổn định phần này sớm.

3. **PR 3 — Settings và Config Cleanup**  
   Làm trước khi production deploy.

4. **PR 8 — Admin Dashboard dùng dữ liệu thật**  
   Giúp quan sát hệ thống tốt hơn.

5. **PR 9 — Observability và Retrieval Trace**  
   Cần có trace trước khi refactor sâu RAG.

6. **PR 4 — RAG Retriever Decomposition**  
   Refactor phần phức tạp nhất sau khi đã có trace và benchmark.

7. **PR 5 — Answering Module Cleanup**  
   Làm sau retriever hoặc song song nếu có test tốt.

8. **PR 6 — Rule YAML Consolidation**  
   Làm khi các module đã ổn.

9. **PR 7 — Frontend API Client Cleanup**  
   Có thể làm song song với backend nếu không đổi API contract.

10. **PR 10 — Benchmark và CI/CD**  
    Nên làm sớm nếu có thời gian, nhưng có thể tách thành workflow riêng.

---

## 6. Definition of Done tổng thể

Một phase refactor chỉ được coi là xong khi:

- [ ] Không đổi behavior ngoài ý muốn.
- [ ] Endpoint public cũ vẫn hoạt động.
- [ ] Frontend vẫn login/chat/load conversation được.
- [ ] Admin vẫn bị chặn nếu role không phải admin.
- [ ] Test hiện tại pass.
- [ ] Có test mới cho phần vừa tách.
- [ ] Không leak secret/token trong response hoặc log.
- [ ] README hoặc docs được cập nhật nếu command/env thay đổi.

---

## 7. Các quyết định kiến trúc nên giữ

- Giữ FastAPI + Next.js.
- Giữ Qdrant hybrid retrieval.
- Giữ SQLite cho local/dev và phase đầu, nhưng thiết kế repository để đổi Postgres sau.
- Giữ HTTP-only cookie cho frontend auth.
- Giữ route proxy `/api/*` phía Next.js để không expose backend URL/token trực tiếp cho browser.
- Giữ legal chunk metadata: `document_id`, `article_number`, `clause_ref`, `point_refs`, `citation_text`, `topic`, `actor`, `issue_type`.
- Giữ benchmark/evaluation làm tiêu chuẩn trước khi sửa RAG sâu.

---

## 8. Những việc không nên làm ngay

- Không tách microservice lúc này.
- Không rewrite toàn bộ retriever một lần.
- Không đổi cả chunking, retriever, prompt và frontend trong cùng một PR.
- Không thêm OAuth/social login trước khi auth/password/session hiện tại ổn định.
- Không đưa admin settings thành editable nếu backend chưa validate và persist config an toàn.
- Không lưu API key hoặc token vào `NEXT_PUBLIC_*`.

---

## 9. Gợi ý branch strategy

Từ branch hiện tại:

```text
codex/user-admin-auth-phase1
```

Tạo các branch nhỏ:

```text
refactor/backend-api-modularization
refactor/auth-store-service-layer
refactor/settings-pydantic
feature/admin-real-stats
feature/chat-tracing
refactor/rag-retriever-components
refactor/answering-components
refactor/frontend-api-client
ci/backend-frontend-checks
```

Mỗi branch nên merge riêng sau khi test pass.

---

## 10. Kết luận

Hướng tối ưu phù hợp nhất là **modular monolith + service/repository pattern + RAG components rõ ràng**.

Ưu tiên trước mắt không phải viết thêm nhiều tính năng, mà là:

1. Làm code auth/admin/chat dễ đọc hơn.
2. Tăng an toàn config và auth secret.
3. Có trace để debug RAG.
4. Có benchmark/CI để tránh regression.
5. Sau đó mới refactor sâu retriever và answering.
