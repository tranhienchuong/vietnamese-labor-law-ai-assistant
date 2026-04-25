# Frontend Design Specification

**Project:** Vietnamese Labor Law AI Assistant  
**File:** `frontend.md`  
**Goal:** Thiết kế giao diện web hiện đại, chuyên nghiệp, mượt mà cho trợ lí AI hỏi đáp về Luật Lao động Việt Nam, đặc biệt tập trung vào các tình huống chấm dứt hợp đồng lao động.

---

## 1. Mục tiêu giao diện

Frontend cần đạt các mục tiêu sau:

- Tạo trải nghiệm giống các sản phẩm AI hiện đại như ChatGPT, Perplexity, Harvey hoặc các legal AI assistant.
- Hỗ trợ hội thoại thời gian thực với hiệu ứng chữ xuất hiện dần thông qua streaming.
- Hiển thị câu trả lời rõ ràng, có trích dẫn nguồn luật nếu backend cung cấp.
- Phù hợp với môi trường doanh nghiệp: sạch, tối giản, đáng tin cậy, dễ mở rộng.
- Tối ưu cho người dùng không chuyên về pháp luật: câu trả lời dễ đọc, có cảnh báo khi thông tin cần luật sư hoặc nguồn chính thức.
- Có cấu trúc frontend rõ ràng để dễ bảo trì, mở rộng và deploy.

---

## 2. Stack công nghệ khuyên dùng

### 2.1 Core Framework: Next.js

**Công nghệ:** Next.js + React + TypeScript

**Lý do chọn:**

- Hỗ trợ App Router hiện đại.
- Dễ xây dựng API route nếu cần proxy request tới backend.
- Tối ưu SEO, routing, performance và deploy tốt trên Vercel.
- Phù hợp với các ứng dụng dashboard, chatbot, AI assistant.

**Khuyến nghị:**

```bash
npx create-next-app@latest frontend
```

Nên chọn:

```text
TypeScript: Yes
ESLint: Yes
Tailwind CSS: Yes
App Router: Yes
src/ directory: Yes
Import alias: Yes
```

---

### 2.2 AI Tooling: Vercel AI SDK

**Công nghệ:** Vercel AI SDK

**Lý do chọn:**

- Có sẵn hook `useChat`.
- Tự quản lý danh sách tin nhắn.
- Hỗ trợ streaming text rất thuận tiện.
- Giảm đáng kể độ phức tạp khi xây dựng UI chatbot.
- Dễ tích hợp với backend dùng OpenAI, Gemini, Claude hoặc backend RAG riêng.

Cài đặt:

```bash
npm install ai
```

Ví dụ frontend dùng `useChat`:

```tsx
'use client'

import { useChat } from 'ai/react'

export function ChatInterface() {
  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    stop,
    reload,
  } = useChat({
    api: '/api/chat',
  })

  return (
    <form onSubmit={handleSubmit}>
      <div>
        {messages.map((message) => (
          <div key={message.id}>
            <strong>{message.role}</strong>
            <p>{message.content}</p>
          </div>
        ))}
      </div>

      <input
        value={input}
        onChange={handleInputChange}
        placeholder="Nhập câu hỏi về luật lao động..."
      />

      <button type="submit" disabled={isLoading}>
        Gửi
      </button>
    </form>
  )
}
```

---

### 2.3 UI Components: Tailwind CSS + shadcn/ui

**Công nghệ:** Tailwind CSS + shadcn/ui

**Lý do chọn:**

- Không cần viết CSS từ đầu.
- Component hiện đại, dễ tuỳ biến.
- Phù hợp với giao diện enterprise.
- Dễ kết hợp với icon, dark mode, responsive layout.
- Không bị phụ thuộc nặng vào một thư viện UI đóng.

Cài đặt shadcn/ui:

```bash
npx shadcn@latest init
```

Các component nên cài:

```bash
npx shadcn@latest add button input textarea card scroll-area avatar badge separator dialog sheet dropdown-menu tooltip tabs skeleton alert
```

Icon khuyên dùng:

```bash
npm install lucide-react
```

---

## 3. Định hướng sản phẩm

### 3.1 Tên sản phẩm đề xuất

Một số tên có thể dùng cho giao diện:

- **LaborLaw AI**
- **Vietnam Labor Law Assistant**
- **Legal Assistant for Labor Contract Termination**
- **Trợ lí AI Luật Lao động Việt Nam**

Tên hiển thị khuyên dùng:

```text
Vietnam Labor Law AI Assistant
```

Tagline:

```text
Hỏi đáp thông minh về chấm dứt hợp đồng lao động theo Bộ luật Lao động Việt Nam 2019.
```

---

## 4. Kiến trúc giao diện tổng thể

```mermaid
flowchart TD
    User[User] --> UI[Next.js Frontend]
    UI --> ChatAPI[/api/chat]
    ChatAPI --> Backend[AI Backend / RAG API]
    Backend --> Retriever[Legal Document Retriever]
    Backend --> LLM[Large Language Model]
    Backend --> Sources[Law Sources / Database]
    Backend --> ChatAPI
    ChatAPI --> UI
```

Frontend chịu trách nhiệm:

- Hiển thị giao diện chat.
- Gửi câu hỏi người dùng.
- Nhận streaming response.
- Hiển thị câu trả lời, citation, trạng thái loading.
- Quản lý lịch sử hội thoại phía client hoặc qua API.
- Cung cấp giao diện upload tài liệu nếu cần.
- Cung cấp trang đánh giá model nếu dự án cần benchmark.

Backend chịu trách nhiệm:

- Xử lý truy vấn.
- Retrieval tài liệu luật liên quan.
- Gọi mô hình ngôn ngữ lớn.
- Sinh câu trả lời có căn cứ.
- Trả về citation, metadata, confidence score nếu có.

---

## 5. Cấu trúc trang chính

### 5.1 Trang `/`

Trang landing ngắn gọn.

Nội dung nên có:

- Tên hệ thống.
- Mô tả ngắn về trợ lí AI.
- Nút bắt đầu chat.
- Cảnh báo pháp lý ngắn.
- Một vài câu hỏi mẫu.

Ví dụ câu hỏi mẫu:

```text
Khi nào người lao động có quyền đơn phương chấm dứt hợp đồng lao động?
```

```text
Công ty đơn phương chấm dứt hợp đồng trái luật thì phải bồi thường gì?
```

```text
Thời hạn báo trước khi chấm dứt hợp đồng lao động là bao lâu?
```

---

### 5.2 Trang `/chat`

Đây là trang quan trọng nhất.

Layout đề xuất:

```text
+-------------------------------------------------------+
| Header: Logo, project name, theme toggle, settings    |
+----------------------+--------------------------------+
| Sidebar              | Chat Area                      |
| - New chat           | - Message list                 |
| - Chat history       | - Assistant answer             |
| - Example questions  | - Citations                    |
| - Evaluation mode    | - Input box                    |
+----------------------+--------------------------------+
```

Thành phần chính:

- Sidebar lịch sử hội thoại.
- Vùng hiển thị tin nhắn.
- Input box cố định phía dưới.
- Citation panel hoặc citation cards dưới câu trả lời.
- Loading skeleton khi AI đang xử lý.
- Nút dừng generation khi đang streaming.
- Nút copy câu trả lời.
- Nút regenerate câu trả lời.
- Nút đánh giá câu trả lời: đúng, sai, thiếu thông tin, hallucination.

---

### 5.3 Trang `/documents`

Dùng nếu hệ thống có chức năng quản lý nguồn luật hoặc tài liệu nội bộ.

Chức năng:

- Hiển thị danh sách tài liệu đã index.
- Upload file PDF/DOCX nếu backend hỗ trợ.
- Trạng thái index: pending, processing, completed, failed.
- Tìm kiếm tài liệu theo tên.
- Xem metadata: tên file, ngày upload, số chunk, trạng thái.

---

### 5.4 Trang `/evaluation`

Dùng cho dự án đánh giá chất lượng chatbot.

Chức năng:

- Hiển thị bộ câu hỏi benchmark.
- Hiển thị câu trả lời chuẩn.
- Hiển thị câu trả lời của AI.
- Chấm điểm theo các tiêu chí:
  - answer_correct
  - citation_correct
  - hallucination_flag
  - abstention_correct
  - clarity
  - format
  - final_score_10
- Cho phép export kết quả ra CSV hoặc JSON.

---

### 5.5 Trang `/settings`

Cấu hình người dùng hoặc hệ thống.

Chức năng:

- Chọn model.
- Chọn chế độ trả lời: ngắn gọn, chi tiết, có căn cứ luật.
- Bật/tắt citation.
- Bật/tắt streaming.
- Bật/tắt dark mode.
- Cấu hình language: Vietnamese / English.

---

## 6. Thiết kế giao diện Chatbot

### 6.1 Header

Nội dung header:

- Logo.
- Tên app.
- Badge trạng thái: `Beta`, `RAG enabled`, hoặc `Demo`.
- Nút chuyển dark mode.
- Nút settings.
- Nút GitHub hoặc Docs nếu cần.

Ví dụ:

```text
Vietnam Labor Law AI Assistant     [Beta] [Dark Mode] [Settings]
```

---

### 6.2 Sidebar

Sidebar giúp người dùng điều hướng nhanh.

Nội dung:

- Nút `New Chat`.
- Danh sách hội thoại gần đây.
- Nhóm câu hỏi mẫu.
- Link sang Evaluation.
- Link sang Documents.
- Footer nhỏ hiển thị version.

Component shadcn/ui nên dùng:

- `Button`
- `ScrollArea`
- `Separator`
- `Badge`

---

### 6.3 Message List

Mỗi message gồm:

- Avatar user hoặc assistant.
- Nội dung tin nhắn.
- Thời gian nếu cần.
- Trạng thái streaming.
- Button copy.
- Button regenerate cho assistant message.

User message:

- Căn phải hoặc dùng nền hơi nổi bật.
- Nội dung ngắn gọn.

Assistant message:

- Căn trái.
- Render Markdown.
- Hỗ trợ list, table, bold, code block.
- Có vùng citation phía dưới.

Thư viện render Markdown đề xuất:

```bash
npm install react-markdown remark-gfm
```

---

### 6.4 Input Box

Input nên đặt cố định phía dưới.

Yêu cầu:

- Dùng `Textarea`, không chỉ `Input`, vì câu hỏi pháp luật thường dài.
- Enter để gửi.
- Shift + Enter để xuống dòng.
- Có nút send.
- Có nút stop khi đang streaming.
- Có placeholder rõ ràng.

Placeholder đề xuất:

```text
Nhập câu hỏi về chấm dứt hợp đồng lao động...
```

Ví dụ câu hỏi:

```text
Người lao động ký hợp đồng không xác định thời hạn muốn nghỉ việc thì phải báo trước bao lâu?
```

---

### 6.5 Citation Cards

Nếu backend trả về nguồn, frontend nên hiển thị dạng card.

Mỗi citation gồm:

- Tên văn bản luật.
- Điều, khoản nếu có.
- Đoạn trích liên quan.
- Link nguồn nếu có.
- Mức độ liên quan nếu backend cung cấp.

Ví dụ hiển thị:

```text
Nguồn tham khảo
1. Bộ luật Lao động 2019 - Điều 35
   Người lao động có quyền đơn phương chấm dứt hợp đồng lao động...
```

Component nên dùng:

- `Card`
- `Badge`
- `Accordion`
- `Tooltip`

---

## 7. Visual Design

### 7.1 Phong cách

Phong cách nên hướng tới:

- Clean.
- Enterprise.
- Minimal.
- Trustworthy.
- Legal-tech.
- Ít màu mè, ưu tiên dễ đọc.

Không nên:

- Dùng quá nhiều animation.
- Dùng màu quá chói.
- Giao diện giống app giải trí.
- Nhồi quá nhiều thông tin trong một màn hình.

---

### 7.2 Màu sắc đề xuất

Theme sáng:

```text
Background: #F8FAFC
Surface: #FFFFFF
Primary: #2563EB
Text: #0F172A
Muted Text: #64748B
Border: #E2E8F0
Success: #16A34A
Warning: #F59E0B
Error: #DC2626
```

Theme tối:

```text
Background: #020617
Surface: #0F172A
Primary: #60A5FA
Text: #F8FAFC
Muted Text: #94A3B8
Border: #1E293B
```

---

### 7.3 Typography

Khuyến nghị:

- Font: Inter hoặc Geist.
- Heading rõ ràng.
- Body text dễ đọc.
- Cỡ chữ message: `15px` hoặc `16px`.
- Line height: `1.6`.

Cài font Geist nếu dùng Next.js:

```tsx
import { Geist } from 'next/font/google'

const geist = Geist({
  subsets: ['latin'],
})
```

---

## 8. Component Design

### 8.1 Component tree đề xuất

```text
AppLayout
├── AppHeader
├── AppSidebar
│   ├── NewChatButton
│   ├── ChatHistoryList
│   └── ExampleQuestionList
└── ChatPage
    ├── MessageList
    │   ├── UserMessage
    │   └── AssistantMessage
    │       ├── MarkdownRenderer
    │       ├── CitationList
    │       └── FeedbackButtons
    └── ChatInput
```

---

### 8.2 Các component chính

#### `ChatInterface`

Quản lý logic hội thoại chính.

Trách nhiệm:

- Gọi `useChat`.
- Truyền messages xuống `MessageList`.
- Truyền input handlers xuống `ChatInput`.
- Quản lý trạng thái loading.

---

#### `MessageList`

Trách nhiệm:

- Hiển thị toàn bộ message.
- Auto-scroll xuống cuối khi có message mới.
- Hiển thị empty state nếu chưa có hội thoại.

---

#### `AssistantMessage`

Trách nhiệm:

- Render nội dung Markdown.
- Hiển thị citation.
- Hiển thị feedback buttons.
- Hiển thị warning nếu câu trả lời thiếu nguồn.

---

#### `ChatInput`

Trách nhiệm:

- Nhận input của người dùng.
- Submit message.
- Hỗ trợ keyboard shortcut.
- Disable khi đang xử lý nếu cần.

---

#### `CitationList`

Trách nhiệm:

- Hiển thị nguồn tham khảo.
- Cho phép mở rộng/thu gọn citation.
- Hiển thị điều luật hoặc đoạn trích.

---

#### `FeedbackButtons`

Trách nhiệm:

- Cho phép người dùng đánh giá câu trả lời.
- Gửi feedback về backend.
- Dữ liệu feedback có thể dùng để cải thiện model.

Các nút đề xuất:

```text
Helpful
Not helpful
Wrong citation
Missing information
Hallucination
```

---

## 9. Cấu trúc thư mục đề xuất

```text
frontend/
├── src/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── chat/
│   │   │   └── page.tsx
│   │   ├── documents/
│   │   │   └── page.tsx
│   │   ├── evaluation/
│   │   │   └── page.tsx
│   │   ├── settings/
│   │   │   └── page.tsx
│   │   ├── api/
│   │   │   └── chat/
│   │   │       └── route.ts
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── layout/
│   │   │   ├── app-header.tsx
│   │   │   └── app-sidebar.tsx
│   │   ├── chat/
│   │   │   ├── chat-interface.tsx
│   │   │   ├── chat-input.tsx
│   │   │   ├── message-list.tsx
│   │   │   ├── user-message.tsx
│   │   │   ├── assistant-message.tsx
│   │   │   ├── citation-list.tsx
│   │   │   └── feedback-buttons.tsx
│   │   ├── documents/
│   │   │   └── document-table.tsx
│   │   ├── evaluation/
│   │   │   └── evaluation-table.tsx
│   │   └── ui/
│   │       └── shadcn components
│   ├── lib/
│   │   ├── api.ts
│   │   ├── constants.ts
│   │   ├── types.ts
│   │   └── utils.ts
│   └── hooks/
│       ├── use-auto-scroll.ts
│       └── use-local-storage.ts
├── public/
├── package.json
├── tailwind.config.ts
├── next.config.ts
└── README.md
```

---

## 10. API Contract giữa Frontend và Backend

### 10.1 Request

Frontend gửi request tới:

```http
POST /api/chat
```

Body đề xuất:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Công ty đơn phương chấm dứt hợp đồng trái luật thì phải bồi thường gì?"
    }
  ],
  "conversationId": "optional-conversation-id",
  "mode": "legal_qa",
  "language": "vi",
  "includeCitations": true
}
```

---

### 10.2 Response dạng streaming

Nếu dùng Vercel AI SDK, response nên là stream text để frontend hiển thị từng token.

Nội dung assistant trả về nên có cấu trúc rõ:

```text
Theo Bộ luật Lao động 2019, nếu người sử dụng lao động đơn phương chấm dứt hợp đồng lao động trái pháp luật, họ có thể phải nhận người lao động trở lại làm việc, trả tiền lương, đóng bảo hiểm và bồi thường theo quy định...
```

---

### 10.3 Response có metadata citation

Nếu backend hỗ trợ metadata, có thể dùng dạng custom response:

```json
{
  "answer": "Câu trả lời của AI...",
  "citations": [
    {
      "title": "Bộ luật Lao động 2019",
      "article": "Điều 41",
      "snippet": "Nghĩa vụ của người sử dụng lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật...",
      "url": "https://example.com"
    }
  ],
  "confidence": 0.86
}
```

---

## 11. Next.js API Route đề xuất

Nếu backend AI chạy riêng, ví dụ FastAPI, Next.js có thể đóng vai trò proxy.

File:

```text
src/app/api/chat/route.ts
```

Ví dụ:

```ts
import { NextRequest } from 'next/server'

export async function POST(req: NextRequest) {
  const body = await req.json()

  const response = await fetch(`${process.env.BACKEND_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    return new Response('Backend error', { status: 500 })
  }

  return new Response(response.body, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
    },
  })
}
```

File `.env.local`:

```env
BACKEND_URL=http://localhost:8000
```

Lưu ý quan trọng:

- Không đặt API key LLM ở frontend.
- API key phải nằm ở backend hoặc server-side route.
- Không expose secret qua biến môi trường bắt đầu bằng `NEXT_PUBLIC_`.

---

## 12. UX cho Legal AI

Vì đây là trợ lí AI về luật, frontend nên có một số cảnh báo và thiết kế đặc thù.

### 12.1 Legal Disclaimer

Hiển thị disclaimer nhỏ ở đầu hoặc cuối trang:

```text
Thông tin do AI cung cấp chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức từ luật sư hoặc cơ quan có thẩm quyền.
```

Không nên làm disclaimer quá dài vì sẽ gây khó chịu.

---

### 12.2 Khi câu trả lời thiếu chắc chắn

Nếu backend trả về `confidence` thấp, frontend nên hiển thị cảnh báo:

```text
Câu trả lời này có độ chắc chắn thấp. Bạn nên kiểm tra lại văn bản pháp luật gốc hoặc hỏi chuyên gia pháp lý.
```

---

### 12.3 Khi không tìm thấy nguồn

Nếu backend không tìm thấy citation:

```text
Không tìm thấy nguồn pháp lý đủ chắc chắn cho câu hỏi này. Vui lòng diễn đạt lại câu hỏi hoặc kiểm tra văn bản luật gốc.
```

---

## 13. Evaluation UI cho chấm điểm AI

Vì dự án có phần đánh giá model, nên nên có giao diện riêng cho evaluation.

### 13.1 Bảng đánh giá

Các cột đề xuất:

```text
Question ID
Question
Ground Truth Answer
AI Answer
Retrieved Sources
answer_correct
citation_correct
hallucination_flag
abstention_correct
clarity
format
final_score_10
Reviewer Note
```

---

### 13.2 Form chấm điểm

Mỗi câu hỏi nên có form:

```text
Answer Correct: Yes / Partial / No
Citation Correct: Yes / Partial / No
Hallucination: Yes / No
Missing Information: None / Minor / Major
Clarity: 1-5
Format: 1-5
Final Score: 0-10
Reviewer Note: text
```

---

### 13.3 Export

Frontend nên hỗ trợ:

- Export CSV.
- Export JSON.
- Import bộ câu hỏi test.
- Lọc câu có điểm thấp.
- Lọc câu bị hallucination.
- Lọc câu citation sai.

---

## 14. Responsive Design

Giao diện cần hoạt động tốt trên:

- Laptop.
- Desktop.
- Tablet.
- Mobile cơ bản.

Breakpoint đề xuất:

```text
Mobile: < 768px
Tablet: 768px - 1024px
Desktop: > 1024px
```

Trên mobile:

- Sidebar chuyển thành drawer.
- Input vẫn cố định phía dưới.
- Citation có thể thu gọn.
- Header gọn hơn.

---

## 15. Accessibility

Yêu cầu tối thiểu:

- Có focus state rõ ràng.
- Button có `aria-label` nếu chỉ dùng icon.
- Contrast đủ tốt giữa chữ và nền.
- Có thể dùng keyboard để gửi câu hỏi.
- Không phụ thuộc hoàn toàn vào màu sắc để biểu thị trạng thái.
- Loading state có text, không chỉ spinner.

Ví dụ:

```tsx
<button aria-label="Gửi câu hỏi">
  <SendIcon />
</button>
```

---

## 16. Performance

Các nguyên tắc:

- Dùng streaming để người dùng thấy phản hồi sớm.
- Lazy load các trang ít dùng như `/evaluation` hoặc `/documents`.
- Không render lại toàn bộ message list không cần thiết.
- Giới hạn độ dài conversation gửi lên backend.
- Dùng skeleton loading thay vì màn hình trống.
- Tối ưu Markdown renderer nếu câu trả lời dài.
- Nén asset và dùng image optimization của Next.js nếu có ảnh.

---

## 17. Security Frontend

Các lưu ý:

- Không lưu API key trong frontend.
- Không hiển thị raw error từ backend cho người dùng cuối.
- Validate input cơ bản trước khi gửi.
- Giới hạn độ dài câu hỏi.
- Escape hoặc render Markdown an toàn.
- Không cho phép HTML nguy hiểm trong Markdown.
- Nếu có đăng nhập, dùng session bảo mật.
- Nếu có upload file, kiểm tra định dạng và kích thước file.

Khuyến nghị khi render Markdown:

```tsx
<ReactMarkdown
  remarkPlugins={[remarkGfm]}
  disallowedElements={['script', 'iframe']}
>
  {content}
</ReactMarkdown>
```

---

## 18. Trạng thái giao diện cần xử lý

Frontend phải xử lý đủ các trạng thái:

```text
Idle
User typing
Submitting
Streaming answer
Completed
Error
Backend unavailable
No citation found
Rate limited
Empty conversation
```

Ví dụ error message:

```text
Không thể kết nối tới máy chủ AI. Vui lòng thử lại sau.
```

Ví dụ empty state:

```text
Bạn có thể bắt đầu bằng cách hỏi: “Người lao động nghỉ việc cần báo trước bao lâu?”
```

---

## 19. Checklist triển khai

### Setup ban đầu

- [ ] Tạo project Next.js với TypeScript.
- [ ] Cài Tailwind CSS.
- [ ] Cài shadcn/ui.
- [ ] Cài Vercel AI SDK.
- [ ] Cài lucide-react.
- [ ] Tạo layout chính.
- [ ] Tạo trang `/chat`.

### Chat UI

- [ ] Tạo `ChatInterface`.
- [ ] Tạo `MessageList`.
- [ ] Tạo `ChatInput`.
- [ ] Tạo `AssistantMessage`.
- [ ] Tạo `CitationList`.
- [ ] Thêm streaming với `useChat`.
- [ ] Thêm loading state.
- [ ] Thêm error state.
- [ ] Thêm copy answer.
- [ ] Thêm regenerate answer.

### Legal AI features

- [ ] Thêm legal disclaimer.
- [ ] Hiển thị citation.
- [ ] Hiển thị confidence warning.
- [ ] Hiển thị no-source warning.
- [ ] Thêm feedback buttons.

### Evaluation

- [ ] Tạo trang `/evaluation`.
- [ ] Tạo bảng benchmark.
- [ ] Tạo form chấm điểm.
- [ ] Hỗ trợ export CSV.
- [ ] Hỗ trợ lọc câu trả lời điểm thấp.

### Production

- [ ] Kiểm tra responsive.
- [ ] Kiểm tra dark mode.
- [ ] Kiểm tra accessibility.
- [ ] Kiểm tra lỗi backend down.
- [ ] Kiểm tra deploy trên Vercel.
- [ ] Kiểm tra biến môi trường.
- [ ] Kiểm tra không lộ API key.

---

## 20. Acceptance Criteria

Frontend được coi là đạt yêu cầu khi:

- Người dùng có thể nhập câu hỏi và nhận câu trả lời AI.
- Câu trả lời hiển thị streaming mượt.
- Giao diện sạch, chuyên nghiệp, phù hợp sản phẩm enterprise.
- Có citation hoặc thông báo rõ khi không có nguồn.
- Có legal disclaimer.
- Có trạng thái loading và error rõ ràng.
- Hoạt động tốt trên desktop và mobile.
- Code chia component rõ ràng, dễ bảo trì.
- Không expose API key hoặc secret ở frontend.
- Có nền tảng để mở rộng sang evaluation, document management và user feedback.

---

## 21. Đề xuất thứ tự làm việc

Nên triển khai theo thứ tự:

1. Setup Next.js + Tailwind + shadcn/ui.
2. Làm layout `/chat`.
3. Tạo message list và input.
4. Tích hợp `useChat`.
5. Kết nối API backend.
6. Thêm Markdown rendering.
7. Thêm citation cards.
8. Thêm legal disclaimer.
9. Thêm feedback buttons.
10. Làm evaluation page.
11. Tối ưu responsive và dark mode.
12. Deploy bản demo.

---

## 22. Kết luận

Stack khuyên dùng cho frontend là:

```text
Next.js + TypeScript + Tailwind CSS + shadcn/ui + Vercel AI SDK
```

Đây là lựa chọn phù hợp cho một sản phẩm AI tạo sinh hiện đại, đặc biệt là chatbot pháp lý cần giao diện chuyên nghiệp, phản hồi nhanh, có streaming, có citation và dễ mở rộng về sau.
