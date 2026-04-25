import type { Citation, DocumentRecord, EvaluationRecord } from "@/lib/types"

export const APP_NAME = "Vietnam Labor Law AI Assistant"

export const APP_TAGLINE =
  "Hỏi đáp thông minh về chấm dứt hợp đồng lao động theo Bộ luật Lao động Việt Nam 2019."

export const LEGAL_DISCLAIMER =
  "Thông tin do AI cung cấp chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức từ luật sư hoặc cơ quan có thẩm quyền."

export const EXAMPLE_QUESTIONS = [
  "Người lao động ký hợp đồng không xác định thời hạn muốn nghỉ việc thì phải báo trước bao lâu?",
  "Công ty đơn phương chấm dứt hợp đồng trái luật thì phải bồi thường gì?",
  "Khi nào người lao động có quyền đơn phương chấm dứt hợp đồng lao động mà không cần báo trước?",
  "Trợ cấp thôi việc theo Điều 46 được tính như thế nào?"
]

export const RECENT_CHATS = [
  "Báo trước khi nghỉ việc",
  "Bồi thường do chấm dứt trái luật",
  "Trợ cấp thôi việc và mất việc"
]

export const DEMO_CITATIONS: Citation[] = [
  {
    title: "Bộ luật Lao động 2019",
    article: "Điều 35",
    snippet:
      "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động và phải báo trước theo loại hợp đồng, trừ một số trường hợp đặc biệt.",
    relevance: "Cao"
  },
  {
    title: "Bộ luật Lao động 2019",
    article: "Điều 41",
    snippet:
      "Quy định nghĩa vụ của người sử dụng lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật.",
    relevance: "Cao"
  },
  {
    title: "Bộ luật Lao động 2019",
    article: "Điều 46",
    snippet:
      "Quy định về trợ cấp thôi việc, điều kiện hưởng và cách tính theo thời gian làm việc thực tế.",
    relevance: "Trung bình"
  }
]

export const DOCUMENTS: DocumentRecord[] = [
  {
    id: "45-2019-qh14",
    name: "Bộ luật Lao động 2019",
    type: "Văn bản luật",
    chunks: 348,
    status: "completed",
    updatedAt: "2026-04-25"
  },
  {
    id: "145-2020-nd-cp",
    name: "Nghị định 145/2020/NĐ-CP",
    type: "Nghị định",
    chunks: 226,
    status: "completed",
    updatedAt: "2026-04-25"
  },
  {
    id: "internal-guidance",
    name: "Hướng dẫn nội bộ về chấm dứt hợp đồng",
    type: "Tài liệu nội bộ",
    chunks: 0,
    status: "pending",
    updatedAt: "Chưa index"
  }
]

export const EVALUATION_ROWS: EvaluationRecord[] = [
  {
    id: "Q-001",
    question: "Người lao động nghỉ việc cần báo trước bao lâu?",
    expectedCitation: "Bộ luật Lao động 2019, Điều 35",
    aiAnswer: "Cần xác định loại hợp đồng và trường hợp miễn báo trước.",
    answerCorrect: "partial",
    citationCorrect: "yes",
    hallucination: "no",
    finalScore: 7.5
  },
  {
    id: "Q-014",
    question: "Công ty chấm dứt hợp đồng trái luật phải bồi thường gì?",
    expectedCitation: "Bộ luật Lao động 2019, Điều 41",
    aiAnswer: "Có thể phải nhận lại làm việc, trả lương, BHXH và bồi thường.",
    answerCorrect: "yes",
    citationCorrect: "yes",
    hallucination: "no",
    finalScore: 9
  },
  {
    id: "Q-039",
    question: "Trợ cấp thôi việc áp dụng khi nào?",
    expectedCitation: "Bộ luật Lao động 2019, Điều 46",
    aiAnswer: "Cần kiểm tra thời gian làm việc và thời gian đã đóng bảo hiểm thất nghiệp.",
    answerCorrect: "partial",
    citationCorrect: "partial",
    hallucination: "no",
    finalScore: 6.5
  }
]
