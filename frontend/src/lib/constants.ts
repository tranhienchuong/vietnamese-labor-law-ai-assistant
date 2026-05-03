import type {
  AdminMetric,
  Citation,
  DocumentRecord,
  EvaluationRecord,
  RetrievalRecord,
  SystemLogRecord
} from "@/lib/types"

export const APP_NAME = "Trợ lý AI Luật Lao động Việt Nam"

export const APP_TAGLINE =
  "Hỗ trợ hỏi đáp về chấm dứt hợp đồng, thời hạn báo trước, bồi thường và trợ cấp theo pháp luật lao động Việt Nam."

export const LEGAL_DISCLAIMER =
  "Thông tin do AI cung cấp chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức từ luật sư hoặc cơ quan có thẩm quyền."

export const EXAMPLE_QUESTIONS = [
  "Người lao động ký hợp đồng không xác định thời hạn muốn nghỉ việc thì phải báo trước bao lâu?",
  "Công ty đơn phương chấm dứt hợp đồng trái luật thì phải bồi thường gì?",
  "Khi nào người lao động được nghỉ việc mà không cần báo trước?",
  "Trợ cấp thôi việc theo Điều 46 được tính như thế nào?"
]

export const USER_BENEFITS = [
  {
    title: "Trả lời nhanh",
    description: "Đặt câu hỏi trực tiếp và nhận câu trả lời ngắn gọn, dễ hiểu."
  },
  {
    title: "Có căn cứ pháp lý",
    description: "Mỗi câu trả lời ưu tiên đi kèm điều luật hoặc văn bản liên quan."
  },
  {
    title: "Dễ hỏi tiếp",
    description: "Tiếp tục cuộc trò chuyện trong cùng ngữ cảnh khi cần làm rõ thêm."
  }
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
      "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động và phải báo trước theo loại hợp đồng, trừ một số trường hợp đặc biệt."
  },
  {
    title: "Bộ luật Lao động 2019",
    article: "Điều 41",
    snippet:
      "Quy định nghĩa vụ của người sử dụng lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật."
  },
  {
    title: "Bộ luật Lao động 2019",
    article: "Điều 46",
    snippet:
      "Quy định về trợ cấp thôi việc, điều kiện hưởng và cách tính theo thời gian làm việc thực tế."
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
    aiAnswer:
      "Có thể phải nhận lại người lao động, trả lương, đóng bảo hiểm và bồi thường.",
    answerCorrect: "yes",
    citationCorrect: "yes",
    hallucination: "no",
    finalScore: 9
  },
  {
    id: "Q-039",
    question: "Trợ cấp thôi việc áp dụng khi nào?",
    expectedCitation: "Bộ luật Lao động 2019, Điều 46",
    aiAnswer:
      "Cần kiểm tra thời gian làm việc và thời gian đã đóng bảo hiểm thất nghiệp.",
    answerCorrect: "partial",
    citationCorrect: "partial",
    hallucination: "no",
    finalScore: 6.5
  }
]

export const ADMIN_METRICS: AdminMetric[] = [
  {
    label: "Tài liệu đã index",
    value: "2",
    description: "Bộ luật Lao động 2019 và Nghị định 145/2020/NĐ-CP",
    tone: "success"
  },
  {
    label: "Phiên chat gần đây",
    value: "18",
    description: "Số phiên thử nghiệm trong 7 ngày gần nhất"
  },
  {
    label: "Câu trả lời có citation",
    value: "86%",
    description: "Tỷ lệ benchmark có căn cứ pháp lý đi kèm",
    tone: "success"
  },
  {
    label: "Lỗi truy xuất",
    value: "2",
    description: "Cần kiểm tra các truy vấn thiếu nguồn phù hợp",
    tone: "warning"
  }
]

export const RETRIEVAL_ROWS: RetrievalRecord[] = [
  {
    id: "R-1021",
    query: "báo trước khi nghỉ việc hợp đồng không xác định thời hạn",
    sources: 3,
    selectedChunk: "BLLD2019-035-01",
    relevanceScore: "0.91",
    confidence: "0.84",
    metadata: "Bộ luật Lao động 2019, Điều 35",
    status: "ok"
  },
  {
    id: "R-1022",
    query: "bồi thường khi công ty chấm dứt hợp đồng trái luật",
    sources: 4,
    selectedChunk: "BLLD2019-041-01",
    relevanceScore: "0.89",
    confidence: "0.82",
    metadata: "Bộ luật Lao động 2019, Điều 41",
    status: "ok"
  },
  {
    id: "R-1023",
    query: "trợ cấp thôi việc có tính thời gian thử việc không",
    sources: 1,
    selectedChunk: "BLLD2019-046-02",
    relevanceScore: "0.62",
    confidence: "0.55",
    metadata: "Cần bổ sung hướng dẫn chi tiết",
    status: "warning"
  }
]

export const SYSTEM_LOGS: SystemLogRecord[] = [
  {
    id: "LOG-2301",
    level: "info",
    area: "chat",
    message: "Hoàn tất phiên hỏi đáp có citation.",
    createdAt: "2026-05-03 09:46"
  },
  {
    id: "LOG-2302",
    level: "warning",
    area: "retrieval",
    message: "Một truy vấn chỉ có 1 nguồn đạt ngưỡng liên quan.",
    createdAt: "2026-05-03 09:51"
  },
  {
    id: "LOG-2303",
    level: "error",
    area: "index",
    message: "Tài liệu nội bộ đang chờ index, chưa có chunk khả dụng.",
    createdAt: "2026-05-03 09:58"
  }
]
