import type { Session } from "@supabase/supabase-js"
import { apiBaseUrl } from "./config"

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

export type ChatMessage = {
  role: "user" | "assistant"
  content: string
  citations?: ChatCitations
}

export type EvidenceQuote = {
  citation: string
  quote: string
}

export type ChatCitations = {
  legalBasis: string[]
  evidenceQuotes: EvidenceQuote[]
}

export type CurrentUser = {
  id: string
  name: string
  email: string
  role: "user" | "admin"
  avatarUrl?: string | null
}

export type ConversationSummary = {
  id: string
  title: string
  created_at: string
  updated_at: string
  last_message_at?: string | null
  message_count: number
}

export type ConversationDetail = {
  conversation: ConversationSummary
  messages: Array<{
    id: string
    role: "user" | "assistant" | "system"
    content: string
    citations?: unknown
    created_at?: string
  }>
}

type ChatResponsePayload = {
  answer?: unknown
  legalBasis?: unknown
  legal_basis?: unknown
  evidenceQuotes?: unknown
  evidence_quotes?: unknown
  citations?: unknown
}

export type AdminStats = {
  totalUsers: number
  activeUsers: number
  adminUsers: number
  totalConversations: number
  totalMessages: number
  activeSessions?: number
  totalTraces: number
  tracesWithErrors: number
  insufficientContextTraces: number
}

export type AdminTrace = {
  id: string
  createdAt?: string
  created_at?: string
  question: string
  provider?: string | null
  model?: string | null
  insufficientContext?: boolean
  insufficient_context?: boolean
  error?: string | null
  latencyMs?: number | null
  latency_ms?: number | null
}

function authHeaders(session: Session): HeadersInit {
  return {
    Authorization: `Bearer ${session.access_token}`
  }
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message =
      typeof payload?.detail === "string"
        ? payload.detail
        : typeof payload?.error === "string"
          ? payload.error
          : `Yêu cầu thất bại với mã ${response.status}`
    throw new ApiError(localizeApiMessage(message), response.status)
  }
  return payload as T
}

async function apiFetch(input: string, init?: RequestInit) {
  try {
    return await fetch(input, init)
  } catch {
    const message =
      `Không thể kết nối backend tại ${apiBaseUrl}. ` +
      "Hãy kiểm tra FastAPI server đã chạy, VITE_API_BASE_URL đúng và kết nối cơ sở dữ liệu Supabase hợp lệ."
    throw new ApiError(message, 0)
  }
}

function localizeApiMessage(message: string): string {
  const normalized = message.toLowerCase()
  if (normalized.includes("authentication required")) {
    return "Bạn cần đăng nhập để tiếp tục."
  }
  if (normalized.includes("invalid or expired session")) {
    return "Phiên đăng nhập không hợp lệ hoặc đã hết hạn."
  }
  if (normalized.includes("conversation not found")) {
    return "Không tìm thấy cuộc trò chuyện."
  }
  if (normalized.includes("admin role required")) {
    return "Bạn cần quyền quản trị để xem mục này."
  }
  if (normalized.includes("extractive generation")) {
    return "Chế độ trích xuất chỉ dùng cho đánh giá, không dùng trong giao diện hỏi đáp."
  }
  if (normalized.includes("request failed with status")) {
    return message.replace("Request failed with status", "Yêu cầu thất bại với mã")
  }
  if (normalized.includes("chat request failed with status")) {
    return message.replace("Chat request failed with status", "Yêu cầu hỏi đáp thất bại với mã")
  }
  return message
}

export function emptyChatCitations(): ChatCitations {
  return {
    legalBasis: [],
    evidenceQuotes: []
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

export function normalizeChatCitations(value: unknown): ChatCitations {
  const record = asRecord(value)
  const nested = asRecord(record.citations)
  const legalBasisSource =
    record.legalBasis ?? record.legal_basis ?? nested.legalBasis ?? nested.legal_basis
  const evidenceQuotesSource =
    record.evidenceQuotes ?? record.evidence_quotes ?? nested.evidenceQuotes ?? nested.evidence_quotes

  const legalBasis = Array.isArray(legalBasisSource)
    ? legalBasisSource.map((item) => String(item || "").trim()).filter(Boolean)
    : []
  const evidenceQuotes = Array.isArray(evidenceQuotesSource)
    ? evidenceQuotesSource.flatMap((item) => {
        const quoteRecord = asRecord(item)
        const citation = String(quoteRecord.citation || "").trim()
        const quote = String(quoteRecord.quote || "").trim()
        return citation && quote ? [{ citation, quote }] : []
      })
    : []

  return {
    legalBasis,
    evidenceQuotes
  }
}

export async function sendChatQuestion(
  session: Session,
  question: string,
  conversationId?: string | null
) {
  const response = await apiFetch(`${apiBaseUrl}/chat`, {
    method: "POST",
    headers: {
      ...authHeaders(session),
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      messages: [{ role: "user", content: question }],
      includeCitations: true,
      responseFormat: "json",
      ...(conversationId ? { conversationId } : {})
    })
  })
  const contentType = response.headers.get("content-type") || ""
  if (!response.ok) {
    const payload = contentType.includes("application/json")
      ? await response.json().catch(() => ({}))
      : {}
    const fallback = contentType.includes("application/json")
      ? ""
      : await response.text().catch(() => "")
    const message =
      typeof payload?.detail === "string"
        ? payload.detail
        : typeof payload?.error === "string"
          ? payload.error
          : fallback || `Yêu cầu hỏi đáp thất bại với mã ${response.status}`
    throw new Error(localizeApiMessage(message))
  }
  if (contentType.includes("application/json")) {
    const payload = (await response.json()) as ChatResponsePayload
    return {
      answer: String(payload.answer || ""),
      citations: normalizeChatCitations(payload),
      conversationId: response.headers.get("X-Conversation-Id")
    }
  }
  const answer = await response.text()
  return {
    answer,
    citations: emptyChatCitations(),
    conversationId: response.headers.get("X-Conversation-Id")
  }
}

export async function listConversations(session: Session) {
  const response = await apiFetch(`${apiBaseUrl}/conversations`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<{ conversations: ConversationSummary[] }>(response)
}

export async function createConversation(session: Session, title = "Cuộc trò chuyện mới") {
  const response = await apiFetch(`${apiBaseUrl}/conversations`, {
    method: "POST",
    headers: {
      ...authHeaders(session),
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ title })
  })
  return parseJsonResponse<{ conversation: ConversationSummary }>(response)
}

export async function getConversation(session: Session, conversationId: string) {
  const response = await apiFetch(`${apiBaseUrl}/conversations/${conversationId}`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<ConversationDetail>(response)
}

export async function getCurrentUser(session: Session) {
  const response = await apiFetch(`${apiBaseUrl}/auth/me`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<{ user: CurrentUser }>(response)
}

export async function getAdminStats(session: Session) {
  const response = await apiFetch(`${apiBaseUrl}/admin/stats`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<{
    user: CurrentUser
    stats: AdminStats
    runtime: Record<string, unknown>
  }>(response)
}

export async function getAdminHealth(session: Session) {
  const response = await apiFetch(`${apiBaseUrl}/admin/health`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<{
    status: string
    checks: Record<string, Record<string, unknown>>
  }>(response)
}

export async function getAdminRetrievalConfig(session: Session) {
  const response = await apiFetch(`${apiBaseUrl}/admin/retrieval-config`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<Record<string, unknown>>(response)
}

export async function getAdminTraces(session: Session, limit = 20) {
  const response = await apiFetch(`${apiBaseUrl}/admin/traces?limit=${limit}`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<{ traces: AdminTrace[] }>(response)
}
