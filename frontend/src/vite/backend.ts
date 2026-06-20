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
    created_at?: string
  }>
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
          : `Request failed with status ${response.status}`
    throw new ApiError(message, response.status)
  }
  return payload as T
}

export async function sendChatQuestion(
  session: Session,
  question: string,
  conversationId?: string | null
) {
  const response = await fetch(`${apiBaseUrl}/chat`, {
    method: "POST",
    headers: {
      ...authHeaders(session),
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      messages: [{ role: "user", content: question }],
      includeCitations: true,
      ...(conversationId ? { conversationId } : {})
    })
  })
  const answer = await response.text()
  if (!response.ok) {
    throw new Error(answer || `Chat request failed with status ${response.status}`)
  }
  return {
    answer,
    conversationId: response.headers.get("X-Conversation-Id")
  }
}

export async function listConversations(session: Session) {
  const response = await fetch(`${apiBaseUrl}/conversations`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<{ conversations: ConversationSummary[] }>(response)
}

export async function createConversation(session: Session, title = "New research") {
  const response = await fetch(`${apiBaseUrl}/conversations`, {
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
  const response = await fetch(`${apiBaseUrl}/conversations/${conversationId}`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<ConversationDetail>(response)
}

export async function getCurrentUser(session: Session) {
  const response = await fetch(`${apiBaseUrl}/auth/me`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<{ user: CurrentUser }>(response)
}

export async function getAdminStats(session: Session) {
  const response = await fetch(`${apiBaseUrl}/admin/stats`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<{
    user: CurrentUser
    stats: AdminStats
    runtime: Record<string, unknown>
  }>(response)
}

export async function getAdminHealth(session: Session) {
  const response = await fetch(`${apiBaseUrl}/admin/health`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<{
    status: string
    checks: Record<string, Record<string, unknown>>
  }>(response)
}

export async function getAdminRetrievalConfig(session: Session) {
  const response = await fetch(`${apiBaseUrl}/admin/retrieval-config`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<Record<string, unknown>>(response)
}

export async function getAdminTraces(session: Session, limit = 20) {
  const response = await fetch(`${apiBaseUrl}/admin/traces?limit=${limit}`, {
    headers: authHeaders(session)
  })
  return parseJsonResponse<{ traces: AdminTrace[] }>(response)
}
