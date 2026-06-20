import type { Session } from "@supabase/supabase-js"
import { apiBaseUrl } from "./config"

export type ChatMessage = {
  role: "user" | "assistant"
  content: string
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
    throw new Error(message)
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
