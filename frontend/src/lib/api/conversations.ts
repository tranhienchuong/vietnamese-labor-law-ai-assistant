import { apiFetchJson } from "@/lib/api/client"
import type {
  ConversationCreateResponse,
  ConversationDetailResponse,
  ConversationMessagesResponse,
  ConversationsResponse
} from "@/lib/types"

export function listConversations() {
  return apiFetchJson<ConversationsResponse>("/api/conversations")
}

export function createConversation(title?: string) {
  return apiFetchJson<ConversationCreateResponse>("/api/conversations", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ title })
  })
}

export function getConversation(conversationId: string) {
  return apiFetchJson<ConversationDetailResponse>(
    `/api/conversations/${encodeURIComponent(conversationId)}`
  )
}

export function listConversationMessages(conversationId: string) {
  return apiFetchJson<ConversationMessagesResponse>(
    `/api/conversations/${encodeURIComponent(conversationId)}/messages`
  )
}
