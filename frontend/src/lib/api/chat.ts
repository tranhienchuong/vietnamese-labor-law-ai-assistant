import type { ChatRequestBody } from "@/lib/types"

export function sendChatRequest(
  body: ChatRequestBody,
  signal?: AbortSignal,
  url = "/api/chat"
) {
  return fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body),
    signal
  })
}
