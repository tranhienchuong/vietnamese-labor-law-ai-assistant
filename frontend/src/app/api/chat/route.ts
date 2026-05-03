import { NextRequest } from "next/server"
import { authHeaders, authTokenFromRequest, backendBaseUrl } from "@/lib/server-api"

export const runtime = "nodejs"

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}))
  const token = authTokenFromRequest(req)

  if (!token) {
    return new Response("Authentication required", { status: 401 })
  }

  const response = await fetch(`${backendBaseUrl()}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(token)
    },
    body: JSON.stringify(body)
  })

  if (!response.ok || !response.body) {
    const errorText = await response.text().catch(() => "Backend error")
    return new Response(errorText || "Backend error", { status: response.status })
  }

  return new Response(response.body, {
    headers: {
      "Cache-Control": "no-cache",
      "Content-Type": response.headers.get("Content-Type") ?? "text/plain; charset=utf-8",
      "X-Conversation-Id": response.headers.get("X-Conversation-Id") ?? ""
    }
  })
}
