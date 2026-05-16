import "server-only"

import { NextRequest, NextResponse } from "next/server"
import { authHeaders, authTokenFromRequest, backendBaseUrl } from "@/lib/server-api"

type JsonProxyOptions = {
  method?: string
  body?: unknown
  headers?: HeadersInit
}

function backendUrl(path: string) {
  return `${backendBaseUrl()}${path.startsWith("/") ? path : `/${path}`}`
}

function authRequiredJsonResponse() {
  return NextResponse.json({ error: "Authentication required." }, { status: 401 })
}

function authRequiredStreamResponse() {
  return new Response("Authentication required", { status: 401 })
}

function requestHeaders(token: string, headers?: HeadersInit, hasBody = false) {
  const nextHeaders = new Headers(headers)
  for (const [key, value] of Object.entries(authHeaders(token))) {
    nextHeaders.set(key, value)
  }
  if (hasBody && !nextHeaders.has("Content-Type")) {
    nextHeaders.set("Content-Type", "application/json")
  }
  return nextHeaders
}

export async function proxyBackendJson(
  request: NextRequest,
  backendPath: string,
  options: JsonProxyOptions = {}
) {
  const token = authTokenFromRequest(request)
  if (!token) {
    return authRequiredJsonResponse()
  }

  const hasBody = options.body !== undefined
  const backendResponse = await fetch(backendUrl(backendPath), {
    method: options.method ?? "GET",
    headers: requestHeaders(token, options.headers, hasBody),
    body: hasBody ? JSON.stringify(options.body) : undefined,
    cache: "no-store"
  })
  const payload = await backendResponse.json().catch(() => ({}))
  return NextResponse.json(payload, { status: backendResponse.status })
}

export async function proxyBackendStream(
  request: NextRequest,
  backendPath: string,
  body: unknown
) {
  const token = authTokenFromRequest(request)
  if (!token) {
    return authRequiredStreamResponse()
  }

  const backendResponse = await fetch(backendUrl(backendPath), {
    method: "POST",
    headers: requestHeaders(token, undefined, true),
    body: JSON.stringify(body)
  })

  if (!backendResponse.ok || !backendResponse.body) {
    const errorText = await backendResponse.text().catch(() => "Backend error")
    return new Response(errorText || "Backend error", { status: backendResponse.status })
  }

  return new Response(backendResponse.body, {
    headers: {
      "Cache-Control": "no-cache",
      "Content-Type": backendResponse.headers.get("Content-Type") ?? "text/plain; charset=utf-8",
      "X-Conversation-Id": backendResponse.headers.get("X-Conversation-Id") ?? ""
    }
  })
}
