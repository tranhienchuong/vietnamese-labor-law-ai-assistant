import { NextRequest, NextResponse } from "next/server"
import { AUTH_COOKIE_NAME, authCookieOptions, backendBaseUrl } from "@/lib/server-api"

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => ({}))
  const backendResponse = await fetch(`${backendBaseUrl()}/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  })

  const payload = await backendResponse.json().catch(() => ({}))
  if (!backendResponse.ok) {
    return NextResponse.json(payload, { status: backendResponse.status })
  }

  const accessToken = String(payload.accessToken || "")
  const expiresAt = Number(payload.expiresAt || 0)
  const maxAge = expiresAt > 0 ? Math.max(0, expiresAt - Math.floor(Date.now() / 1000)) : undefined
  const response = NextResponse.json({ user: payload.user })
  response.cookies.set(AUTH_COOKIE_NAME, accessToken, authCookieOptions(maxAge))
  return response
}
