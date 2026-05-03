import { NextRequest, NextResponse } from "next/server"
import {
  AUTH_COOKIE_NAME,
  authCookieOptions,
  authHeaders,
  authTokenFromRequest,
  backendBaseUrl
} from "@/lib/server-api"

export async function POST(request: NextRequest) {
  const token = authTokenFromRequest(request)
  if (token) {
    await fetch(`${backendBaseUrl()}/auth/logout`, {
      method: "POST",
      headers: authHeaders(token)
    }).catch(() => undefined)
  }

  const response = NextResponse.json({ ok: true })
  response.cookies.set(AUTH_COOKIE_NAME, "", authCookieOptions(0))
  return response
}
