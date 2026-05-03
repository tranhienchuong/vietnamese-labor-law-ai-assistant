import type { NextRequest } from "next/server"

export const AUTH_COOKIE_NAME = "auth_token"

export function backendBaseUrl() {
  return (process.env.BACKEND_URL || "http://localhost:8000").replace(/\/$/, "")
}

export function authTokenFromRequest(request: NextRequest) {
  return request.cookies.get(AUTH_COOKIE_NAME)?.value ?? ""
}

export function authCookieOptions(maxAge?: number) {
  return {
    httpOnly: true,
    sameSite: "lax" as const,
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge
  }
}

export function authHeaders(token: string): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {}
}
