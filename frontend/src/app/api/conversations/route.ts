import { NextRequest, NextResponse } from "next/server"
import { authHeaders, authTokenFromRequest, backendBaseUrl } from "@/lib/server-api"

export async function GET(request: NextRequest) {
  const token = authTokenFromRequest(request)
  if (!token) {
    return NextResponse.json({ error: "Authentication required." }, { status: 401 })
  }

  const backendResponse = await fetch(`${backendBaseUrl()}/conversations`, {
    headers: authHeaders(token),
    cache: "no-store"
  })
  const payload = await backendResponse.json().catch(() => ({}))
  return NextResponse.json(payload, { status: backendResponse.status })
}

export async function POST(request: NextRequest) {
  const token = authTokenFromRequest(request)
  if (!token) {
    return NextResponse.json({ error: "Authentication required." }, { status: 401 })
  }

  const body = await request.json().catch(() => ({}))
  const backendResponse = await fetch(`${backendBaseUrl()}/conversations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(token)
    },
    body: JSON.stringify(body)
  })
  const payload = await backendResponse.json().catch(() => ({}))
  return NextResponse.json(payload, { status: backendResponse.status })
}
