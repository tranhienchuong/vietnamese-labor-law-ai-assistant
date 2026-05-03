import { NextRequest, NextResponse } from "next/server"
import { authHeaders, authTokenFromRequest, backendBaseUrl } from "@/lib/server-api"

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const token = authTokenFromRequest(request)
  if (!token) {
    return NextResponse.json({ error: "Authentication required." }, { status: 401 })
  }

  const { id } = await params
  const backendResponse = await fetch(
    `${backendBaseUrl()}/conversations/${encodeURIComponent(id)}`,
    {
      headers: authHeaders(token),
      cache: "no-store"
    }
  )
  const payload = await backendResponse.json().catch(() => ({}))
  return NextResponse.json(payload, { status: backendResponse.status })
}
