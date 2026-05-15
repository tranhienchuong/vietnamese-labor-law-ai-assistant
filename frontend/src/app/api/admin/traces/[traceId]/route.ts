import { NextRequest, NextResponse } from "next/server"
import { authHeaders, authTokenFromRequest, backendBaseUrl } from "@/lib/server-api"

type RouteContext = {
  params: Promise<{ traceId: string }>
}

export async function GET(request: NextRequest, context: RouteContext) {
  const token = authTokenFromRequest(request)
  if (!token) {
    return NextResponse.json({ error: "Authentication required." }, { status: 401 })
  }

  const { traceId } = await context.params
  const backendResponse = await fetch(
    `${backendBaseUrl()}/admin/traces/${encodeURIComponent(traceId)}`,
    {
      headers: authHeaders(token),
      cache: "no-store"
    }
  )
  const payload = await backendResponse.json().catch(() => ({}))
  return NextResponse.json(payload, { status: backendResponse.status })
}
