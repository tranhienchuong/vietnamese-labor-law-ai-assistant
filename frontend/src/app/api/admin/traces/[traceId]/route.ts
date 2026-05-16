import { NextRequest } from "next/server"
import { proxyBackendJson } from "@/lib/api/server-proxy"

type RouteContext = {
  params: Promise<{ traceId: string }>
}

export async function GET(request: NextRequest, context: RouteContext) {
  const { traceId } = await context.params
  return proxyBackendJson(request, `/admin/traces/${encodeURIComponent(traceId)}`)
}
