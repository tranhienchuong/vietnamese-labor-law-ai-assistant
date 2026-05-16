import { NextRequest } from "next/server"
import { proxyBackendJson } from "@/lib/api/server-proxy"

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params
  return proxyBackendJson(request, `/conversations/${encodeURIComponent(id)}/messages`)
}
