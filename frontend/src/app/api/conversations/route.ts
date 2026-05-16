import { NextRequest } from "next/server"
import { proxyBackendJson } from "@/lib/api/server-proxy"

export async function GET(request: NextRequest) {
  return proxyBackendJson(request, "/conversations")
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => ({}))
  return proxyBackendJson(request, "/conversations", {
    method: "POST",
    body
  })
}
