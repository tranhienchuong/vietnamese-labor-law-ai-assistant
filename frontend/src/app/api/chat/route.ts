import { NextRequest } from "next/server"
import { proxyBackendStream } from "@/lib/api/server-proxy"

export const runtime = "nodejs"

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}))
  return proxyBackendStream(req, "/chat", body)
}
