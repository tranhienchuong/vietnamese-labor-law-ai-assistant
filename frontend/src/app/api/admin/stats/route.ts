import { NextRequest } from "next/server"
import { proxyBackendJson } from "@/lib/api/server-proxy"

export async function GET(request: NextRequest) {
  return proxyBackendJson(request, "/admin/stats")
}
