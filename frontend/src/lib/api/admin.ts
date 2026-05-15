import type {
  AdminHealthResponse,
  AdminRetrievalConfigResponse,
  AdminStatsResponse,
  AdminTraceDetailResponse,
  AdminTracesResponse
} from "@/lib/types"

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message =
      typeof payload.error === "string" ? payload.error : "Request failed."
    throw new Error(message)
  }
  return payload as T
}

export function getAdminStats() {
  return getJson<AdminStatsResponse>("/api/admin/stats")
}

export function getAdminHealth() {
  return getJson<AdminHealthResponse>("/api/admin/health")
}

export function getAdminRetrievalConfig() {
  return getJson<AdminRetrievalConfigResponse>("/api/admin/retrieval-config")
}

export function getAdminTraces(params?: URLSearchParams) {
  const query = params?.toString()
  return getJson<AdminTracesResponse>(`/api/admin/traces${query ? `?${query}` : ""}`)
}

export function getAdminTrace(traceId: string) {
  return getJson<AdminTraceDetailResponse>(
    `/api/admin/traces/${encodeURIComponent(traceId)}`
  )
}
