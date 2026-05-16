import { apiFetchJson } from "@/lib/api/client"
import type {
  AdminHealthResponse,
  AdminRetrievalConfigResponse,
  AdminStatsResponse,
  AdminTraceDetailResponse,
  AdminTracesResponse
} from "@/lib/types"

export function getAdminStats() {
  return apiFetchJson<AdminStatsResponse>("/api/admin/stats")
}

export function getAdminHealth() {
  return apiFetchJson<AdminHealthResponse>("/api/admin/health")
}

export function getAdminRetrievalConfig() {
  return apiFetchJson<AdminRetrievalConfigResponse>("/api/admin/retrieval-config")
}

export function getAdminTraces(params?: URLSearchParams) {
  const query = params?.toString()
  return apiFetchJson<AdminTracesResponse>(`/api/admin/traces${query ? `?${query}` : ""}`)
}

export function getAdminTrace(traceId: string) {
  return apiFetchJson<AdminTraceDetailResponse>(
    `/api/admin/traces/${encodeURIComponent(traceId)}`
  )
}
