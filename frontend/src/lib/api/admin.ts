import type {
  AdminHealthResponse,
  AdminRetrievalConfigResponse,
  AdminStatsResponse
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
