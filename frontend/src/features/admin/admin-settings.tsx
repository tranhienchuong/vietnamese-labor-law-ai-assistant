"use client"

import {
  Braces,
  Database,
  RefreshCw,
  Search,
  Settings2,
  SlidersHorizontal
} from "lucide-react"
import type { ComponentType, ReactNode } from "react"
import { useEffect, useState } from "react"
import { AppPageShell } from "@/components/layout/app-page-shell"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getAdminRetrievalConfig } from "@/lib/api/admin"
import type { AdminRetrievalConfigResponse } from "@/lib/types"

export function AdminSettings() {
  const [config, setConfig] = useState<AdminRetrievalConfigResponse | null>(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(true)

  async function loadConfig() {
    setLoading(true)
    setError("")
    try {
      setConfig(await getAdminRetrievalConfig())
    } catch (err) {
      setError(err instanceof Error ? err.message : "Khong the tai cau hinh.")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadConfig()
  }, [])

  return (
    <AppPageShell
      actions={<Button disabled>Chi doc trong phase nay</Button>}
      description="Cau hinh retrieval va model dang duoc backend su dung."
      title="Cau hinh he thong"
    >
      {loading ? (
        <StatePanel label="Dang tai cau hinh..." />
      ) : error ? (
        <StatePanel
          action={<Button onClick={loadConfig}>Thu lai</Button>}
          label={error}
          tone="destructive"
        />
      ) : config ? (
        <div className="grid gap-5 lg:grid-cols-[1fr_0.75fr]">
          <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
            <div className="mb-5 flex items-center gap-3">
              <SlidersHorizontal className="h-5 w-5 text-primary" />
              <div>
                <h2 className="text-base font-semibold">Retrieval runtime</h2>
                <p className="text-sm text-muted-foreground">
                  {config.indexPath}
                </p>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <ConfigTile
                icon={Database}
                label="Qdrant collection"
                value={config.qdrantCollection || "default"}
              />
              <ConfigTile
                icon={Search}
                label="Record source"
                value={config.retrieverRecordSource || "sqlite"}
              />
              <ConfigTile
                icon={Settings2}
                label="Dense model"
                value={config.denseModel}
              />
              <ConfigTile
                icon={Braces}
                label="Embedding provider"
                value={config.embeddingProvider}
              />
            </div>
          </section>

          <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
            <h2 className="text-base font-semibold">Routing va reranking</h2>
            <div className="mt-4 space-y-3">
              <ToggleRow
                label="Query router"
                value={config.queryRouterEnabled ? "enabled" : "disabled"}
              />
              <ToggleRow
                label="Router provider"
                value={config.queryRouterProvider || "default"}
              />
              <ToggleRow
                label="Router model"
                value={config.queryRouterModel || "default"}
              />
              <ToggleRow
                label="Fallback heuristic"
                value={config.queryRouterFallbackToHeuristic ? "enabled" : "disabled"}
              />
              <ToggleRow
                label="Reranker"
                value={
                  config.rerankerEnabled
                    ? `${config.rerankerModel} | top ${config.rerankerTopN}`
                    : "disabled"
                }
              />
            </div>
          </section>
        </div>
      ) : null}
    </AppPageShell>
  )
}

function ConfigTile({
  icon: Icon,
  label,
  value
}: {
  icon: ComponentType<{ className?: string }>
  label: string
  value: string
}) {
  return (
    <div className="rounded-md border border-border bg-background p-3">
      <Icon className="h-4 w-4 text-primary" />
      <p className="mt-3 text-sm text-muted-foreground">{label}</p>
      <p className="mt-1 break-words text-sm font-semibold">{value}</p>
    </div>
  )
}

function ToggleRow({ label, value }: { label: string; value: string }) {
  const variant = value === "enabled" ? "success" : "secondary"
  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-background px-3 py-3">
      <span className="text-sm font-medium">{label}</span>
      <Badge variant={variant}>{value}</Badge>
    </div>
  )
}

function StatePanel({
  action,
  label,
  tone = "default"
}: {
  action?: ReactNode
  label: string
  tone?: "default" | "destructive"
}) {
  return (
    <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <RefreshCw
            className={
              tone === "destructive" ? "h-5 w-5 text-destructive" : "h-5 w-5 text-primary"
            }
          />
          <p className="text-sm font-medium">{label}</p>
        </div>
        {action}
      </div>
    </section>
  )
}
