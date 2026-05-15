"use client"

import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Database,
  MessageSquareText,
  RefreshCw,
  Search,
  ShieldCheck,
  Users
} from "lucide-react"
import type { ComponentType, ReactNode } from "react"
import { useEffect, useMemo, useState } from "react"
import { AppPageShell } from "@/components/layout/app-page-shell"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  getAdminHealth,
  getAdminRetrievalConfig,
  getAdminStats
} from "@/lib/api/admin"
import type {
  AdminHealthResponse,
  AdminRetrievalConfigResponse,
  AdminStatsResponse
} from "@/lib/types"

type DashboardState = {
  stats: AdminStatsResponse
  health: AdminHealthResponse
  retrieval: AdminRetrievalConfigResponse
}

export function AdminDashboard() {
  const [data, setData] = useState<DashboardState | null>(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(true)

  async function loadDashboard() {
    setLoading(true)
    setError("")
    try {
      const [stats, health, retrieval] = await Promise.all([
        getAdminStats(),
        getAdminHealth(),
        getAdminRetrievalConfig()
      ])
      setData({ stats, health, retrieval })
    } catch (err) {
      setError(err instanceof Error ? err.message : "Khong the tai dashboard.")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadDashboard()
  }, [])

  const issueChecks = useMemo(() => {
    if (!data) return []
    return Object.entries(data.health.checks).filter(([, check]) =>
      ["error", "missing"].includes(check.status)
    )
  }, [data])

  return (
    <AppPageShell
      description="Theo doi nhanh du lieu nguoi dung, hoi thoai, message va trang thai cau hinh van hanh."
      title="Tong quan"
    >
      {loading ? (
        <StatePanel label="Dang tai du lieu quan tri..." />
      ) : error ? (
        <StatePanel
          action={<Button onClick={loadDashboard}>Thu lai</Button>}
          label={error}
          tone="destructive"
        />
      ) : data ? (
        <>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              description={`${data.stats.stats.activeUsers} tai khoan dang hoat dong`}
              icon={Users}
              label="Nguoi dung"
              value={formatNumber(data.stats.stats.totalUsers)}
            />
            <MetricCard
              description="Tong so hoi thoai da luu trong database"
              icon={MessageSquareText}
              label="Hoi thoai"
              value={formatNumber(data.stats.stats.totalConversations)}
            />
            <MetricCard
              description="Tong message user va assistant"
              icon={Activity}
              label="Messages"
              value={formatNumber(data.stats.stats.totalMessages)}
            />
            <MetricCard
              description={`${data.stats.stats.adminUsers} tai khoan admin`}
              icon={ShieldCheck}
              label="Session dang hoat dong"
              value={formatNumber(data.stats.stats.activeSessions)}
            />
          </div>

          <div className="mt-6 grid gap-5 lg:grid-cols-[1fr_0.85fr]">
            <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
              <div className="mb-5 flex items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <Activity className="h-5 w-5 text-primary" />
                  <div>
                    <h2 className="text-base font-semibold">Trang thai he thong</h2>
                    <p className="text-sm text-muted-foreground">
                      {data.stats.runtime.appEnv} | {data.stats.runtime.databasePath}
                    </p>
                  </div>
                </div>
                <Badge variant={data.health.status === "ok" ? "success" : "warning"}>
                  {data.health.status}
                </Badge>
              </div>
              <div className="grid gap-3 sm:grid-cols-3">
                <StatusItem
                  icon={Database}
                  label="Database"
                  value={data.health.checks.database.status}
                />
                <StatusItem
                  icon={Search}
                  label="Index"
                  value={data.health.checks.index.status}
                />
                <StatusItem
                  icon={CheckCircle2}
                  label="LLM"
                  value={data.health.checks.llmConfig.status}
                />
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <ConfigRow
                  label="Qdrant"
                  value={`${data.health.checks.qdrantConfig.usesCloud ? "cloud" : "local"} | ${data.health.checks.qdrantConfig.collection || "default"}`}
                />
                <ConfigRow
                  label="Retriever source"
                  value={data.retrieval.retrieverRecordSource || "sqlite"}
                />
                <ConfigRow
                  label="Query router"
                  value={data.retrieval.queryRouterEnabled ? "enabled" : "disabled"}
                />
                <ConfigRow
                  label="Reranker"
                  value={
                    data.retrieval.rerankerEnabled
                      ? `${data.retrieval.rerankerModel} | top ${data.retrieval.rerankerTopN}`
                      : "disabled"
                  }
                />
              </div>
            </section>

            <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-warning-foreground" />
                  <h2 className="text-base font-semibold">Can kiem tra</h2>
                </div>
                <Badge variant={issueChecks.length ? "warning" : "success"}>
                  {issueChecks.length} muc
                </Badge>
              </div>
              <div className="space-y-3">
                {issueChecks.length ? (
                  issueChecks.map(([key, check]) => (
                    <div
                      className="rounded-md border border-border bg-background px-3 py-3 text-sm"
                      key={key}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-medium">{key}</span>
                        <Badge variant="warning">{check.status}</Badge>
                      </div>
                      {"message" in check ? (
                        <p className="mt-2 leading-6 text-muted-foreground">
                          {check.message}
                        </p>
                      ) : null}
                    </div>
                  ))
                ) : (
                  <div className="rounded-md border border-border bg-background px-3 py-3 text-sm text-muted-foreground">
                    Khong co canh bao cau hinh.
                  </div>
                )}
              </div>
            </section>
          </div>
        </>
      ) : null}
    </AppPageShell>
  )
}

function MetricCard({
  description,
  icon: Icon,
  label,
  value
}: {
  description: string
  icon: ComponentType<{ className?: string }>
  label: string
  value: string
}) {
  return (
    <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground">{label}</p>
        <Icon className="h-4 w-4 text-primary" />
      </div>
      <p className="mt-3 text-3xl font-semibold text-foreground">{value}</p>
      <p className="mt-2 text-sm leading-6 text-muted-foreground">{description}</p>
    </section>
  )
}

function StatusItem({
  icon: Icon,
  label,
  value
}: {
  icon: ComponentType<{ className?: string }>
  label: string
  value: string
}) {
  const variant =
    value === "ok" || value === "configured" || value === "local"
      ? "success"
      : "warning"
  return (
    <div className="rounded-md border border-border bg-background p-3">
      <Icon className="h-4 w-4 text-primary" />
      <p className="mt-3 text-sm text-muted-foreground">{label}</p>
      <Badge className="mt-2" variant={variant}>
        {value}
      </Badge>
    </div>
  )
}

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-background p-3">
      <p className="text-xs uppercase text-muted-foreground">{label}</p>
      <p className="mt-2 break-words text-sm font-medium">{value}</p>
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
          {tone === "destructive" ? (
            <AlertTriangle className="h-5 w-5 text-destructive" />
          ) : (
            <RefreshCw className="h-5 w-5 text-primary" />
          )}
          <p className="text-sm font-medium">{label}</p>
        </div>
        {action}
      </div>
    </section>
  )
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("vi-VN").format(value)
}
