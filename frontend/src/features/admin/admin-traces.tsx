"use client"

import {
  AlertTriangle,
  Clock3,
  FileSearch,
  Loader2,
  RefreshCw,
  Search,
  type LucideIcon
} from "lucide-react"
import type { ReactNode } from "react"
import { useEffect, useState } from "react"
import { AppPageShell } from "@/components/layout/app-page-shell"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getAdminTrace, getAdminTraces } from "@/lib/api/admin"
import type { AdminTraceDetail, AdminTraceSummary } from "@/lib/types"

type DetailState = {
  traceId: string
  trace: AdminTraceDetail | null
  error: string
  loading: boolean
}

export function AdminTraces() {
  const [traces, setTraces] = useState<AdminTraceSummary[]>([])
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(true)
  const [detail, setDetail] = useState<DetailState | null>(null)

  async function loadTraces() {
    setLoading(true)
    setError("")
    try {
      const params = new URLSearchParams({ limit: "50" })
      const response = await getAdminTraces(params)
      setTraces(response.traces)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Khong the tai trace.")
    } finally {
      setLoading(false)
    }
  }

  async function openTrace(traceId: string) {
    if (detail?.traceId === traceId && detail.trace) {
      setDetail(null)
      return
    }

    setDetail({ traceId, trace: null, error: "", loading: true })
    try {
      const response = await getAdminTrace(traceId)
      setDetail({ traceId, trace: response.trace, error: "", loading: false })
    } catch (err) {
      setDetail({
        traceId,
        trace: null,
        error: err instanceof Error ? err.message : "Khong the tai chi tiet trace.",
        loading: false
      })
    }
  }

  useEffect(() => {
    void loadTraces()
  }, [])

  return (
    <AppPageShell
      actions={
        <Button onClick={loadTraces} variant="outline">
          <RefreshCw className="h-4 w-4" />
          Tai lai
        </Button>
      }
      description="Theo doi query routing, context duoc chon, citations, latency va loi cua cac luot chat gan day."
      title="Retrieval traces"
    >
      {loading ? (
        <StatePanel label="Dang tai traces..." />
      ) : error ? (
        <StatePanel
          action={<Button onClick={loadTraces}>Thu lai</Button>}
          label={error}
          tone="destructive"
        />
      ) : traces.length ? (
        <div className="space-y-3">
          {traces.map((trace) => (
            <article
              className="rounded-lg border border-border bg-surface shadow-sm"
              key={trace.id}
            >
              <button
                className="w-full px-4 py-4 text-left"
                onClick={() => void openTrace(trace.id)}
                type="button"
              >
                <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant={trace.error ? "destructive" : "outline"}>
                        {trace.error ? "error" : "ok"}
                      </Badge>
                      {trace.insufficientContext ? (
                        <Badge variant="warning">insufficient</Badge>
                      ) : null}
                      {trace.retrieveOnly ? (
                        <Badge variant="secondary">retrieve only</Badge>
                      ) : null}
                      <span className="text-xs text-muted-foreground">
                        {formatDate(trace.createdAt)}
                      </span>
                    </div>
                    <p className="mt-3 line-clamp-2 text-sm font-medium leading-6">
                      {trace.question}
                    </p>
                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
                      <span>{trace.provider || "unknown"} / {trace.model || "default"}</span>
                      <span>{trace.selectedContextCount} contexts</span>
                      <span>{trace.citationCount} citations</span>
                      <span>{trace.latencyMs ?? 0} ms</span>
                    </div>
                  </div>
                  <div className="grid min-w-48 grid-cols-3 gap-2 text-center text-xs">
                    <Metric label="retrieve" value={formatMs(trace.retrievalLatencyMs)} />
                    <Metric label="generate" value={formatMs(trace.generationLatencyMs)} />
                    <Metric label="total" value={formatMs(trace.latencyMs)} />
                  </div>
                </div>
              </button>

              {detail?.traceId === trace.id ? (
                <TraceDetailPanel detail={detail} />
              ) : null}
            </article>
          ))}
        </div>
      ) : (
        <StatePanel label="Chua co trace nao." />
      )}
    </AppPageShell>
  )
}

function TraceDetailPanel({ detail }: { detail: DetailState }) {
  if (detail.loading) {
    return (
      <div className="border-t border-border px-4 py-4 text-sm text-muted-foreground">
        Dang tai chi tiet...
      </div>
    )
  }
  if (detail.error) {
    return (
      <div className="border-t border-border px-4 py-4 text-sm text-destructive">
        {detail.error}
      </div>
    )
  }
  if (!detail.trace) return null

  const trace = detail.trace
  return (
    <div className="space-y-4 border-t border-border px-4 py-4">
      <section className="grid gap-3 md:grid-cols-3">
        <InfoTile icon={Clock3} label="Latency" value={formatMs(trace.latencyMs)} />
        <InfoTile
          icon={Search}
          label="Contexts"
          value={String(trace.selectedContextCount)}
        />
        <InfoTile
          icon={FileSearch}
          label="Request ID"
          value={trace.requestId || "none"}
        />
      </section>

      <JsonBlock label="Intent" value={trace.intent} />
      <JsonBlock label="Retrieved hits" value={trace.retrievedHits} />
      <JsonBlock label="Selected contexts" value={trace.selectedContexts} />
      <JsonBlock label="Citations" value={trace.citations} />
      {trace.error ? <JsonBlock label="Error" value={{ error: trace.error }} /> : null}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-background px-2 py-2">
      <p className="font-semibold text-foreground">{value}</p>
      <p className="mt-1 text-muted-foreground">{label}</p>
    </div>
  )
}

function InfoTile({
  icon: Icon,
  label,
  value
}: {
  icon: LucideIcon
  label: string
  value: string
}) {
  return (
    <div className="rounded-md border border-border bg-background p-3">
      <Icon className="h-4 w-4 text-primary" />
      <p className="mt-2 text-xs uppercase text-muted-foreground">{label}</p>
      <p className="mt-1 break-words text-sm font-medium">{value}</p>
    </div>
  )
}

function JsonBlock({ label, value }: { label: string; value: unknown }) {
  return (
    <section>
      <h3 className="mb-2 text-sm font-semibold">{label}</h3>
      <pre className="max-h-96 overflow-auto rounded-md border border-border bg-background p-3 text-xs leading-5 text-muted-foreground">
        {JSON.stringify(value, null, 2)}
      </pre>
    </section>
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
            <Loader2 className="h-5 w-5 text-primary" />
          )}
          <p className="text-sm font-medium">{label}</p>
        </div>
        {action}
      </div>
    </section>
  )
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("vi-VN", {
    dateStyle: "short",
    timeStyle: "medium"
  }).format(new Date(value))
}

function formatMs(value?: number | null) {
  return typeof value === "number" ? `${value} ms` : "-"
}
