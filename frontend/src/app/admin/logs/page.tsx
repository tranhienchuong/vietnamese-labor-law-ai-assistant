import { AlertCircle, Info, TriangleAlert } from "lucide-react"
import { AppPageShell } from "@/components/layout/app-page-shell"
import { Badge } from "@/components/ui/badge"
import { SYSTEM_LOGS } from "@/lib/constants"
import type { SystemLogRecord } from "@/lib/types"

function logIcon(level: SystemLogRecord["level"]) {
  if (level === "error") return AlertCircle
  if (level === "warning") return TriangleAlert
  return Info
}

function logBadge(level: SystemLogRecord["level"]) {
  if (level === "error") return <Badge variant="destructive">Lỗi</Badge>
  if (level === "warning") return <Badge variant="warning">Cảnh báo</Badge>
  return <Badge variant="secondary">Thông tin</Badge>
}

export default function AdminLogsPage() {
  return (
    <AppPageShell
      description="Theo dõi nhật ký vận hành, lỗi truy xuất và các sự kiện liên quan đến index dữ liệu."
      title="Nhật ký"
    >
      <div className="space-y-3">
        {SYSTEM_LOGS.map((log) => {
          const Icon = logIcon(log.level)
          return (
            <article
              className="rounded-lg border border-border bg-surface p-4 shadow-sm"
              key={log.id}
            >
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="flex gap-3">
                  <span className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-muted text-primary">
                    <Icon className="h-4 w-4" />
                  </span>
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="text-sm font-semibold">{log.id}</h2>
                      {logBadge(log.level)}
                      <Badge variant="outline">{log.area}</Badge>
                    </div>
                    <p className="mt-2 text-sm leading-6 text-muted-foreground">
                      {log.message}
                    </p>
                  </div>
                </div>
                <span className="text-xs text-muted-foreground">{log.createdAt}</span>
              </div>
            </article>
          )
        })}
      </div>
    </AppPageShell>
  )
}
