import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Database,
  MessageSquareText
} from "lucide-react"
import type { ComponentType } from "react"
import { AppPageShell } from "@/components/layout/app-page-shell"
import { Badge } from "@/components/ui/badge"
import { ADMIN_METRICS, SYSTEM_LOGS } from "@/lib/constants"
import type { AdminMetric } from "@/lib/types"

function metricClass(tone: AdminMetric["tone"]) {
  if (tone === "success") return "text-success"
  if (tone === "warning") return "text-warning-foreground"
  if (tone === "destructive") return "text-destructive"
  return "text-foreground"
}

export default function AdminDashboardPage() {
  const recentWarnings = SYSTEM_LOGS.filter((log) => log.level !== "info")

  return (
    <AppPageShell
      description="Theo dõi nhanh dữ liệu, chất lượng câu trả lời, citation và các lỗi truy xuất gần đây."
      title="Tổng quan"
    >
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {ADMIN_METRICS.map((metric) => (
          <section
            className="rounded-lg border border-border bg-surface p-5 shadow-sm"
            key={metric.label}
          >
            <p className="text-sm text-muted-foreground">{metric.label}</p>
            <p className={`mt-3 text-3xl font-semibold ${metricClass(metric.tone)}`}>
              {metric.value}
            </p>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              {metric.description}
            </p>
          </section>
        ))}
      </div>

      <div className="mt-6 grid gap-5 lg:grid-cols-[1fr_0.85fr]">
        <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
          <div className="mb-5 flex items-center gap-3">
            <Activity className="h-5 w-5 text-primary" />
            <div>
              <h2 className="text-base font-semibold">Trạng thái hệ thống</h2>
              <p className="text-sm text-muted-foreground">
                Các chỉ báo vận hành chính của trợ lý pháp lý.
              </p>
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <StatusItem icon={Database} label="Corpus" value="Sẵn sàng" />
            <StatusItem icon={MessageSquareText} label="Chat API" value="Hoạt động" />
            <StatusItem icon={CheckCircle2} label="Citation" value="Theo dõi" />
          </div>
        </section>

        <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-warning-foreground" />
              <h2 className="text-base font-semibold">Cần kiểm tra</h2>
            </div>
            <Badge variant="warning">{recentWarnings.length} mục</Badge>
          </div>
          <div className="space-y-3">
            {recentWarnings.map((log) => (
              <div
                className="rounded-md border border-border bg-background px-3 py-3 text-sm"
                key={log.id}
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium">{log.area}</span>
                  <span className="text-xs text-muted-foreground">{log.createdAt}</span>
                </div>
                <p className="mt-2 leading-6 text-muted-foreground">{log.message}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </AppPageShell>
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
  return (
    <div className="rounded-md border border-border bg-background p-3">
      <Icon className="h-4 w-4 text-primary" />
      <p className="mt-3 text-sm text-muted-foreground">{label}</p>
      <p className="mt-1 text-sm font-semibold">{value}</p>
    </div>
  )
}
