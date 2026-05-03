import { AlertTriangle, RefreshCw } from "lucide-react"
import { AppPageShell } from "@/components/layout/app-page-shell"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { RETRIEVAL_ROWS } from "@/lib/constants"
import type { RetrievalRecord } from "@/lib/types"

function statusBadge(status: RetrievalRecord["status"]) {
  if (status === "ok") return <Badge variant="success">Đạt</Badge>
  if (status === "warning") return <Badge variant="warning">Cần xem</Badge>
  return <Badge variant="destructive">Lỗi</Badge>
}

export default function AdminRetrievalPage() {
  const warnings = RETRIEVAL_ROWS.filter((row) => row.status !== "ok")

  return (
    <AppPageShell
      actions={
        <Button variant="outline">
          <RefreshCw className="h-4 w-4" />
          Làm mới
        </Button>
      }
      description="Kiểm tra nguồn được truy xuất, chunk được chọn, điểm liên quan, confidence và metadata."
      title="Trạng thái truy xuất"
    >
      {warnings.length > 0 ? (
        <div className="mb-5 flex items-start gap-3 rounded-md border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning-foreground">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          Có {warnings.length} truy vấn cần kiểm tra vì nguồn hoặc confidence chưa đủ mạnh.
        </div>
      ) : null}

      <div className="overflow-hidden rounded-lg border border-border bg-surface shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[1050px] text-left text-sm">
            <thead className="bg-muted text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-4 py-3 font-semibold">ID</th>
                <th className="px-4 py-3 font-semibold">Truy vấn</th>
                <th className="px-4 py-3 font-semibold">Nguồn</th>
                <th className="px-4 py-3 font-semibold">Chunk</th>
                <th className="px-4 py-3 font-semibold">Relevance</th>
                <th className="px-4 py-3 font-semibold">Confidence</th>
                <th className="px-4 py-3 font-semibold">Metadata</th>
                <th className="px-4 py-3 font-semibold">Trạng thái</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {RETRIEVAL_ROWS.map((row) => (
                <tr className="align-top hover:bg-muted/50" key={row.id}>
                  <td className="px-4 py-4 font-medium">{row.id}</td>
                  <td className="max-w-sm px-4 py-4 leading-6">{row.query}</td>
                  <td className="px-4 py-4">{row.sources}</td>
                  <td className="px-4 py-4 font-mono text-xs">{row.selectedChunk}</td>
                  <td className="px-4 py-4">{row.relevanceScore}</td>
                  <td className="px-4 py-4">{row.confidence}</td>
                  <td className="max-w-xs px-4 py-4 leading-6 text-muted-foreground">
                    {row.metadata}
                  </td>
                  <td className="px-4 py-4">{statusBadge(row.status)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </AppPageShell>
  )
}
