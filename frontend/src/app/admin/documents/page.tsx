import { Database, Search, Upload } from "lucide-react"
import { AppPageShell } from "@/components/layout/app-page-shell"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { DOCUMENTS } from "@/lib/constants"
import type { DocumentRecord } from "@/lib/types"

function statusBadge(status: DocumentRecord["status"]) {
  const map = {
    completed: { label: "Đã index", variant: "success" as const },
    processing: { label: "Đang xử lý", variant: "warning" as const },
    pending: { label: "Chờ index", variant: "secondary" as const },
    failed: { label: "Lỗi", variant: "destructive" as const }
  }
  const item = map[status]
  return <Badge variant={item.variant}>{item.label}</Badge>
}

export default function AdminDocumentsPage() {
  return (
    <AppPageShell
      actions={
        <>
          <Button variant="outline">
            <Database className="h-4 w-4" />
            Re-index
          </Button>
          <Button>
            <Upload className="h-4 w-4" />
            Thêm tài liệu
          </Button>
        </>
      }
      description="Quản lý corpus pháp lý, trạng thái index và metadata dùng cho truy xuất nguồn."
      title="Tài liệu nguồn"
    >
      <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="relative max-w-md flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input className="pl-9" placeholder="Tìm theo tên văn bản..." />
        </div>
        <Badge variant="secondary">{DOCUMENTS.length} tài liệu</Badge>
      </div>

      <div className="overflow-hidden rounded-lg border border-border bg-surface shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[780px] text-left text-sm">
            <thead className="bg-muted text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-4 py-3 font-semibold">Tên tài liệu</th>
                <th className="px-4 py-3 font-semibold">Loại</th>
                <th className="px-4 py-3 font-semibold">Chunks</th>
                <th className="px-4 py-3 font-semibold">Trạng thái</th>
                <th className="px-4 py-3 font-semibold">Cập nhật</th>
                <th className="px-4 py-3 font-semibold">Hành động</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {DOCUMENTS.map((document) => (
                <tr className="hover:bg-muted/50" key={document.id}>
                  <td className="px-4 py-4 font-medium">{document.name}</td>
                  <td className="px-4 py-4 text-muted-foreground">{document.type}</td>
                  <td className="px-4 py-4">{document.chunks}</td>
                  <td className="px-4 py-4">{statusBadge(document.status)}</td>
                  <td className="px-4 py-4 text-muted-foreground">
                    {document.updatedAt}
                  </td>
                  <td className="px-4 py-4">
                    <Button size="sm" variant="outline">
                      Cập nhật
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </AppPageShell>
  )
}
