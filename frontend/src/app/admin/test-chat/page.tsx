import { Bot, Play, UserRound } from "lucide-react"
import { AppPageShell } from "@/components/layout/app-page-shell"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export default function AdminTestChatPage() {
  return (
    <AppPageShell
      actions={
        <Button>
          <Play className="h-4 w-4" />
          Chạy thử
        </Button>
      }
      description="Không gian thử nghiệm hội thoại có thể hiển thị prompt, output, citation và metadata phục vụ kiểm thử."
      title="Hội thoại thử nghiệm"
    >
      <div className="grid gap-5 lg:grid-cols-[1fr_0.8fr]">
        <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground">
                <UserRound className="h-4 w-4" />
              </span>
              <div className="rounded-lg border border-border bg-background px-4 py-3 text-sm leading-6">
                Người lao động ký hợp đồng không xác định thời hạn muốn nghỉ việc
                thì phải báo trước bao lâu?
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-muted text-primary">
                <Bot className="h-4 w-4" />
              </span>
              <div className="rounded-lg border border-border bg-background px-4 py-3 text-sm leading-6">
                <p className="font-semibold">Kết luận ngắn</p>
                <p className="mt-2">
                  Trường hợp hợp đồng không xác định thời hạn, người lao động
                  thường phải báo trước ít nhất 45 ngày.
                </p>
                <p className="mt-4 font-semibold">Căn cứ pháp lý</p>
                <p className="mt-2">Bộ luật Lao động 2019, Điều 35.</p>
              </div>
            </div>
          </div>
        </section>

        <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
          <h2 className="text-base font-semibold">Thông tin đánh giá</h2>
          <div className="mt-4 grid gap-3 text-sm">
            <MetaRow label="Citation" value="Điều 35" />
            <MetaRow label="Confidence" value="0.84" />
            <MetaRow label="Chunk" value="BLLD2019-035-01" />
            <MetaRow label="Benchmark" value="Q-001" />
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <Badge variant="success">Có căn cứ</Badge>
            <Badge variant="secondary">Không hallucination</Badge>
          </div>
        </section>
      </div>
    </AppPageShell>
  )
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded-md border border-border bg-background px-3 py-2">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  )
}
