import { Bell, Braces, Languages, Moon, SlidersHorizontal } from "lucide-react"
import type { ComponentType } from "react"
import { AppPageShell } from "@/components/layout/app-page-shell"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

const modelOptions = ["qwen3:4b", "qwen/qwen3-32b", "gpt-4.1-mini"]

export default function SettingsPage() {
  return (
    <AppPageShell
      actions={<Button>Save settings</Button>}
      description="Cấu hình chế độ trả lời, citation, streaming và ngôn ngữ cho giao diện."
      title="Settings"
    >
      <div className="grid gap-5 lg:grid-cols-[1fr_0.75fr]">
        <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
          <div className="mb-5 flex items-center gap-3">
            <SlidersHorizontal className="h-5 w-5 text-primary" />
            <div>
              <h2 className="text-base font-semibold">Model và chế độ trả lời</h2>
              <p className="text-sm text-muted-foreground">
                Đồng bộ với backend qua biến môi trường hoặc API settings.
              </p>
            </div>
          </div>

          <div className="grid gap-5">
            <label className="grid gap-2 text-sm font-medium">
              Model
              <select className="h-10 rounded-md border border-input bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring">
                {modelOptions.map((model) => (
                  <option key={model}>{model}</option>
                ))}
              </select>
            </label>

            <label className="grid gap-2 text-sm font-medium">
              Chế độ trả lời
              <select className="h-10 rounded-md border border-input bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring">
                <option>Có căn cứ luật</option>
                <option>Ngắn gọn</option>
                <option>Chi tiết</option>
              </select>
            </label>

            <div className="grid gap-3 sm:grid-cols-2">
              <ToggleRow icon={Braces} label="Citation" value="Bật" />
              <ToggleRow icon={Bell} label="Streaming" value="Bật" />
              <ToggleRow icon={Moon} label="Dark mode" value="Theo thiết bị" />
              <ToggleRow icon={Languages} label="Language" value="Vietnamese" />
            </div>
          </div>
        </section>

        <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
          <h2 className="text-base font-semibold">Guardrails</h2>
          <div className="mt-4 space-y-3 text-sm leading-6 text-muted-foreground">
            <div className="rounded-md border border-border bg-background p-3">
              Chỉ trả lời trong phạm vi dữ liệu pháp lý đã index.
            </div>
            <div className="rounded-md border border-border bg-background p-3">
              Yêu cầu bổ sung dữ kiện khi câu hỏi chưa đủ thông tin.
            </div>
            <div className="rounded-md border border-border bg-background p-3">
              Không expose API key hoặc secret qua biến `NEXT_PUBLIC_*`.
            </div>
          </div>
        </section>
      </div>
    </AppPageShell>
  )
}

function ToggleRow({
  icon: Icon,
  label,
  value
}: {
  icon: ComponentType<{ className?: string }>
  label: string
  value: string
}) {
  return (
    <div className="flex items-center justify-between rounded-md border border-border bg-background px-3 py-3">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Icon className="h-4 w-4 text-primary" />
        {label}
      </div>
      <Badge variant="secondary">{value}</Badge>
    </div>
  )
}
