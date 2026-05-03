"use client"

import { ChevronDown, Clock3, Plus, Sparkles } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { EXAMPLE_QUESTIONS } from "@/lib/constants"
import type { ConversationSummary } from "@/lib/types"
import { cn } from "@/lib/utils"

type AppSidebarProps = {
  conversations?: ConversationSummary[]
  activeConversationId?: string | null
  onNewChat?: () => void
  onConversationSelect?: (conversationId: string) => void
  onExampleSelect?: (question: string) => void
  className?: string
}

export function AppSidebar({
  conversations = [],
  activeConversationId,
  onNewChat,
  onConversationSelect,
  onExampleSelect,
  className
}: AppSidebarProps) {
  return (
    <aside
      className={cn(
        "flex h-full w-72 shrink-0 flex-col border-r border-border bg-surface",
        className
      )}
    >
      <div className="p-4">
        <Button className="w-full justify-start" onClick={onNewChat} type="button">
          <Plus className="h-4 w-4" />
          Cuộc trò chuyện mới
        </Button>
      </div>

      <Separator />

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <section className="mb-6">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock3 className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Lịch sử hội thoại
              </h2>
            </div>
            <Badge variant="secondary">{conversations.length}</Badge>
          </div>
          <div className="space-y-1">
            {conversations.length > 0 ? (
              conversations.map((conversation) => {
                const isActive = conversation.id === activeConversationId
                return (
                  <button
                    className={cn(
                      "line-clamp-1 w-full rounded-md px-2 py-2 text-left text-sm text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                      isActive && "bg-muted text-foreground"
                    )}
                    key={conversation.id}
                    onClick={() => onConversationSelect?.(conversation.id)}
                    type="button"
                  >
                    {conversation.title}
                  </button>
                )
              })
            ) : (
              <p className="rounded-md border border-border bg-background px-3 py-2 text-sm leading-5 text-muted-foreground">
                Chưa có cuộc trò chuyện nào.
              </p>
            )}
          </div>
        </section>

        <details className="group" open>
          <summary className="flex cursor-pointer list-none items-center gap-2 rounded-md px-2 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:bg-muted">
            <Sparkles className="h-4 w-4 text-primary" />
            <span className="flex-1">Câu hỏi mẫu</span>
            <ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" />
          </summary>
          <div className="mt-2 space-y-2">
            {EXAMPLE_QUESTIONS.map((question) => (
              <button
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-left text-sm leading-5 text-foreground transition-colors hover:border-primary/40 hover:bg-primary/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                key={question}
                onClick={() => onExampleSelect?.(question)}
                type="button"
              >
                {question}
              </button>
            ))}
          </div>
        </details>
      </div>

      <div className="border-t border-border p-4" />
    </aside>
  )
}
