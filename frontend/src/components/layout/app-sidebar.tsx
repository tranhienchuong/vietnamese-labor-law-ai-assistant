"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  BarChart3,
  FileText,
  MessageSquarePlus,
  Plus,
  Settings,
  Sparkles
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { EXAMPLE_QUESTIONS, RECENT_CHATS } from "@/lib/constants"
import { cn } from "@/lib/utils"

type AppSidebarProps = {
  onNewChat?: () => void
  onExampleSelect?: (question: string) => void
  className?: string
}

const navItems = [
  { href: "/chat", label: "Chat", icon: MessageSquarePlus },
  { href: "/documents", label: "Documents", icon: FileText },
  { href: "/evaluation", label: "Evaluation", icon: BarChart3 },
  { href: "/settings", label: "Settings", icon: Settings }
]

export function AppSidebar({
  onNewChat,
  onExampleSelect,
  className
}: AppSidebarProps) {
  const pathname = usePathname()

  return (
    <aside
      className={cn(
        "flex h-full w-72 shrink-0 flex-col border-r border-border bg-surface",
        className
      )}
    >
      <div className="p-4">
        {onNewChat ? (
          <Button className="w-full justify-start" onClick={onNewChat} type="button">
            <Plus className="h-4 w-4" />
            New chat
          </Button>
        ) : (
          <Button asChild className="w-full justify-start">
            <Link href="/chat">
              <Plus className="h-4 w-4" />
              New chat
            </Link>
          </Button>
        )}
      </div>

      <nav className="space-y-1 px-3">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = pathname === item.href
          return (
            <Button
              asChild
              className={cn(
                "w-full justify-start",
                isActive && "bg-muted text-foreground"
              )}
              key={item.href}
              variant="ghost"
            >
              <Link href={item.href}>
                <Icon className="h-4 w-4" />
                {item.label}
              </Link>
            </Button>
          )
        })}
      </nav>

      <Separator className="my-4" />

      <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-4">
        <div className="mb-6">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Gần đây
            </h2>
            <Badge variant="secondary">{RECENT_CHATS.length}</Badge>
          </div>
          <div className="space-y-1">
            {RECENT_CHATS.map((chat) => (
              <button
                className="line-clamp-1 w-full rounded-md px-2 py-2 text-left text-sm text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                key={chat}
                type="button"
              >
                {chat}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="mb-2 flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Câu hỏi mẫu
            </h2>
          </div>
          <div className="space-y-2">
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
        </div>
      </div>

      <div className="border-t border-border p-4 text-xs text-muted-foreground">
        Version 0.1.0 · Demo frontend
      </div>
    </aside>
  )
}
