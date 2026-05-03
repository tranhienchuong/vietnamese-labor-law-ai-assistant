"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  ClipboardCheck,
  FileText,
  LayoutDashboard,
  MessagesSquare,
  ScrollText,
  SearchCheck,
  Settings
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"

const adminNavItems = [
  { href: "/admin", label: "Tổng quan", icon: LayoutDashboard },
  { href: "/admin/test-chat", label: "Hội thoại thử nghiệm", icon: MessagesSquare },
  { href: "/admin/documents", label: "Tài liệu nguồn", icon: FileText },
  { href: "/admin/evaluation", label: "Đánh giá", icon: ClipboardCheck },
  { href: "/admin/retrieval", label: "Trạng thái truy xuất", icon: SearchCheck },
  { href: "/admin/settings", label: "Cấu hình hệ thống", icon: Settings },
  { href: "/admin/logs", label: "Nhật ký", icon: ScrollText }
]

type AdminSidebarProps = {
  className?: string
}

export function AdminSidebar({ className }: AdminSidebarProps) {
  const pathname = usePathname()

  return (
    <aside
      className={cn(
        "flex h-full w-72 shrink-0 flex-col border-r border-border bg-surface",
        className
      )}
    >
      <div className="px-4 py-5">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Admin / Evaluator
        </p>
        <h2 className="mt-1 text-base font-semibold">Khu quản trị</h2>
      </div>

      <Separator />

      <nav className="space-y-1 px-3 py-4">
        {adminNavItems.map((item) => {
          const Icon = item.icon
          const isActive =
            pathname === item.href ||
            (item.href !== "/admin" && pathname.startsWith(`${item.href}/`))

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

      <div className="mt-auto border-t border-border p-4 text-xs leading-5 text-muted-foreground">
        Các module kỹ thuật được tách khỏi giao diện người dùng.
      </div>
    </aside>
  )
}
