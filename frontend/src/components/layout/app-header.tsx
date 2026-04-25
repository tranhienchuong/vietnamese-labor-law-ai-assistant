"use client"

import Link from "next/link"
import { useEffect, useState } from "react"
import {
  BookOpenText,
  Github,
  Menu,
  Moon,
  Scale,
  Settings,
  Sun
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { APP_NAME } from "@/lib/constants"
import { cn } from "@/lib/utils"

type AppHeaderProps = {
  onMenuClick?: () => void
  className?: string
}

export function AppHeader({ onMenuClick, className }: AppHeaderProps) {
  const [isDark, setIsDark] = useState(false)

  useEffect(() => {
    const storedTheme = window.localStorage.getItem("theme")
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches
    const shouldUseDark = storedTheme ? storedTheme === "dark" : prefersDark
    document.documentElement.classList.toggle("dark", shouldUseDark)
    setIsDark(shouldUseDark)
  }, [])

  function toggleTheme() {
    const next = !isDark
    document.documentElement.classList.toggle("dark", next)
    window.localStorage.setItem("theme", next ? "dark" : "light")
    setIsDark(next)
  }

  return (
    <header
      className={cn(
        "sticky top-0 z-40 border-b border-border bg-surface/95 backdrop-blur",
        className
      )}
    >
      <div className="mx-auto flex h-16 max-w-[1600px] items-center gap-3 px-4 sm:px-6">
        <Button
          aria-label="Mở điều hướng"
          className="lg:hidden"
          onClick={onMenuClick}
          size="iconSm"
          type="button"
          variant="ghost"
        >
          <Menu className="h-4 w-4" />
        </Button>

        <Link className="flex min-w-0 items-center gap-3" href="/">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Scale className="h-5 w-5" />
          </span>
          <span className="min-w-0">
            <span className="block truncate text-sm font-semibold sm:text-base">
              {APP_NAME}
            </span>
            <span className="hidden text-xs text-muted-foreground sm:block">
              Legal RAG cho chấm dứt hợp đồng lao động
            </span>
          </span>
        </Link>

        <div className="hidden items-center gap-2 md:flex">
          <Badge variant="secondary">Beta</Badge>
          <Badge variant="success">RAG enabled</Badge>
        </div>

        <div className="ml-auto flex items-center gap-1">
          <Button
            aria-label="Mở tài liệu"
            asChild
            size="iconSm"
            title="Tài liệu"
            variant="ghost"
          >
            <Link href="/documents">
              <BookOpenText className="h-4 w-4" />
            </Link>
          </Button>
          <Button
            aria-label="Mở GitHub"
            asChild
            className="hidden sm:inline-flex"
            size="iconSm"
            title="GitHub"
            variant="ghost"
          >
            <Link href="https://github.com" target="_blank">
              <Github className="h-4 w-4" />
            </Link>
          </Button>
          <Button
            aria-label="Mở cài đặt"
            asChild
            size="iconSm"
            title="Cài đặt"
            variant="ghost"
          >
            <Link href="/settings">
              <Settings className="h-4 w-4" />
            </Link>
          </Button>
          <Button
            aria-label={isDark ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"}
            onClick={toggleTheme}
            size="iconSm"
            title="Theme"
            type="button"
            variant="ghost"
          >
            {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </header>
  )
}
