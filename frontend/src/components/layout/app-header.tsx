"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { useEffect, useState } from "react"
import {
  LogIn,
  LogOut,
  Menu,
  MessageSquareText,
  Moon,
  Scale,
  Settings,
  ShieldCheck,
  Sun
} from "lucide-react"
import { useAuth } from "@/components/auth/auth-provider"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { APP_NAME } from "@/lib/constants"
import { cn } from "@/lib/utils"

type AppHeaderProps = {
  onMenuClick?: () => void
  className?: string
  variant?: "user" | "admin"
}

export function AppHeader({
  onMenuClick,
  className,
  variant = "user"
}: AppHeaderProps) {
  const [isDark, setIsDark] = useState(false)
  const isAdmin = variant === "admin"
  const router = useRouter()
  const { user, isLoading, logout } = useAuth()

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

  async function handleLogout() {
    await logout()
    router.push("/login")
  }

  return (
    <header
      className={cn(
        "sticky top-0 z-40 border-b border-border bg-surface/95 backdrop-blur",
        className
      )}
    >
      <div className="mx-auto flex h-16 max-w-[1600px] items-center gap-3 px-4 sm:px-6">
        {onMenuClick ? (
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
        ) : null}

        <Link className="flex min-w-0 items-center gap-3" href={isAdmin ? "/admin" : "/"}>
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Scale className="h-5 w-5" />
          </span>
          <span className="min-w-0">
            <span className="block truncate text-sm font-semibold sm:text-base">
              {APP_NAME}
            </span>
            <span className="hidden text-xs text-muted-foreground sm:block">
              {isAdmin
                ? "Không gian quản trị và đánh giá nội bộ"
                : "Hỏi đáp về luật lao động Việt Nam"}
            </span>
          </span>
        </Link>

        {isAdmin ? (
          <div className="hidden items-center gap-2 md:flex">
            <Badge variant="outline">Quản trị</Badge>
            <Badge variant="secondary">Nội bộ</Badge>
          </div>
        ) : null}

        <div className="ml-auto flex items-center gap-1">
          {!isLoading && user?.role === "admin" && !isAdmin ? (
            <Button asChild className="hidden sm:inline-flex" size="sm" variant="outline">
              <Link href="/admin">
                <ShieldCheck className="h-3.5 w-3.5" />
                Quản trị
              </Link>
            </Button>
          ) : null}
          {isAdmin ? (
            <>
              <Button
                aria-label="Mở chat người dùng"
                asChild
                className="hidden sm:inline-flex"
                size="iconSm"
                title="Chat người dùng"
                variant="ghost"
              >
                <Link href="/chat">
                  <MessageSquareText className="h-4 w-4" />
                </Link>
              </Button>
              <Button
                aria-label="Mở cấu hình hệ thống"
                asChild
                size="iconSm"
                title="Cấu hình"
                variant="ghost"
              >
                <Link href="/admin/settings">
                  <Settings className="h-4 w-4" />
                </Link>
              </Button>
              <ShieldCheck className="hidden h-4 w-4 text-success sm:block" />
            </>
          ) : null}
          {!isLoading ? (
            user ? (
              <>
                <span className="hidden max-w-40 truncate px-2 text-sm text-muted-foreground md:block">
                  {user.name}
                </span>
                <Button
                  aria-label="Đăng xuất"
                  onClick={handleLogout}
                  size="iconSm"
                  title="Đăng xuất"
                  type="button"
                  variant="ghost"
                >
                  <LogOut className="h-4 w-4" />
                </Button>
              </>
            ) : (
              <Button asChild size="sm" variant="outline">
                <Link href="/login">
                  <LogIn className="h-3.5 w-3.5" />
                  Đăng nhập
                </Link>
              </Button>
            )
          ) : null}
          <Button
            aria-label={isDark ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"}
            onClick={toggleTheme}
            size="iconSm"
            title="Đổi giao diện"
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
