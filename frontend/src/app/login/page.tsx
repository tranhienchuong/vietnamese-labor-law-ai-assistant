"use client"

import Link from "next/link"
import { useSearchParams } from "next/navigation"
import { Suspense, useState } from "react"
import type { FormEvent } from "react"
import { ArrowRight, Loader2, Scale } from "lucide-react"
import { useAuth } from "@/components/auth/auth-provider"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { APP_NAME } from "@/lib/constants"

export default function LoginPage() {
  return (
    <Suspense fallback={null}>
      <LoginForm />
    </Suspense>
  )
}

function LoginForm() {
  const searchParams = useSearchParams()
  const { login } = useAuth()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError("")
    setIsSubmitting(true)

    try {
      const user = await login(email, password)
      const next = searchParams.get("next")
      window.location.assign(next || (user.role === "admin" ? "/admin" : "/chat"))
    } catch {
      setError("Email hoặc mật khẩu không đúng.")
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="grid min-h-screen bg-background text-foreground lg:grid-cols-[1fr_0.86fr]">
      <section className="flex flex-col justify-between bg-surface px-6 py-8 sm:px-10">
        <Link className="flex items-center gap-3" href="/">
          <span className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Scale className="h-5 w-5" />
          </span>
          <span className="text-sm font-semibold sm:text-base">{APP_NAME}</span>
        </Link>

        <div className="max-w-2xl py-12">
          <h1 className="text-3xl font-semibold tracking-normal sm:text-4xl">
            Đăng nhập để tiếp tục cuộc trò chuyện pháp lý của bạn
          </h1>
          <p className="mt-4 max-w-xl text-sm leading-6 text-muted-foreground sm:text-base">
            Tài khoản giúp lưu lịch sử hỏi đáp riêng cho từng người và chỉ cho
            admin truy cập khu vực quản trị nội bộ.
          </p>
        </div>

        <p className="text-xs leading-5 text-muted-foreground">
          Thông tin đăng nhập được dùng để phân quyền và lưu lịch sử hội thoại.
        </p>
      </section>

      <section className="flex items-center justify-center px-4 py-10 sm:px-6">
        <form
          className="w-full max-w-md rounded-lg border border-border bg-surface p-6 shadow-sm"
          onSubmit={handleSubmit}
        >
          <h2 className="text-xl font-semibold">Đăng nhập</h2>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            Sử dụng tài khoản đã được cấp để vào hệ thống.
          </p>

          <div className="mt-6 grid gap-4">
            <label className="grid gap-2 text-sm font-medium">
              Email
              <Input
                autoComplete="email"
                inputMode="email"
                onChange={(event) => setEmail(event.target.value)}
                placeholder="name@example.com"
                required
                type="email"
                value={email}
              />
            </label>

            <label className="grid gap-2 text-sm font-medium">
              Mật khẩu
              <Input
                autoComplete="current-password"
                onChange={(event) => setPassword(event.target.value)}
                placeholder="Nhập mật khẩu"
                required
                type="password"
                value={password}
              />
            </label>
          </div>

          {error ? (
            <div className="mt-4 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </div>
          ) : null}

          <Button className="mt-6 w-full" disabled={isSubmitting} type="submit">
            {isSubmitting ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <ArrowRight className="h-4 w-4" />
            )}
            Đăng nhập
          </Button>
        </form>
      </section>
    </main>
  )
}
