import { NextRequest, NextResponse } from "next/server"
import { AUTH_COOKIE_NAME, backendBaseUrl } from "@/lib/server-api"

async function getRole(token: string) {
  if (!token) return null

  try {
    const response = await fetch(`${backendBaseUrl()}/auth/me`, {
      headers: {
        Authorization: `Bearer ${token}`
      },
      cache: "no-store"
    })
    if (!response.ok) return null

    const payload = (await response.json()) as { user?: { role?: string } }
    return payload.user?.role ?? null
  } catch {
    return null
  }
}

function loginRedirect(request: NextRequest) {
  const url = request.nextUrl.clone()
  const loginUrl = new URL("/login", request.url)
  loginUrl.searchParams.set("next", `${url.pathname}${url.search}`)
  return NextResponse.redirect(loginUrl)
}

export async function middleware(request: NextRequest) {
  const pathname = request.nextUrl.pathname
  const token = request.cookies.get(AUTH_COOKIE_NAME)?.value ?? ""
  const role = await getRole(token)

  if (pathname.startsWith("/login") && role) {
    return NextResponse.redirect(new URL(role === "admin" ? "/admin" : "/chat", request.url))
  }

  if (pathname.startsWith("/chat") && !role) {
    return loginRedirect(request)
  }

  if (pathname.startsWith("/admin")) {
    if (!role) {
      return loginRedirect(request)
    }
    if (role !== "admin") {
      return NextResponse.redirect(new URL("/chat", request.url))
    }
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/chat/:path*", "/admin/:path*", "/login"]
}
