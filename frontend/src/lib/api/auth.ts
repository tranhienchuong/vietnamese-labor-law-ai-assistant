import { apiFetchJson } from "@/lib/api/client"
import type { CurrentUser, CurrentUserResponse, LoginResponse } from "@/lib/types"

export async function getCurrentUser(): Promise<CurrentUser | null> {
  const payload = await apiFetchJson<CurrentUserResponse>("/api/auth/me")
  return payload.user ?? null
}

export async function login(email: string, password: string): Promise<CurrentUser> {
  const payload = await apiFetchJson<LoginResponse>("/api/auth/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ email, password })
  })
  return payload.user
}

export async function logout(): Promise<void> {
  await apiFetchJson<{ ok: boolean }>("/api/auth/logout", {
    method: "POST"
  })
}
