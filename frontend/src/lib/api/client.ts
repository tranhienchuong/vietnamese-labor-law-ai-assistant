export class ApiError extends Error {
  status: number
  body?: unknown

  constructor(status: number, message: string, body?: unknown) {
    super(message)
    this.name = "ApiError"
    this.status = status
    this.body = body
  }
}

async function parseBody(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") || ""
  if (contentType.includes("application/json")) {
    return response.json().catch(() => ({}))
  }
  return response.text().catch(() => "")
}

function errorMessage(status: number, body: unknown) {
  if (body && typeof body === "object") {
    const payload = body as { error?: unknown; message?: unknown; detail?: unknown }
    if (typeof payload.error === "string") return payload.error
    if (typeof payload.message === "string") return payload.message
    if (typeof payload.detail === "string") return payload.detail
  }
  if (typeof body === "string" && body.trim()) return body
  return `Request failed with status ${status}`
}

function withNoStore(init?: RequestInit): RequestInit {
  return {
    ...init,
    cache: init?.cache ?? "no-store"
  }
}

export async function apiFetchJson<T>(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(input, withNoStore(init))
  const body = await parseBody(response)

  if (!response.ok) {
    throw new ApiError(response.status, errorMessage(response.status, body), body)
  }

  return body as T
}

export async function apiFetchText(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<string> {
  const response = await fetch(input, withNoStore(init))
  const body = await parseBody(response)

  if (!response.ok) {
    throw new ApiError(response.status, errorMessage(response.status, body), body)
  }

  return typeof body === "string" ? body : JSON.stringify(body)
}
