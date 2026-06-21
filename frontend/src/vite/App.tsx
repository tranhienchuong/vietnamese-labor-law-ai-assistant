import {
  ArrowRight,
  Copy,
  History,
  Loader2,
  LogOut,
  Plus,
  Scale,
  SendHorizontal
} from "lucide-react"
import { ChangeEvent, FormEvent, useCallback, useEffect, useRef, useState } from "react"
import { Link, Navigate, Route, Routes, useNavigate } from "react-router-dom"
import {
  ApiError,
  AdminStats,
  AdminTrace,
  ConversationSummary,
  getAdminHealth,
  getAdminRetrievalConfig,
  getAdminStats,
  getAdminTraces,
  getConversation,
  getCurrentUser,
  listConversations,
  normalizeChatCitations,
  sendChatQuestion,
  type ChatCitations,
  type ChatMessage
} from "./backend"
import { useAuth } from "./auth"
import { LEGAL_DISCLAIMER, missingSupabaseConfig, PRODUCT_NAME } from "./config"
import assistantAvatar from "./assistant-avatar.png"

const suggestedPrompts = [
  "Người lao động được định nghĩa như thế nào?",
  "Khi nào người lao động được đơn phương chấm dứt hợp đồng?",
  "Tuổi nghỉ hưu được xác định theo Nghị định 135/2020/NĐ-CP như thế nào?"
]

const insufficientContextMessage =
  "Không tìm thấy đủ căn cứ pháp lý trong nguồn đã lập chỉ mục để trả lời chắc chắn."

export function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/signin" element={<AuthPage mode="signin" />} />
      <Route path="/register" element={<AuthPage mode="register" />} />
      <Route path="/auth/callback" element={<AuthCallbackPage />} />
      <Route
        path="/app"
        element={
          <ProtectedRoute>
            <ResearchApp />
          </ProtectedRoute>
        }
      />
      <Route
        path="/account"
        element={
          <ProtectedRoute>
            <AccountPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin"
        element={
          <ProtectedRoute>
            <AdminPage />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

function LandingPage() {
  return (
    <div className="min-h-screen overflow-hidden bg-background text-foreground">
      <MarketingHeader />
      <main>
        <section className="relative border-b border-border hero-gradient-bg">
          <div className="hero-grid-overlay" />
          <div className="relative mx-auto grid min-h-[calc(100vh-8rem)] max-w-6xl items-center gap-12 px-4 py-20 sm:px-6 lg:grid-cols-[0.9fr_1.1fr] lg:px-8 z-10">
            <div className="text-center lg:text-left">
              <h1 className="text-4xl font-semibold tracking-normal text-foreground sm:text-5xl">
                Hỏi đáp luật lao động Việt Nam
              </h1>
              <p className="mt-5 max-w-2xl text-base leading-8 text-muted-foreground">
                Đặt câu hỏi về hợp đồng lao động, tiền lương, thời giờ làm việc,
                nghỉ phép, chấm dứt hợp đồng và tuổi nghỉ hưu. Câu trả lời được
                hỗ trợ bằng căn cứ pháp lý truy xuất từ hệ thống.
              </p>
              <div className="mt-8 flex justify-center lg:justify-start">
                <Link className="button-primary" to="/signin">
                  Bắt đầu
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </div>
            </div>
            <LegalHeroScene />
          </div>

          {/* Bottom wave curve */}
          <div className="absolute bottom-0 left-0 right-0 w-full overflow-hidden leading-[0] pointer-events-none z-0">
            <svg className="relative block w-full h-[60px] md:h-[80px]" viewBox="0 0 1200 120" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M0,60 C400,110 800,30 1200,80 L1200,120 L0,120 Z" fill="#ffffff" opacity="0.45"></path>
              <path d="M0,80 C300,120 600,40 1200,90 L1200,120 L0,120 Z" fill="#ffffff"></path>
            </svg>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  )
}

function LegalHeroScene() {
  return (
    <div className="chatbot-hero-container mx-auto" aria-hidden="true">
      {/* Sparkles */}
      <div className="floating-element animate-twinkle-1" style={{ top: "10%", left: "8%" }}>
        <svg className="w-5 h-5 floating-sparkle" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 0l3 9 9 3-9 3-3 9-3-9-9-3 9-3z" />
        </svg>
      </div>
      <div className="floating-element animate-twinkle-2" style={{ top: "75%", right: "2%" }}>
        <svg className="w-4 h-4 floating-sparkle" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 0l3 9 9 3-9 3-3 9-3-9-9-3 9-3z" />
        </svg>
      </div>
      <div className="floating-element animate-twinkle-3" style={{ top: "45%", right: "90%" }}>
        <svg className="w-6 h-6 floating-sparkle" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 0l3 9 9 3-9 3-3 9-3-9-9-3 9-3z" />
        </svg>
      </div>

      {/* Floating Scale of Justice */}
      <div className="floating-element floating-scale-card animate-sway">
        <div className="flex h-11 w-11 items-center justify-center rounded-full bg-amber-500/10 text-amber-500">
          <Scale className="h-6 w-6" />
        </div>
      </div>

      {/* Floating Search Glass */}
      <div className="floating-element floating-search-card animate-pulse-glow-1">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-500/10 text-blue-600">
          <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>

      {/* Floating Shield Check */}
      <div className="floating-element floating-shield-card animate-pulse-glow-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-500">
          <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
        </div>
      </div>

      {/* Main Chatbot Window */}
      <div className="chatbot-window">
        <div className="chatbot-header">
          <div className="chatbot-status">
            <div className="chatbot-status-dot" />
            <div className="flex flex-col">
              <span className="text-[11px] font-bold text-gray-800 leading-tight">Trợ lý Luật lao động Việt Nam</span>
              <span className="text-[9px] text-emerald-500 font-medium">Hoạt động</span>
            </div>
          </div>
          <div className="chatbot-window-controls">
            <div className="chatbot-dot bg-red-400" />
            <div className="chatbot-dot bg-yellow-400" />
            <div className="chatbot-dot bg-green-400" />
          </div>
        </div>

        <div className="chatbot-body">
          <div className="chatbot-bubble">
            Xin chào! 👋 Tôi có thể hỗ trợ gì cho bạn về luật lao động hôm nay?
          </div>
          <div className="chatbot-avatar-container">
            <div className="chatbot-avatar-bg-glow" />
            <img src={assistantAvatar} alt="Trợ lý AI" className="chatbot-avatar-img" />
          </div>
        </div>

        <div className="chatbot-input-mock">
          <span className="chatbot-input-text">Hỏi về hợp đồng, tiền lương, nghỉ phép...</span>
          <div className="chatbot-input-btn">
            <SendHorizontal className="h-3.5 w-3.5" />
          </div>
        </div>
      </div>
    </div>
  )
}

function AuthPage({ mode }: { mode: "signin" | "register" }) {
  const { session, signInWithGoogle } = useAuth()
  const [error, setError] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)

  if (session) {
    return <Navigate to="/app" replace />
  }

  async function handleGoogleAuth() {
    setError("")
    setIsSubmitting(true)
    try {
      await signInWithGoogle()
    } catch {
      setError("Không thể bắt đầu đăng nhập bằng Google. Vui lòng thử lại.")
      setIsSubmitting(false)
    }
  }

  return (
    <main className="grid min-h-screen bg-background text-foreground lg:grid-cols-[1fr_0.86fr]">
      <section className="flex flex-col justify-between bg-navy px-6 py-8 text-white sm:px-10">
        <Link className="flex items-center gap-3" to="/">
          <Logo inverted />
        </Link>
        <div className="max-w-2xl py-12">
          <h1 className="text-3xl font-semibold tracking-normal sm:text-4xl">
            {mode === "signin" ? "Đăng nhập" : "Đăng ký"} vào {PRODUCT_NAME}
          </h1>
          <p className="mt-4 max-w-xl text-sm leading-6 text-slate-300 sm:text-base">
            Không gian tra cứu pháp luật lao động Việt Nam có căn cứ.
          </p>
        </div>
        <p className="text-xs leading-5 text-slate-400">{LEGAL_DISCLAIMER}</p>
      </section>

      <section className="flex items-center justify-center px-4 py-10 sm:px-6 relative hero-gradient-bg">
        <div className="hero-grid-overlay" />
        <div className="relative w-full max-w-md rounded-lg border border-border bg-surface/90 backdrop-blur p-6 shadow-soft z-10">
          <h2 className="text-xl font-semibold">
            {mode === "signin" ? "Đăng nhập" : "Tạo tài khoản"}
          </h2>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            Sử dụng tài khoản Google để tiếp tục.
          </p>
          {missingSupabaseConfig ? (
            <div className="mt-4 rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-warning">
              Thiếu cấu hình Supabase cho giao diện.
            </div>
          ) : null}
          {error ? (
            <div className="mt-4 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </div>
          ) : null}
          <button
            className="button-primary mt-6 w-full"
            disabled={isSubmitting || missingSupabaseConfig}
            onClick={handleGoogleAuth}
            type="button"
          >
            {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            Tiếp tục với Google
          </button>
          <p className="mt-5 text-sm text-muted-foreground">
            {mode === "signin" ? "Chưa có quyền truy cập?" : "Đã có quyền truy cập?"}{" "}
            <Link className="font-medium text-primary hover:underline" to={mode === "signin" ? "/register" : "/signin"}>
              {mode === "signin" ? "Đăng ký" : "Đăng nhập"}
            </Link>
          </p>
        </div>
      </section>
    </main>
  )
}

function AuthCallbackPage() {
  const navigate = useNavigate()
  const { refresh } = useAuth()

  useEffect(() => {
    let active = true
    refresh().then((session) => {
      if (!active) return
      navigate(session ? "/app" : "/signin", { replace: true })
    })
    return () => {
      active = false
    }
  }, [navigate, refresh])

  return (
    <main className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="card flex items-center gap-3 p-5 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin text-primary" />
        Đang hoàn tất đăng nhập...
      </div>
    </main>
  )
}

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isLoading, session } = useAuth()

  if (isLoading) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-background">
        <Loader2 className="h-5 w-5 animate-spin text-primary" />
      </main>
    )
  }

  if (!session) {
    return <Navigate to="/signin" replace />
  }

  return <>{children}</>
}

function resizeQuestionInput(element: HTMLTextAreaElement | null) {
  if (!element) return
  element.style.height = "0px"
  element.style.height = `${Math.min(element.scrollHeight, 144)}px`
}

function ResearchApp() {
  const { session } = useAuth()
  const formRef = useRef<HTMLFormElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [question, setQuestion] = useState("")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const refreshConversations = useCallback(async () => {
    if (!session) return
    try {
      const payload = await listConversations(session)
      setConversations(payload.conversations)
    } catch {
      setConversations([])
    }
  }, [session])

  useEffect(() => {
    void refreshConversations()
  }, [refreshConversations])

  useEffect(() => {
    resizeQuestionInput(textareaRef.current)
  }, [question])

  function handleQuestionChange(event: ChangeEvent<HTMLTextAreaElement>) {
    setQuestion(event.target.value)
    resizeQuestionInput(event.currentTarget)
  }

  function handleQuestionKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      formRef.current?.requestSubmit()
    }
  }

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!session || !question.trim() || isLoading) return

    const nextQuestion = question.trim()
    setQuestion("")
    setError("")
    setMessages((current) => [...current, { role: "user", content: nextQuestion }])
    setIsLoading(true)

    try {
      const response = await sendChatQuestion(session, nextQuestion, conversationId)
      if (!response.answer.trim()) {
        setError(insufficientContextMessage)
        return
      }
      setMessages((current) => [
        ...current,
        { role: "assistant", content: response.answer, citations: response.citations }
      ])
      if (response.conversationId) {
        setConversationId(response.conversationId)
      }
      void refreshConversations()
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Không thể lấy câu trả lời."
      setError(isInsufficientContextError(message) ? insufficientContextMessage : message)
    } finally {
      setIsLoading(false)
    }
  }

  async function selectConversation(id: string) {
    if (!session) return
    setError("")
    const payload = await getConversation(session, id)
    setConversationId(id)
    setMessages(
      payload.messages
        .flatMap((message) =>
          message.role === "user" || message.role === "assistant"
            ? [
                {
                  role: message.role,
                  content: message.content,
                  citations: normalizeChatCitations(message.citations)
                }
              ]
            : []
        )
    )
  }

  function startNewResearch() {
    setConversationId(null)
    setMessages([])
    setQuestion("")
    setError("")
  }

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <HistorySidebar
        conversationId={conversationId}
        conversations={conversations}
        onConversationSelect={selectConversation}
        onNewResearch={startNewResearch}
      />
      <main className="flex min-w-0 flex-1 flex-col">
        <AppHeader />
        <section className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-6 lg:px-8 relative hero-gradient-bg">
          <div className="hero-grid-overlay" />
          <div className="relative mx-auto max-w-3xl space-y-5 z-10">
            {messages.length === 0 ? (
              <EmptyState onPromptSelect={setQuestion} />
            ) : (
              messages.map((message, index) =>
                message.role === "user" ? (
                  <div className="flex justify-end" key={`${message.role}-${index}`}>
                    <div className="max-w-[82%] rounded-lg bg-primary px-4 py-3 text-sm leading-6 text-white">
                      {message.content}
                    </div>
                  </div>
                ) : (
                  <AnswerCard
                    citations={message.citations}
                    content={message.content}
                    key={`${message.role}-${index}`}
                  />
                )
              )
            )}
            {isLoading ? (
              <div className="card flex items-center gap-2 p-4 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
                Đang truy xuất căn cứ pháp lý phù hợp...
              </div>
            ) : null}
            {error ? (
              <div className="rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                {error}
              </div>
            ) : null}
          </div>
        </section>
        <form
          className="border-t border-border bg-surface/85 backdrop-blur px-4 py-4 sm:px-6 z-10"
          onSubmit={submitQuestion}
          ref={formRef}
        >
          <div className="mx-auto max-w-3xl">
            <div className="flex items-end gap-2 rounded-full border border-border bg-surface px-3 py-2 shadow-soft">
              <textarea
                className="max-h-36 min-h-11 flex-1 resize-none border-0 bg-transparent px-2 py-3 text-sm leading-5 outline-none placeholder:text-muted-foreground focus:ring-0"
                onChange={handleQuestionChange}
                onKeyDown={handleQuestionKeyDown}
                placeholder="Hỏi về luật lao động Việt Nam..."
                ref={textareaRef}
                rows={1}
                value={question}
              />
              <button
                aria-label="Gửi câu hỏi"
                className="button-primary h-11 w-11 shrink-0 rounded-full px-0"
                disabled={!question.trim() || isLoading}
                type="submit"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <SendHorizontal className="h-4 w-4" />
                )}
              </button>
            </div>
            <p className="mt-2 px-3 text-xs leading-5 text-muted-foreground">{LEGAL_DISCLAIMER}</p>
          </div>
        </form>
      </main>
    </div>
  )
}

function HistorySidebar({
  conversationId,
  conversations,
  onConversationSelect,
  onNewResearch
}: {
  conversationId: string | null
  conversations: ConversationSummary[]
  onConversationSelect: (id: string) => void | Promise<void>
  onNewResearch: () => void
}) {
  return (
    <aside className="hidden w-72 shrink-0 flex-col border-r border-border bg-surface lg:flex">
      <div className="border-b border-border p-4">
        <Logo />
        <button className="button-primary mt-4 w-full justify-start" onClick={onNewResearch} type="button">
          <Plus className="h-4 w-4" />
          Cuộc trò chuyện mới
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-4">
        <div className="mb-2 flex items-center gap-2 px-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          <History className="h-3.5 w-3.5" />
          Lịch sử
        </div>
        {conversations.length ? (
          <div className="space-y-1">
            {conversations.map((conversation) => {
              const active = conversation.id === conversationId
              return (
                <button
                  className={`w-full rounded-md px-3 py-2 text-left text-sm leading-5 transition-colors ${
                    active
                      ? "bg-primary/10 text-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`}
                  key={conversation.id}
                  onClick={() => void onConversationSelect(conversation.id)}
                  type="button"
                >
                  <span className="line-clamp-2">{conversation.title || "Cuộc trò chuyện chưa đặt tên"}</span>
                </button>
              )
            })}
          </div>
        ) : (
          <p className="rounded-md border border-border bg-background px-3 py-3 text-sm leading-6 text-muted-foreground">
            Chưa có cuộc trò chuyện nào.
          </p>
        )}
      </div>
    </aside>
  )
}

function AppHeader() {
  const { signOut, user } = useAuth()

  return (
    <header className="flex min-h-16 items-center justify-between gap-3 border-b border-border bg-surface px-4 py-3 sm:px-6">
      <div className="lg:hidden">
        <Logo compact />
      </div>
      <div className="hidden lg:block" />
      <div className="flex items-center gap-2">
        <span className="hidden max-w-56 truncate text-sm text-muted-foreground md:block">
          {user?.email}
        </span>
        <button className="button-secondary h-9" onClick={() => void signOut()} type="button">
          <LogOut className="h-4 w-4" />
          Đăng xuất
        </button>
      </div>
    </header>
  )
}

function AccountPage() {
  const { signOut, user } = useAuth()
  return (
    <div className="min-h-screen bg-background text-foreground">
      <MarketingHeader />
      <main className="mx-auto max-w-3xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="card p-6 shadow-soft">
          <h1 className="text-2xl font-semibold">Tài khoản</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Quản lý phiên tra cứu đang đăng nhập.
          </p>
          <div className="mt-6 grid gap-3 text-sm">
            <MetaRow label="Email" value={user?.email ?? "Không xác định"} />
            <MetaRow label="Mã người dùng" value={user?.id ?? "Không xác định"} />
          </div>
          <button className="button-primary mt-6" onClick={() => void signOut()} type="button">
            Đăng xuất
          </button>
        </div>
      </main>
    </div>
  )
}

function AdminPage() {
  const { session } = useAuth()
  const [isLoading, setIsLoading] = useState(true)
  const [errorStatus, setErrorStatus] = useState<number | null>(null)
  const [errorMessage, setErrorMessage] = useState("")
  const [stats, setStats] = useState<AdminStats | null>(null)
  const [health, setHealth] = useState<Record<string, Record<string, unknown>>>({})
  const [retrievalConfig, setRetrievalConfig] = useState<Record<string, unknown>>({})
  const [traces, setTraces] = useState<AdminTrace[]>([])

  useEffect(() => {
    let active = true
    async function loadAdminConsole() {
      if (!session) return
      setIsLoading(true)
      setErrorStatus(null)
      setErrorMessage("")
      try {
        const [, statsPayload, healthPayload, retrievalPayload, tracesPayload] =
          await Promise.all([
            getCurrentUser(session),
            getAdminStats(session),
            getAdminHealth(session),
            getAdminRetrievalConfig(session),
            getAdminTraces(session, 20)
          ])
        if (!active) return
        setStats(statsPayload.stats)
        setHealth(healthPayload.checks)
        setRetrievalConfig(retrievalPayload)
        setTraces(tracesPayload.traces)
      } catch (caught) {
        if (!active) return
        if (caught instanceof ApiError) {
          setErrorStatus(caught.status)
          setErrorMessage(caught.message)
        } else {
          setErrorMessage("Không thể tải bảng quản trị.")
        }
      } finally {
        if (active) {
          setIsLoading(false)
        }
      }
    }
    void loadAdminConsole()
    return () => {
      active = false
    }
  }, [session])

  return (
    <div className="min-h-screen bg-background text-foreground">
      <AppHeader />
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-3 border-b border-border pb-6 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-3xl font-semibold">Bảng quản trị</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-muted-foreground">
              Theo dõi xác thực, trạng thái vận hành, cấu hình truy xuất và các lượt trả lời gần đây.
            </p>
          </div>
        </div>

        {isLoading ? (
          <div className="card mt-8 flex items-center gap-3 p-5 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            Đang tải bảng quản trị...
          </div>
        ) : errorStatus ? (
          <AdminAccessState status={errorStatus} fallbackMessage={errorMessage} />
        ) : errorMessage ? (
          <div className="mt-8 rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {errorMessage}
          </div>
        ) : (
          <div className="space-y-8 pt-8">
            <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {adminStatCards(stats).map((card) => (
                <article className="card p-5 shadow-sm" key={card.label}>
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    {card.label}
                  </p>
                  <p className="mt-3 text-3xl font-semibold">{card.value}</p>
                </article>
              ))}
            </section>

            <section className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
              <div className="card p-5 shadow-sm">
                <h2 className="text-lg font-semibold">Trạng thái hệ thống</h2>
                <div className="mt-5 space-y-3">
                  {[
                    ["database", "Cơ sở dữ liệu"],
                    ["settings", "Cấu hình"],
                    ["index", "Chỉ mục"],
                    ["qdrantConfig", "Cấu hình Qdrant"],
                    ["llmConfig", "Cấu hình mô hình"]
                  ].map(([key, label]) => (
                    <AdminKeyValueRow key={key} label={label} value={health[key] ?? { status: "không xác định" }} />
                  ))}
                </div>
              </div>

              <div className="card p-5 shadow-sm">
                <h2 className="text-lg font-semibold">Cấu hình truy xuất</h2>
                <div className="mt-5 max-h-96 space-y-2 overflow-y-auto">
                  {Object.entries(retrievalConfig).map(([key, value]) => (
                    <AdminKeyValueRow key={key} label={adminConfigLabel(key)} value={value} />
                  ))}
                </div>
              </div>
            </section>

            <section className="card overflow-hidden shadow-sm">
              <div className="border-b border-border px-5 py-4">
                <h2 className="text-lg font-semibold">Nhật ký trả lời gần đây</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-border text-sm">
                  <thead className="bg-background text-left text-xs uppercase tracking-wide text-muted-foreground">
                    <tr>
                      <th className="px-5 py-3 font-semibold">Thời điểm</th>
                      <th className="px-5 py-3 font-semibold">Câu hỏi</th>
                      <th className="px-5 py-3 font-semibold">Nhà cung cấp / mô hình</th>
                      <th className="px-5 py-3 font-semibold">Thiếu ngữ cảnh</th>
                      <th className="px-5 py-3 font-semibold">Lỗi</th>
                      <th className="px-5 py-3 font-semibold">Độ trễ</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {traces.length ? (
                      traces.map((trace) => (
                        <tr key={trace.id}>
                          <td className="whitespace-nowrap px-5 py-4 text-muted-foreground">
                            {formatDate(trace.createdAt ?? trace.created_at)}
                          </td>
                          <td className="max-w-md px-5 py-4 leading-6">
                            {trace.question}
                          </td>
                          <td className="px-5 py-4 text-muted-foreground">
                            {trace.provider || "Không xác định"} / {trace.model || "mặc định"}
                          </td>
                          <td className="px-5 py-4">
                            {trace.insufficientContext || trace.insufficient_context ? "Có" : "Không"}
                          </td>
                          <td className="max-w-xs px-5 py-4 text-muted-foreground">
                            {trace.error || "Không có"}
                          </td>
                          <td className="whitespace-nowrap px-5 py-4 text-muted-foreground">
                            {trace.latencyMs ?? trace.latency_ms ?? "Chưa có"} ms
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td className="px-5 py-6 text-muted-foreground" colSpan={6}>
                          Chưa có nhật ký nào.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  )
}

function AdminAccessState({
  fallbackMessage,
  status
}: {
  fallbackMessage: string
  status: number
}) {
  if (status === 403) {
    return (
      <div className="card mt-8 max-w-2xl p-6 shadow-sm">
        <h2 className="text-xl font-semibold">Cần quyền quản trị</h2>
        <p className="mt-3 text-sm leading-6 text-muted-foreground">
          Tài khoản của bạn đã đăng nhập nhưng chưa nằm trong danh sách quản trị ở backend.
        </p>
      </div>
    )
  }

  if (status === 401) {
    return (
      <div className="card mt-8 max-w-2xl p-6 shadow-sm">
        <h2 className="text-xl font-semibold">Không xác minh được phiên đăng nhập. Vui lòng đăng nhập lại.</h2>
      </div>
    )
  }

  return (
    <div className="mt-8 rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
      {fallbackMessage || "Không thể tải bảng quản trị."}
    </div>
  )
}

function EmptyState({ onPromptSelect }: { onPromptSelect: (prompt: string) => void }) {
  return (
    <div className="grid min-h-[45vh] content-center gap-8 text-center">
      <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-primary">
        <Scale className="h-8 w-8" />
      </div>
      <div className="mx-auto max-w-2xl">
        <h1 className="text-2xl font-semibold sm:text-3xl">
          Bạn muốn tra cứu vấn đề lao động nào?
        </h1>
        <p className="mt-3 text-sm leading-6 text-muted-foreground">
          Đặt câu hỏi bằng tiếng Việt; hệ thống sẽ tìm căn cứ pháp lý và trình bày câu trả lời có trích dẫn.
        </p>
      </div>
      <div className="mx-auto flex max-w-2xl flex-wrap justify-center gap-x-4 gap-y-2">
        {suggestedPrompts.map((prompt) => (
          <button
            className="text-sm leading-6 text-primary hover:underline"
            key={prompt}
            onClick={() => onPromptSelect(prompt)}
            type="button"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  )
}

function AnswerCard({
  citations,
  content
}: {
  citations?: ChatCitations
  content: string
}) {
  const [copied, setCopied] = useState(false)
  const legalBasis = citations?.legalBasis ?? []
  async function copyAnswer() {
    await navigator.clipboard.writeText(content)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <article className="card p-4 shadow-sm">
      <div className="space-y-3">
        <p className="whitespace-pre-wrap text-sm leading-7">{content}</p>
        {legalBasis.length ? (
          <p className="border-t border-border pt-3 text-xs leading-5 text-muted-foreground">
            <span className="font-medium text-foreground">Căn cứ pháp lý: </span>
            {legalBasis.slice(0, 3).join(" | ")}
          </p>
        ) : null}
        <button className="button-secondary h-9" onClick={copyAnswer} type="button">
          <Copy className="h-4 w-4" />
          {copied ? "Đã sao chép" : "Sao chép"}
        </button>
      </div>
    </article>
  )
}

function MarketingHeader() {
  const { session } = useAuth()
  return (
    <header className="sticky top-0 z-20 border-b border-border bg-surface/95 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-3xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link to="/">
          <Logo />
        </Link>
        <div className="flex items-center gap-2">
          {session ? (
            <Link className="button-primary h-9" to="/app">Mở trợ lý</Link>
          ) : (
            <Link className="button-primary h-9" to="/signin">Bắt đầu</Link>
          )}
        </div>
      </div>
    </header>
  )
}

function Logo({ compact = false, inverted = false }: { compact?: boolean; inverted?: boolean }) {
  return (
    <span className="flex items-center gap-3">
      <span className={`flex h-9 w-9 items-center justify-center rounded-md ${inverted ? "bg-white text-navy" : "bg-primary text-white"}`}>
        <Scale className="h-5 w-5" />
      </span>
      {compact ? null : <span className="font-semibold">{PRODUCT_NAME}</span>}
    </span>
  )
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-md border border-border bg-background px-3 py-2">
      <span className="text-muted-foreground">{label}</span>
      <span className="break-all text-right font-medium">{value}</span>
    </div>
  )
}

function AdminKeyValueRow({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="grid gap-2 rounded-md border border-border bg-background px-3 py-3 sm:grid-cols-[12rem_1fr]">
      <span className="text-sm font-medium text-muted-foreground">{label}</span>
      <span className="break-words text-sm leading-6">{formatAdminValue(value)}</span>
    </div>
  )
}

function adminStatCards(stats: AdminStats | null) {
  return [
    { label: "Tổng người dùng", value: stats?.totalUsers ?? 0 },
    { label: "Người dùng hoạt động", value: stats?.activeUsers ?? 0 },
    { label: "Quản trị viên", value: stats?.adminUsers ?? 0 },
    { label: "Cuộc trò chuyện", value: stats?.totalConversations ?? 0 },
    { label: "Tin nhắn", value: stats?.totalMessages ?? 0 },
    { label: "Tổng nhật ký", value: stats?.totalTraces ?? 0 },
    { label: "Nhật ký có lỗi", value: stats?.tracesWithErrors ?? 0 },
    {
      label: "Nhật ký thiếu ngữ cảnh",
      value: stats?.insufficientContextTraces ?? 0
    }
  ]
}

function formatAdminValue(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "Chưa cấu hình"
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return localizeAdminScalar(value)
  }
  return JSON.stringify(localizeAdminValue(value))
}

function localizeAdminValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(localizeAdminValue)
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, nested]) => [
        adminConfigLabel(key),
        localizeAdminValue(nested)
      ])
    )
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return localizeAdminScalar(value)
  }
  return value
}

function localizeAdminScalar(value: string | number | boolean): string {
  if (typeof value === "boolean") {
    return value ? "Có" : "Không"
  }
  const text = String(value)
  const normalized = text.toLowerCase()
  const replacements: Record<string, string> = {
    ok: "ổn định",
    configured: "đã cấu hình",
    missing: "thiếu cấu hình",
    local: "cục bộ",
    degraded: "suy giảm",
    error: "lỗi",
    "database is reachable.": "Có thể kết nối cơ sở dữ liệu.",
    "settings loaded.": "Đã tải cấu hình.",
    "required settings are missing.": "Thiếu cấu hình bắt buộc.",
    "index path exists.": "Đường dẫn chỉ mục tồn tại.",
    "index path does not exist.": "Đường dẫn chỉ mục không tồn tại."
  }
  return replacements[normalized] ?? text
}

function adminConfigLabel(key: string): string {
  const labels: Record<string, string> = {
    activeSessions: "Phiên hoạt động",
    authProvider: "Nhà cung cấp xác thực",
    collection: "Bộ sưu tập",
    database: "Cơ sở dữ liệu",
    denseModel: "Mô hình dense",
    embeddingProvider: "Nhà cung cấp embedding",
    error: "Lỗi",
    graphEnabled: "Bật đồ thị",
    index: "Chỉ mục",
    indexPath: "Đường dẫn chỉ mục",
    llmConfig: "Cấu hình mô hình",
    message: "Thông báo",
    model: "Mô hình",
    path: "Đường dẫn",
    provider: "Nhà cung cấp",
    qdrantCollection: "Bộ sưu tập Qdrant",
    qdrantConfig: "Cấu hình Qdrant",
    qdrantUrl: "Đường dẫn Qdrant",
    rerankerModel: "Mô hình xếp hạng lại",
    rerankerTopN: "Số kết quả xếp hạng lại",
    retrieverRecordSource: "Nguồn bản ghi truy xuất",
    settings: "Cấu hình",
    status: "Trạng thái",
    topK: "Số kết quả truy xuất",
    usesCloud: "Dùng dịch vụ đám mây"
  }
  return labels[key] ?? key
}

function formatDate(value: string | undefined): string {
  if (!value) {
    return "Không xác định"
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString()
}

function Footer() {
  return (
    <footer className="border-t border-border bg-surface">
      <div className="mx-auto max-w-3xl px-4 py-4 text-center text-xs leading-5 text-muted-foreground sm:px-6 lg:px-8">
        {LEGAL_DISCLAIMER}
      </div>
    </footer>
  )
}

function isInsufficientContextError(message: string) {
  const normalized = message.toLowerCase()
  return (
    normalized.includes("insufficient") ||
    normalized.includes("not enough") ||
    normalized.includes("outside the supported") ||
    normalized.includes("could not find enough")
  )
}
