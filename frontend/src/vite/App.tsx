import {
  ArrowRight,
  BookOpenCheck,
  CheckCircle2,
  Copy,
  FileText,
  Loader2,
  LogOut,
  MessageSquareText,
  Plus,
  Scale,
  SearchCheck,
  ShieldCheck
} from "lucide-react"
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react"
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
  sendChatQuestion,
  ChatMessage
} from "./backend"
import { useAuth } from "./auth"
import { LEGAL_DISCLAIMER, missingSupabaseConfig, PRODUCT_NAME } from "./config"

const suggestedPrompts = [
  "What is the legal definition of an employee?",
  "When can an employee terminate a contract without prior notice?",
  "What rules apply to workers under 15?",
  "How is retirement age determined under Decree 135/2020/ND-CP?"
]

const commonTopics = [
  "Employment contracts",
  "Termination",
  "Wages",
  "Working hours",
  "Leave",
  "Minor workers",
  "Retirement"
]

const sourceCoverage = [
  "Labor Code 2019",
  "Decree No. 145/2020/ND-CP",
  "Decree No. 135/2020/ND-CP",
  "Selected labor-related legal documents"
]

const sourceMetadata = [
  { label: "Last indexed", value: "Configured by deployment" },
  { label: "Coverage", value: "Vietnamese labor law only" },
  { label: "Language", value: "English interface, Vietnamese legal source grounding" },
  { label: "Citation support", value: "Enabled" }
]

const useCases = [
  {
    title: "Employment contracts",
    description: "Review contract formation, required terms, probation, amendments, and statutory termination grounds."
  },
  {
    title: "Working time and wages",
    description: "Research working hours, overtime, wage payment rules, salary deductions, and related compliance questions."
  },
  {
    title: "Workplace compliance",
    description: "Support HR and legal checks for internal labor rules, obligations, and employer responsibilities."
  },
  {
    title: "Special workers",
    description: "Analyze rules for minor workers, older employees, and other worker categories with additional protections."
  },
  {
    title: "Retirement and social policy",
    description: "Research retirement age, transition schedules, and related labor-policy provisions."
  }
]

const reliabilitySteps = [
  {
    title: "Retrieve",
    description: "Retrieve relevant provisions from the indexed legal corpus."
  },
  {
    title: "Ground",
    description: "Generate answers only from retrieved legal context."
  },
  {
    title: "Cite",
    description: "Attach legal basis and source snippets."
  },
  {
    title: "Limit",
    description: "Refuse to speculate when context is insufficient."
  }
]

const securityLimitations = [
  "The assistant is designed for research support, not final legal advice.",
  "Answers are limited to the indexed Vietnamese labor-law corpus.",
  "The system may refuse questions outside the supported source coverage.",
  "Users should verify citations before relying on an answer.",
  "Authorized retrieval diagnostics are available for debugging and evaluation."
]

const insufficientContextMessage =
  "I could not find enough legal context in the indexed sources to answer this reliably."

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
    <div className="min-h-screen bg-background text-foreground">
      <MarketingHeader />
      <main>
        <section className="border-b border-border bg-gradient-to-b from-white to-background" id="product">
          <div className="mx-auto grid min-h-[calc(100vh-4rem)] max-w-7xl items-center gap-12 px-4 py-16 sm:px-6 lg:grid-cols-[1fr_0.9fr] lg:px-8">
            <div>
              <span className="inline-flex rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-sm font-medium text-primary">
                Legal research assistant for Vietnamese labor law
              </span>
              <h1 className="mt-6 max-w-3xl text-4xl font-semibold tracking-normal text-foreground sm:text-5xl">
                Vietnamese labor-law answers, grounded in official legal sources.
              </h1>
              <p className="mt-5 max-w-2xl text-base leading-8 text-muted-foreground">
                Ask questions about employment contracts, wages, working hours,
                leave, termination, retirement, and minor workers with answers
                supported by retrieved legal provisions and citations.
              </p>
              <div className="mt-8 flex flex-col gap-3 sm:flex-row">
                <Link className="button-primary" to="/signin">
                  Start researching
                  <ArrowRight className="h-4 w-4" />
                </Link>
                <a className="button-secondary" href="#sources">
                  View source coverage
                </a>
              </div>
              <p className="mt-5 text-sm leading-6 text-muted-foreground">
                {LEGAL_DISCLAIMER}
              </p>
            </div>
            <ProductPreview />
          </div>
        </section>

        <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <FeatureCard
              icon={BookOpenCheck}
              title="Grounded legal answers"
              description="Every response is generated from retrieved provisions in the indexed legal corpus."
            />
            <FeatureCard
              icon={FileText}
              title="Citation-first research"
              description="Answers include legal bases, document names, article numbers, and source snippets."
            />
            <FeatureCard
              icon={ShieldCheck}
              title="Refuses unsupported answers"
              description="When the retrieved context is insufficient, the assistant explains the limitation instead of guessing."
            />
            <FeatureCard
              icon={SearchCheck}
              title="Source audit trail"
              description="Review the exact passages used to support each answer, including source metadata."
            />
          </div>
        </section>

        <section className="border-y border-border bg-surface" id="sources">
          <div className="mx-auto grid max-w-7xl gap-8 px-4 py-16 sm:px-6 lg:grid-cols-[0.8fr_1.2fr] lg:px-8">
            <div>
              <h2 className="text-3xl font-semibold">Source coverage</h2>
              <p className="mt-4 text-sm leading-7 text-muted-foreground">
                The assistant currently answers questions based on a curated
                Vietnamese labor-law corpus.
              </p>
            </div>
            <div className="space-y-5">
              <div className="grid gap-3 sm:grid-cols-2">
                {sourceCoverage.map((source) => (
                  <div className="card flex items-center gap-3 p-4" key={source}>
                    <CheckCircle2 className="h-4 w-4 text-success" />
                    <span className="text-sm font-medium">{source}</span>
                  </div>
                ))}
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                {sourceMetadata.map((item) => (
                  <div className="rounded-lg border border-border bg-background p-4" key={item.label}>
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      {item.label}
                    </p>
                    <p className="mt-2 text-sm font-medium">{item.value}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8" id="use-cases">
          <div className="max-w-2xl">
            <h2 className="text-3xl font-semibold">Built for labor-law research workflows</h2>
            <p className="mt-4 text-sm leading-7 text-muted-foreground">
              Move from question to legal basis with workflows suited to legal,
              HR, compliance, and academic review.
            </p>
          </div>
          <div className="mt-8 grid gap-4 md:grid-cols-2 lg:grid-cols-5">
            {useCases.map((useCase) => (
              <article className="card p-5" key={useCase.title}>
                <h3 className="text-base font-semibold">{useCase.title}</h3>
                <p className="mt-3 text-sm leading-6 text-muted-foreground">
                  {useCase.description}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section className="border-y border-border bg-surface" id="reliability">
          <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
            <div className="max-w-2xl">
              <h2 className="text-3xl font-semibold">Reliability by design</h2>
              <p className="mt-4 text-sm leading-7 text-muted-foreground">
                The assistant is built to connect each answer to the legal
                context available in the indexed corpus.
              </p>
            </div>
            <div className="mt-8 grid gap-4 md:grid-cols-4">
              {reliabilitySteps.map((step, index) => (
                <article className="rounded-lg border border-border bg-background p-5" key={step.title}>
                  <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary text-sm font-semibold text-white">
                    {index + 1}
                  </div>
                  <h3 className="mt-5 text-base font-semibold">{step.title}</h3>
                  <p className="mt-3 text-sm leading-6 text-muted-foreground">
                    {step.description}
                  </p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8" id="security">
          <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
            <div>
              <h2 className="text-3xl font-semibold">Security and limitations</h2>
              <p className="mt-4 text-sm leading-7 text-muted-foreground">
                Designed for responsible legal research support with clear
                limitations and citation review built into the workflow.
              </p>
            </div>
            <div className="card p-5 shadow-sm">
              <ul className="space-y-3">
                {securityLimitations.map((item) => (
                  <li className="flex gap-3 text-sm leading-6 text-muted-foreground" key={item}>
                    <CheckCircle2 className="mt-1 h-4 w-4 shrink-0 text-success" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <section className="border-t border-border bg-surface" id="about">
          <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
            <div className="max-w-3xl">
              <h2 className="text-3xl font-semibold">About {PRODUCT_NAME}</h2>
              <p className="mt-4 text-sm leading-7 text-muted-foreground">
                {PRODUCT_NAME} is a focused legal research workspace for
                Vietnamese labor-law questions. It helps researchers, HR teams,
                compliance reviewers, and students work with official source
                grounding and clear limitations.
              </p>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  )
}

function ProductPreview() {
  return (
    <div className="card overflow-hidden shadow-soft">
      <div className="border-b border-border bg-navy px-5 py-4 text-white">
        <p className="text-xs uppercase tracking-wide text-slate-300">Research workspace</p>
        <p className="mt-1 text-sm font-semibold">Contract termination review</p>
      </div>
      <div className="grid gap-4 p-5">
        <div className="rounded-lg border border-border bg-background p-4">
          <p className="text-xs font-semibold uppercase text-muted-foreground">Question</p>
          <p className="mt-2 text-sm font-medium">
            Can an employee terminate an employment contract without prior notice?
          </p>
        </div>
        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <MessageSquareText className="h-4 w-4 text-primary" />
            Generated legal answer
          </div>
          <p className="mt-3 text-sm leading-7 text-muted-foreground">
            Under Vietnamese labor law, an employee may terminate the employment
            contract without prior notice in specific statutory situations,
            including not being assigned the agreed work, not being paid fully or
            on time, being abused or harassed, or other legally defined grounds.
          </p>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Legal basis
          </p>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <CitationPill title="Labor Code 2019" detail="Article 35" />
            <CitationPill title="Decree No. 145/2020/ND-CP" detail="Related guidance" />
          </div>
        </div>
        <div className="rounded-lg border border-border bg-background p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Retrieved source snippet
          </p>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            Employee termination rights may apply where statutory grounds are
            satisfied, including non-payment, mistreatment, or failure to assign
            agreed work.
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <button className="button-secondary h-9" type="button">Open source</button>
          <button className="button-primary h-9" type="button">Copy answer</button>
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
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to start Google sign-in.")
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
            {mode === "signin" ? "Sign in" : "Register"} to {PRODUCT_NAME}
          </h1>
          <p className="mt-4 max-w-xl text-sm leading-6 text-slate-300 sm:text-base">
            Secure legal research workspace for Vietnamese labor-law Q&A.
          </p>
        </div>
        <p className="text-xs leading-5 text-slate-400">{LEGAL_DISCLAIMER}</p>
      </section>

      <section className="flex items-center justify-center px-4 py-10 sm:px-6">
        <div className="w-full max-w-md rounded-lg border border-border bg-surface p-6 shadow-soft">
          <h2 className="text-xl font-semibold">
            {mode === "signin" ? "Sign in" : "Create your account"}
          </h2>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            Use Google OAuth through Supabase Auth.
          </p>
          {missingSupabaseConfig ? (
            <div className="mt-4 rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-warning">
              Missing Supabase frontend environment variables.
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
            Continue with Google
          </button>
          <p className="mt-5 text-sm text-muted-foreground">
            {mode === "signin" ? "Need access?" : "Already have access?"}{" "}
            <Link className="font-medium text-primary hover:underline" to={mode === "signin" ? "/register" : "/signin"}>
              {mode === "signin" ? "Register" : "Sign in"}
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
        Completing secure sign-in...
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

function ResearchApp() {
  const { session } = useAuth()
  const [question, setQuestion] = useState("")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const latestAnswer = useMemo(
    () => [...messages].reverse().find((message) => message.role === "assistant"),
    [messages]
  )

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
      setMessages((current) => [...current, { role: "assistant", content: response.answer }])
      if (response.conversationId) {
        setConversationId(response.conversationId)
      }
      void refreshConversations()
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Unable to retrieve an answer."
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
            ? [{ role: message.role, content: message.content }]
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
      <Sidebar
        conversations={conversations}
        onNewResearch={startNewResearch}
        onPromptSelect={setQuestion}
        onConversationSelect={selectConversation}
      />
      <main className="flex min-w-0 flex-1 flex-col">
        <AppHeader />
        <section className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-4xl space-y-6">
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
                  <AnswerCard content={message.content} key={`${message.role}-${index}`} />
                )
              )
            )}
            {isLoading ? (
              <div className="card flex items-center gap-2 p-4 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
                Retrieving relevant legal provisions...
              </div>
            ) : null}
            {error ? (
              <div className="rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                {error}
              </div>
            ) : null}
          </div>
        </section>
        <form className="border-t border-border bg-background/95 px-4 py-4 sm:px-6" onSubmit={submitQuestion}>
          <div className="mx-auto max-w-4xl rounded-lg border border-border bg-surface p-2 shadow-soft">
            <textarea
              className="min-h-24 w-full resize-none rounded-md border-0 bg-transparent px-3 py-3 text-sm leading-6 outline-none placeholder:text-muted-foreground focus:ring-0"
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question about Vietnamese labor law..."
              value={question}
            />
            <div className="flex flex-col gap-3 border-t border-border px-1 pt-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-xs leading-5 text-muted-foreground">{LEGAL_DISCLAIMER}</p>
              <button className="button-primary" disabled={!question.trim() || isLoading} type="submit">
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                Send
              </button>
            </div>
          </div>
        </form>
      </main>
      {latestAnswer ? <SourcePanel answer={latestAnswer.content} /> : null}
    </div>
  )
}

function Sidebar({
  conversations,
  onConversationSelect,
  onNewResearch,
  onPromptSelect
}: {
  conversations: ConversationSummary[]
  onConversationSelect: (id: string) => void
  onNewResearch: () => void
  onPromptSelect: (prompt: string) => void
}) {
  return (
    <aside className="hidden w-80 shrink-0 flex-col border-r border-border bg-surface lg:flex">
      <div className="p-4">
        <button className="button-primary w-full justify-start" onClick={onNewResearch} type="button">
          <Plus className="h-4 w-4" />
          New research
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-4">
        <SidebarSection title="Recent questions">
          {conversations.length ? (
            <div className="space-y-1">
              {conversations.slice(0, 8).map((conversation) => (
                <button
                  className="w-full rounded-md px-3 py-2 text-left text-sm leading-5 text-muted-foreground hover:bg-muted hover:text-foreground"
                  key={conversation.id}
                  onClick={() => onConversationSelect(conversation.id)}
                  type="button"
                >
                  {conversation.title}
                </button>
              ))}
            </div>
          ) : (
            <p className="rounded-md border border-border bg-background px-3 py-2 text-sm leading-5 text-muted-foreground">
              Recent research will appear here.
            </p>
          )}
        </SidebarSection>
        <SidebarSection title="Source library">
          <div className="rounded-md border border-border bg-background p-3">
            <div className="space-y-2">
              {sourceCoverage.map((source) => (
                <div className="flex items-start gap-2 text-sm leading-5 text-muted-foreground" key={source}>
                  <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-success" />
                  <span>{source}</span>
                </div>
              ))}
            </div>
          </div>
        </SidebarSection>
        <SidebarSection title="Common topics">
          <div className="grid gap-2">
            {commonTopics.map((topic) => (
              <button
                className="rounded-md border border-border bg-background px-3 py-2 text-left text-sm hover:border-primary/40 hover:bg-primary/5"
                key={topic}
                onClick={() => onPromptSelect(promptForTopic(topic))}
                type="button"
              >
                {topic}
              </button>
            ))}
          </div>
        </SidebarSection>
      </div>
    </aside>
  )
}

function AppHeader() {
  const { session, signOut, user } = useAuth()
  const [isAdmin, setIsAdmin] = useState(false)

  useEffect(() => {
    let active = true
    if (!session) {
      setIsAdmin(false)
      return
    }
    getCurrentUser(session)
      .then((payload) => {
        if (active) {
          setIsAdmin(payload.user.role === "admin")
        }
      })
      .catch(() => {
        if (active) {
          setIsAdmin(false)
        }
      })
    return () => {
      active = false
    }
  }, [session])

  return (
    <header className="flex h-16 items-center justify-between border-b border-border bg-surface px-4 sm:px-6">
      <Logo />
      <div className="flex items-center gap-3">
        <span className="hidden max-w-56 truncate text-sm text-muted-foreground md:block">
          {user?.email}
        </span>
        <Link className="button-secondary h-9" to="/account">
          Account
        </Link>
        {isAdmin ? (
          <Link className="button-secondary h-9" to="/admin">
            Admin
          </Link>
        ) : null}
        <button className="button-secondary h-9" onClick={() => void signOut()} type="button">
          <LogOut className="h-4 w-4" />
          Sign out
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
          <h1 className="text-2xl font-semibold">Account</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Manage your authenticated research session.
          </p>
          <div className="mt-6 grid gap-3 text-sm">
            <MetaRow label="Email" value={user?.email ?? "Unknown"} />
            <MetaRow label="User ID" value={user?.id ?? "Unknown"} />
          </div>
          <button className="button-primary mt-6" onClick={() => void signOut()} type="button">
            Sign out
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
          setErrorMessage(caught instanceof Error ? caught.message : "Unable to load admin console.")
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
            <h1 className="text-3xl font-semibold">Admin console</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-muted-foreground">
              Monitor authentication, runtime health, retrieval configuration,
              and recent answer traces.
            </p>
          </div>
        </div>

        {isLoading ? (
          <div className="card mt-8 flex items-center gap-3 p-5 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            Loading admin console...
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
                <h2 className="text-lg font-semibold">Health</h2>
                <div className="mt-5 space-y-3">
                  {["database", "settings", "index", "qdrantConfig", "llmConfig"].map((key) => (
                    <AdminKeyValueRow key={key} label={key} value={health[key] ?? { status: "unknown" }} />
                  ))}
                </div>
              </div>

              <div className="card p-5 shadow-sm">
                <h2 className="text-lg font-semibold">Retrieval configuration</h2>
                <div className="mt-5 max-h-96 space-y-2 overflow-y-auto">
                  {Object.entries(retrievalConfig).map(([key, value]) => (
                    <AdminKeyValueRow key={key} label={key} value={value} />
                  ))}
                </div>
              </div>
            </section>

            <section className="card overflow-hidden shadow-sm">
              <div className="border-b border-border px-5 py-4">
                <h2 className="text-lg font-semibold">Recent traces</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-border text-sm">
                  <thead className="bg-background text-left text-xs uppercase tracking-wide text-muted-foreground">
                    <tr>
                      <th className="px-5 py-3 font-semibold">Created</th>
                      <th className="px-5 py-3 font-semibold">Question</th>
                      <th className="px-5 py-3 font-semibold">Provider / model</th>
                      <th className="px-5 py-3 font-semibold">Insufficient context</th>
                      <th className="px-5 py-3 font-semibold">Error</th>
                      <th className="px-5 py-3 font-semibold">Latency</th>
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
                            {trace.provider || "Unknown"} / {trace.model || "default"}
                          </td>
                          <td className="px-5 py-4">
                            {trace.insufficientContext || trace.insufficient_context ? "Yes" : "No"}
                          </td>
                          <td className="max-w-xs px-5 py-4 text-muted-foreground">
                            {trace.error || "None"}
                          </td>
                          <td className="whitespace-nowrap px-5 py-4 text-muted-foreground">
                            {trace.latencyMs ?? trace.latency_ms ?? "N/A"} ms
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td className="px-5 py-6 text-muted-foreground" colSpan={6}>
                          No traces found.
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
        <h2 className="text-xl font-semibold">Admin access required</h2>
        <p className="mt-3 text-sm leading-6 text-muted-foreground">
          Your account is authenticated, but it is not listed in ADMIN_EMAILS on the backend.
        </p>
      </div>
    )
  }

  if (status === 401) {
    return (
      <div className="card mt-8 max-w-2xl p-6 shadow-sm">
        <h2 className="text-xl font-semibold">Your session could not be verified. Please sign in again.</h2>
      </div>
    )
  }

  return (
    <div className="mt-8 rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
      {fallbackMessage || "Unable to load admin console."}
    </div>
  )
}

function EmptyState({ onPromptSelect }: { onPromptSelect: (prompt: string) => void }) {
  return (
    <div className="grid min-h-[55vh] content-center gap-8">
      <div className="mx-auto max-w-2xl text-center">
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-md bg-primary/10 text-primary">
          <SearchCheck className="h-6 w-6" />
        </div>
        <h1 className="mt-4 text-2xl font-semibold sm:text-3xl">
          Ask a Vietnamese labor-law question
        </h1>
        <p className="mt-3 text-sm leading-6 text-muted-foreground">
          Research employment contracts, wages, working hours, leave,
          termination, minor workers, and retirement with grounded legal answers.
        </p>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {suggestedPrompts.map((prompt) => (
          <button
            className="card p-4 text-left text-sm leading-6 hover:border-primary/50 hover:bg-primary/5"
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

function AnswerCard({ content }: { content: string }) {
  const [copied, setCopied] = useState(false)
  async function copyAnswer() {
    await navigator.clipboard.writeText(content)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  function exportAnswer() {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = "vietnam-labor-law-answer.txt"
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <article className="card overflow-hidden shadow-sm">
      <div className="border-b border-border bg-background px-4 py-3">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <ShieldCheck className="h-4 w-4 text-success" />
          Grounded legal answer
        </div>
      </div>
      <div className="space-y-5 p-4">
        <section>
          <h2 className="text-sm font-semibold">Answer</h2>
          <p className="mt-3 whitespace-pre-wrap text-sm leading-7">{content}</p>
        </section>
        <section className="grid gap-3 md:grid-cols-2">
          <div className="rounded-md border border-border bg-background px-3 py-3">
            <h2 className="text-sm font-semibold">Legal basis</h2>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              Review the cited documents and article references included in the
              answer and source panel.
            </p>
          </div>
          <div className="rounded-md border border-border bg-background px-3 py-3">
            <h2 className="text-sm font-semibold">Retrieved provisions</h2>
            <p className="mt-2 line-clamp-4 text-sm leading-6 text-muted-foreground">
              Source passages used by the answer are shown when available from
              the backend response and supporting source panel.
            </p>
          </div>
        </section>
        <section className="rounded-md border border-success/20 bg-success/10 px-3 py-2 text-sm leading-6 text-success">
          <span className="font-semibold">Reliability note: </span>
          This answer is based only on the indexed labor-law source library.
          Verify the cited provisions before use.
        </section>
        <div className="flex flex-wrap gap-2 border-t border-border pt-4">
          <button className="button-secondary h-9" onClick={copyAnswer} type="button">
            <Copy className="h-4 w-4" />
            {copied ? "Copied" : "Copy answer"}
          </button>
          <button className="button-secondary h-9" onClick={exportAnswer} type="button">
            Export
          </button>
          <button className="button-secondary h-9" type="button">
            Helpful
          </button>
          <button className="button-secondary h-9" type="button">
            Not helpful
          </button>
        </div>
      </div>
    </article>
  )
}

function SourcePanel({ answer }: { answer: string }) {
  return (
    <aside className="hidden h-screen w-96 shrink-0 overflow-y-auto border-l border-border bg-surface xl:block">
      <div className="border-b border-border px-5 py-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Source panel
        </p>
        <h2 className="mt-1 text-base font-semibold">Legal basis and provisions</h2>
      </div>
      <div className="space-y-4 p-5">
        <CitationPill title="Legal basis" detail="Review cited documents and article references in the answer." />
        <div className="rounded-lg border border-border bg-background p-4">
          <h3 className="text-sm font-semibold">Retrieved provisions</h3>
          <p className="mt-3 line-clamp-[12] text-sm leading-6 text-muted-foreground">
            {answer}
          </p>
        </div>
        <div className="rounded-lg border border-success/20 bg-success/10 p-4 text-sm leading-6 text-success">
          This system provides legal research support only and does not replace
          professional legal advice.
        </div>
      </div>
    </aside>
  )
}

function MarketingHeader() {
  const { session } = useAuth()
  return (
    <header className="sticky top-0 z-20 border-b border-border bg-surface/95 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link to="/">
          <Logo />
        </Link>
        <nav className="hidden items-center gap-6 text-sm font-medium text-muted-foreground md:flex">
          <a className="hover:text-foreground" href="/#product">Product</a>
          <a className="hover:text-foreground" href="/#sources">Sources</a>
          <a className="hover:text-foreground" href="/#use-cases">Use Cases</a>
          <a className="hover:text-foreground" href="/#reliability">Reliability</a>
          <a className="hover:text-foreground" href="/#security">Security</a>
          <a className="hover:text-foreground" href="/#about">About</a>
        </nav>
        <div className="flex items-center gap-2">
          {session ? (
            <Link className="button-primary h-9" to="/app">Open app</Link>
          ) : (
            <>
              <Link className="button-secondary h-9" to="/signin">Sign in</Link>
              <Link className="button-primary h-9" to="/register">Start researching</Link>
            </>
          )}
        </div>
      </div>
    </header>
  )
}

function Logo({ inverted = false }: { inverted?: boolean }) {
  return (
    <span className="flex items-center gap-3">
      <span className={`flex h-9 w-9 items-center justify-center rounded-md ${inverted ? "bg-white text-navy" : "bg-primary text-white"}`}>
        <Scale className="h-5 w-5" />
      </span>
      <span className="font-semibold">{PRODUCT_NAME}</span>
    </span>
  )
}

function FeatureCard({
  description,
  icon: Icon,
  title
}: {
  description: string
  icon: typeof BookOpenCheck
  title: string
}) {
  return (
    <article className="card p-5 shadow-sm">
      <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary/10 text-primary">
        <Icon className="h-5 w-5" />
      </div>
      <h2 className="mt-5 text-base font-semibold">{title}</h2>
      <p className="mt-3 text-sm leading-6 text-muted-foreground">{description}</p>
    </article>
  )
}

function SidebarSection({ children, title }: { children: React.ReactNode; title: string }) {
  return (
    <section className="mb-6">
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </h2>
      {children}
    </section>
  )
}

function CitationPill({ detail, title }: { detail: string; title: string }) {
  return (
    <div className="rounded-lg border border-border bg-background p-4">
      <p className="text-sm font-semibold">{title}</p>
      <p className="mt-1 text-xs leading-5 text-muted-foreground">{detail}</p>
    </div>
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
    { label: "Total users", value: stats?.totalUsers ?? 0 },
    { label: "Active users", value: stats?.activeUsers ?? 0 },
    { label: "Admin users", value: stats?.adminUsers ?? 0 },
    { label: "Conversations", value: stats?.totalConversations ?? 0 },
    { label: "Messages", value: stats?.totalMessages ?? 0 },
    { label: "Total traces", value: stats?.totalTraces ?? 0 },
    { label: "Traces with errors", value: stats?.tracesWithErrors ?? 0 },
    {
      label: "Insufficient-context traces",
      value: stats?.insufficientContextTraces ?? 0
    }
  ]
}

function formatAdminValue(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "Not configured"
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value)
  }
  return JSON.stringify(value)
}

function formatDate(value: string | undefined): string {
  if (!value) {
    return "Unknown"
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString()
}

function Footer() {
  return (
    <footer className="bg-navy text-white">
      <div className="mx-auto grid max-w-7xl gap-8 px-4 py-10 sm:px-6 lg:grid-cols-[1fr_auto] lg:px-8">
        <div>
          <Logo inverted />
          <p className="mt-4 max-w-xl text-sm leading-6 text-slate-300">
            Secure legal research support for Vietnamese labor-law questions.
          </p>
          <p className="mt-4 text-xs leading-5 text-slate-400">{LEGAL_DISCLAIMER}</p>
        </div>
        <nav className="grid gap-2 text-sm text-slate-300 sm:grid-cols-2">
          <a className="hover:text-white" href="/#product">Product</a>
          <a className="hover:text-white" href="/#sources">Sources</a>
          <a className="hover:text-white" href="/#reliability">Reliability</a>
          <a className="hover:text-white" href="/#security">Security</a>
          <a className="hover:text-white" href="/#about">About</a>
        </nav>
      </div>
    </footer>
  )
}

function promptForTopic(topic: string) {
  const prompts: Record<string, string> = {
    "Employment contracts": "What clauses must be included in an employment contract?",
    Termination: "When can an employee terminate a contract without prior notice?",
    Wages: "What rules govern wage payment under Vietnamese labor law?",
    "Working hours": "What are the legal limits on working hours and overtime?",
    Leave: "What leave rights are recognized under Vietnamese labor law?",
    "Minor workers": "What rules apply to workers under 15?",
    Retirement: "How is retirement age determined under Decree 135/2020/ND-CP?"
  }
  return prompts[topic] ?? topic
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
