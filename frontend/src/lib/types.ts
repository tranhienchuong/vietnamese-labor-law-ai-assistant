export type Citation = {
  title: string
  article: string
  snippet: string
  url?: string
}

export type DocumentRecord = {
  id: string
  name: string
  type: string
  chunks: number
  status: "completed" | "processing" | "pending" | "failed"
  updatedAt: string
}

export type EvaluationRecord = {
  id: string
  question: string
  expectedCitation: string
  aiAnswer: string
  answerCorrect: "yes" | "partial" | "no"
  citationCorrect: "yes" | "partial" | "no"
  hallucination: "yes" | "no"
  finalScore: number
}

export type AdminMetric = {
  label: string
  value: string
  description: string
  tone?: "success" | "warning" | "destructive"
}

export type RetrievalRecord = {
  id: string
  query: string
  sources: number
  selectedChunk: string
  relevanceScore: string
  confidence: string
  metadata: string
  status: "ok" | "warning" | "error"
}

export type SystemLogRecord = {
  id: string
  level: "info" | "warning" | "error"
  area: string
  message: string
  createdAt: string
}

export type UserRole = "user" | "admin"

export type CurrentUser = {
  id: string
  name: string
  email: string
  role: UserRole
  avatarUrl?: string | null
}

export type ConversationSummary = {
  id: string
  title: string
  created_at: string
  updated_at: string
  last_message_at?: string | null
  message_count: number
}
