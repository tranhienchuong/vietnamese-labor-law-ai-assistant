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

export type CurrentUserResponse = {
  user?: CurrentUser
}

export type LoginResponse = {
  user: CurrentUser
}

export type ChatMessage = {
  id: string
  role: "user" | "assistant"
  content: string
}

export type ChatRequestBody = Record<string, unknown> & {
  messages: Array<Pick<ChatMessage, "role" | "content">>
}

export type ConversationSummary = {
  id: string
  title: string
  created_at: string
  updated_at: string
  last_message_at?: string | null
  message_count: number
}

export type ConversationMessage = ChatMessage & {
  conversation_id?: string
  created_at?: string
}

export type ConversationsResponse = {
  conversations: ConversationSummary[]
}

export type ConversationCreateResponse = {
  conversation: ConversationSummary
}

export type ConversationDetailResponse = {
  conversation: ConversationSummary
  messages: ConversationMessage[]
}

export type ConversationMessagesResponse = {
  messages: ConversationMessage[]
}

export type AdminStatsResponse = {
  user: CurrentUser
  stats: {
    totalUsers: number
    activeUsers: number
    adminUsers: number
    totalConversations: number
    totalMessages: number
    activeSessions: number
    totalTraces: number
    tracesWithErrors: number
    insufficientContextTraces: number
  }
  runtime: {
    appEnv: string
    databasePath: string
    qdrantCollection: string
    retrieverRecordSource: string
    indexPath: string
    rerankerEnabled: boolean
    queryRouterEnabled: boolean
    llmProvider: string
    groqModel: string
    benchmarkPath: string
    benchmarkMetricMode: string
    citationValidationMode: string
  }
}

export type AdminCheckStatus =
  | "ok"
  | "error"
  | "missing"
  | "configured"
  | "local"

export type AdminHealthResponse = {
  status: "ok" | "degraded"
  checks: {
    database: {
      status: AdminCheckStatus
      message: string
    }
    settings: {
      status: AdminCheckStatus
      message: string
    }
    index: {
      status: AdminCheckStatus
      message: string
      path: string
    }
    qdrantConfig: {
      status: AdminCheckStatus
      collection: string
      usesCloud: boolean
    }
    llmConfig: {
      status: AdminCheckStatus
      provider: string
      model: string
    }
  }
}

export type AdminRetrievalConfigResponse = {
  qdrantCollection: string
  retrieverRecordSource: string
  indexPath: string
  rerankerModel: string
  rerankerEnabled: boolean
  rerankerTopN: number
  queryRouterEnabled: boolean
  queryRouterProvider: string
  queryRouterModel: string
  queryRouterFallbackToHeuristic: boolean
  embeddingProvider: string
  denseModel: string
  benchmarkPath: string
  benchmarkMetricMode: string
  citationValidationMode: string
  legalGraphEnabled: boolean
  legalGraphBackend: string
  legalGraphExpansionDepth: number
  legalGraphMaxExpandedChunks: number
}

export type AdminTraceSummary = {
  id: string
  requestId?: string | null
  userId: string
  conversationId?: string | null
  messageId?: string | null
  question: string
  provider?: string | null
  model?: string | null
  retrieveOnly: boolean
  insufficientContext: boolean
  latencyMs?: number | null
  retrievalLatencyMs?: number | null
  generationLatencyMs?: number | null
  citationCount: number
  selectedContextCount: number
  error?: string | null
  createdAt: string
}

export type AdminTraceDetail = AdminTraceSummary & {
  intent: Record<string, unknown>
  retrievedHits: Array<Record<string, unknown>>
  selectedContexts: Array<Record<string, unknown>>
  citations: {
    legal_basis?: string[]
    evidence_quotes?: Array<{
      citation: string
      quote: string
    }>
  }
}

export type AdminTracesResponse = {
  traces: AdminTraceSummary[]
}

export type AdminTraceDetailResponse = {
  trace: AdminTraceDetail
}
