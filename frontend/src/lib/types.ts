export type Citation = {
  title: string
  article: string
  snippet: string
  relevance?: string
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
