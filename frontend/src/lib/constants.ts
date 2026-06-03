import type {
  AdminMetric,
  Citation,
  DocumentRecord,
  EvaluationRecord,
  RetrievalRecord,
  SystemLogRecord
} from "@/lib/types"

export const APP_NAME = "Vietnamese Labor Law GraphRAG QA"

export const APP_TAGLINE =
  "Scoped question answering over a six-document Vietnamese labor-law corpus with grounded citations."

export const LEGAL_DISCLAIMER =
  "This system provides corpus-grounded legal information only. It is not legal advice and does not claim correctness beyond the indexed sources."

export const EXAMPLE_QUESTIONS = [
  "How does the Labor Code define an employee?",
  "What conditions apply when a person under 15 works?",
  "How is retirement age guided by Decree 135/2020/ND-CP?",
  "Does a dismissal dispute require mediation before filing a lawsuit?"
]

export const USER_BENEFITS = [
  {
    title: "Scoped corpus",
    description: "Answers are limited to the indexed Vietnamese labor-law sources."
  },
  {
    title: "Grounded citations",
    description: "Responses cite retrieved provisions and refuse when context is insufficient."
  },
  {
    title: "Traceable retrieval",
    description: "Admin views show retrieval, citation, and insufficient-context traces."
  }
]

export const RECENT_CHATS = [
  "Employee definition",
  "Minor worker conditions",
  "Retirement age guidance"
]

export const DEMO_CITATIONS: Citation[] = [
  {
    title: "Labor Code 2019",
    article: "Article 3",
    snippet: "Definition of employee and employer used by the indexed corpus."
  },
  {
    title: "Circular 09/2020/TT-BLDTBXH",
    article: "Article 3",
    snippet: "Guidance related to work by persons under 15 years old."
  },
  {
    title: "Civil Procedure Code 2015, labor-only scope",
    article: "Article 32",
    snippet: "Labor dispute jurisdiction provisions included in the scoped corpus."
  }
]

export const DOCUMENTS: DocumentRecord[] = [
  {
    id: "45-2019-qh14",
    name: "Labor Code 2019",
    type: "Code",
    chunks: 697,
    status: "completed",
    updatedAt: "2026-05-26"
  },
  {
    id: "92-2015-qh13-labor-only",
    name: "Civil Procedure Code 2015, labor-only scope",
    type: "Code",
    chunks: 159,
    status: "completed",
    updatedAt: "2026-05-26"
  },
  {
    id: "nghi-dinh-135-2020-nd-cp",
    name: "Decree 135/2020/ND-CP",
    type: "Decree",
    chunks: 41,
    status: "completed",
    updatedAt: "2026-05-26"
  },
  {
    id: "nghi-dinh-145-2020-nd-cp",
    name: "Decree 145/2020/ND-CP",
    type: "Decree",
    chunks: 520,
    status: "completed",
    updatedAt: "2026-05-26"
  },
  {
    id: "thong-tu-09-2020-tt-bldtbxh",
    name: "Circular 09/2020/TT-BLDTBXH",
    type: "Circular",
    chunks: 73,
    status: "completed",
    updatedAt: "2026-05-26"
  },
  {
    id: "thong-tu-10-2020-tt-bldtbxh",
    name: "Circular 10/2020/TT-BLDTBXH",
    type: "Circular",
    chunks: 66,
    status: "completed",
    updatedAt: "2026-05-26"
  }
]

export const EVALUATION_ROWS: EvaluationRecord[] = [
  {
    id: "def_employee",
    question: "How does the Labor Code define an employee?",
    expectedCitation: "Labor Code 2019, Article 3 clause 1",
    aiAnswer: "The answer must cite the retrieved definition from the indexed corpus.",
    answerCorrect: "yes",
    citationCorrect: "yes",
    hallucination: "no",
    finalScore: 1
  },
  {
    id: "minor_worker_conditions",
    question: "What conditions apply when a person under 15 works?",
    expectedCitation: "Labor Code 2019 and Circular 09/2020/TT-BLDTBXH",
    aiAnswer: "The answer requires multi-source context from the official corpus.",
    answerCorrect: "yes",
    citationCorrect: "yes",
    hallucination: "no",
    finalScore: 1
  },
  {
    id: "ooc_weather",
    question: "What is the weather tomorrow?",
    expectedCitation: "Out of corpus",
    aiAnswer: "The system should refuse because the query is outside the corpus scope.",
    answerCorrect: "yes",
    citationCorrect: "yes",
    hallucination: "no",
    finalScore: 1
  }
]

export const ADMIN_METRICS: AdminMetric[] = [
  {
    label: "Indexed documents",
    value: "6",
    description: "Official thesis corpus",
    tone: "success"
  },
  {
    label: "Indexed chunks",
    value: "1,556",
    description: "Hierarchy-aware enriched chunks",
    tone: "success"
  },
  {
    label: "Benchmark queries",
    value: "100",
    description: "94 in-corpus and 6 out-of-corpus checks",
    tone: "success"
  },
  {
    label: "Graph evidence chunks",
    value: "1,556",
    description: "Neo4j graph build summary",
    tone: "success"
  }
]

export const RETRIEVAL_ROWS: RetrievalRecord[] = [
  {
    id: "R-1021",
    query: "employee definition under Labor Code 2019",
    sources: 5,
    selectedChunk: "45-2019-qh14::article-3",
    relevanceScore: "0.91",
    confidence: "0.84",
    metadata: "Labor Code 2019, Article 3",
    status: "ok"
  },
  {
    id: "R-1022",
    query: "retirement age under Decree 135",
    sources: 6,
    selectedChunk: "nghi-dinh-135-2020-nd-cp::article-4",
    relevanceScore: "0.89",
    confidence: "0.82",
    metadata: "Decree 135/2020/ND-CP",
    status: "ok"
  },
  {
    id: "R-1023",
    query: "weather tomorrow",
    sources: 0,
    selectedChunk: "none",
    relevanceScore: "0.00",
    confidence: "0.00",
    metadata: "Out of corpus",
    status: "warning"
  }
]

export const SYSTEM_LOGS: SystemLogRecord[] = [
  {
    id: "LOG-2301",
    level: "info",
    area: "retrieval",
    message: "Hybrid retrieval completed with grounded context.",
    createdAt: "2026-05-26 12:28"
  },
  {
    id: "LOG-2302",
    level: "info",
    area: "graph",
    message: "Neo4j expansion returned evidence chunks from the official graph.",
    createdAt: "2026-05-26 12:56"
  },
  {
    id: "LOG-2303",
    level: "warning",
    area: "scope",
    message: "Out-of-corpus query returned an insufficient-context response.",
    createdAt: "2026-05-26 13:04"
  }
]
