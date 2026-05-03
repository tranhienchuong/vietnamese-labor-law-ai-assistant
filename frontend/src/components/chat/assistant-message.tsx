"use client"

import { useMemo, useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Bot, Copy, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { CitationList } from "@/components/chat/citation-list"
import { FeedbackButtons } from "@/components/chat/feedback-buttons"
import { DEMO_CITATIONS } from "@/lib/constants"
import type { Citation } from "@/lib/types"

type AssistantMessageProps = {
  content: string
  isLatest?: boolean
  onRegenerate?: () => void
}

function citationsForContent(content: string): Citation[] {
  const normalized = content.toLowerCase()
  const matches = DEMO_CITATIONS.filter((citation) =>
    normalized.includes(citation.article.toLowerCase())
  )

  if (matches.length > 0) {
    return matches
  }

  if (normalized.includes("căn cứ") || normalized.includes("nguồn")) {
    return DEMO_CITATIONS.slice(0, 2)
  }

  return []
}

export function AssistantMessage({
  content,
  isLatest,
  onRegenerate
}: AssistantMessageProps) {
  const [copied, setCopied] = useState(false)
  const citations = useMemo(() => citationsForContent(content), [content])

  async function copyAnswer() {
    await navigator.clipboard.writeText(content)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1600)
  }

  return (
    <article className="flex justify-start">
      <div className="flex w-full max-w-3xl items-start gap-3">
        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-muted text-primary">
          <Bot className="h-4 w-4" />
        </div>

        <div className="min-w-0 flex-1 space-y-4">
          <div className="rounded-lg border border-border bg-surface px-4 py-4 shadow-sm">
            <ReactMarkdown
              className="message-markdown text-sm text-foreground sm:text-[15px]"
              disallowedElements={["script", "iframe"]}
              remarkPlugins={[remarkGfm]}
            >
              {content}
            </ReactMarkdown>
          </div>

          <CitationList citations={citations} />

          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={copyAnswer} size="sm" type="button" variant="outline">
              <Copy className="h-3.5 w-3.5" />
              {copied ? "Đã sao chép" : "Sao chép"}
            </Button>
            {isLatest && onRegenerate ? (
              <Button onClick={onRegenerate} size="sm" type="button" variant="outline">
                <RefreshCw className="h-3.5 w-3.5" />
                Tạo lại
              </Button>
            ) : null}
            <FeedbackButtons />
          </div>
        </div>
      </div>
    </article>
  )
}
