"use client"

import { AlertCircle, Loader2, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { AssistantMessage } from "@/components/chat/assistant-message"
import { UserMessage } from "@/components/chat/user-message"
import { EXAMPLE_QUESTIONS, LEGAL_DISCLAIMER } from "@/lib/constants"
import { useAutoScroll } from "@/hooks/use-auto-scroll"
import type { ChatMessage } from "@/hooks/use-streaming-chat"

type MessageListProps = {
  messages: ChatMessage[]
  isLoading: boolean
  error?: Error
  onExampleSelect: (question: string) => void
  onRegenerate?: () => void
}

export function MessageList({
  messages,
  isLoading,
  error,
  onExampleSelect,
  onRegenerate
}: MessageListProps) {
  const bottomRef = useAutoScroll(messages.map((message) => message.content).join(""))
  const isEmpty = messages.length === 0

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto flex min-h-full w-full max-w-5xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        {isEmpty ? (
          <div className="grid flex-1 content-center gap-8 py-8">
            <div className="max-w-2xl">
              <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-md bg-primary/10 text-primary">
                <Sparkles className="h-5 w-5" />
              </div>
              <h1 className="text-2xl font-semibold tracking-normal text-foreground sm:text-3xl">
                Bạn muốn hỏi gì về luật lao động?
              </h1>
              <p className="mt-3 max-w-xl text-sm leading-6 text-muted-foreground sm:text-base">
                Hãy đặt câu hỏi về chấm dứt hợp đồng, thời hạn báo trước,
                bồi thường, trợ cấp thôi việc hoặc căn cứ pháp lý liên quan.
              </p>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              {EXAMPLE_QUESTIONS.map((question) => (
                <button
                  className="rounded-lg border border-border bg-surface p-4 text-left text-sm leading-6 shadow-sm transition-colors hover:border-primary/50 hover:bg-primary/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  key={question}
                  onClick={() => onExampleSelect(question)}
                  type="button"
                >
                  {question}
                </button>
              ))}
            </div>

            <div className="rounded-md border border-warning/40 bg-warning/10 px-4 py-3 text-sm leading-6 text-warning-foreground">
              {LEGAL_DISCLAIMER}
            </div>
          </div>
        ) : (
          messages.map((message, index) => {
            const isLatest = index === messages.length - 1
            if (message.role === "user") {
              return <UserMessage content={message.content} key={message.id} />
            }
            if (!message.content && isLatest && isLoading) {
              return (
                <div
                  className="flex items-center gap-2 rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted-foreground"
                  key={message.id}
                >
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Đang chuẩn bị câu trả lời có căn cứ pháp lý...
                </div>
              )
            }
            return (
              <AssistantMessage
                content={message.content}
                isLatest={isLatest}
                key={message.id}
                onRegenerate={onRegenerate}
              />
            )
          })
        )}

        {isLoading && messages[messages.length - 1]?.role === "user" ? (
          <div className="flex items-center gap-2 rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Đang chuẩn bị câu trả lời có căn cứ pháp lý...
          </div>
        ) : null}

        {error ? (
          <div className="flex items-start gap-3 rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              Không thể kết nối tới máy chủ AI. Vui lòng thử lại sau.
              <Button
                className="ml-2 h-auto px-0 py-0 text-destructive underline"
                onClick={onRegenerate}
                type="button"
                variant="ghost"
              >
                Thử lại
              </Button>
            </div>
          </div>
        ) : null}

        <div ref={bottomRef} />
      </div>
    </div>
  )
}
