"use client"

import { useEffect, useState } from "react"
import { X } from "lucide-react"
import { AppHeader } from "@/components/layout/app-header"
import { AppSidebar } from "@/components/layout/app-sidebar"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ChatInput } from "@/components/chat/chat-input"
import { MessageList } from "@/components/chat/message-list"
import { useStreamingChat } from "@/hooks/use-streaming-chat"

export function ChatInterface() {
  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    error,
    stop,
    reload,
    setInput,
    setMessages
  } = useStreamingChat({
    api: "/api/chat",
    body: {
      mode: "legal_qa",
      language: "vi",
      includeCitations: true
    }
  })

  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    const question = new URLSearchParams(window.location.search).get("question")
    if (question) {
      setInput(question)
    }
  }, [setInput])

  function startNewChat() {
    stop()
    setInput("")
    setMessages([])
    setSidebarOpen(false)
  }

  function selectExample(question: string) {
    setInput(question)
    setSidebarOpen(false)
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <AppHeader onMenuClick={() => setSidebarOpen(true)} />

      <div className="mx-auto flex h-[calc(100vh-4rem)] max-w-[1600px]">
        <AppSidebar
          className="hidden lg:flex"
          onExampleSelect={selectExample}
          onNewChat={startNewChat}
        />

        {sidebarOpen ? (
          <div className="fixed inset-0 z-50 lg:hidden">
            <button
              aria-label="Đóng điều hướng"
              className="absolute inset-0 bg-foreground/30"
              onClick={() => setSidebarOpen(false)}
              type="button"
            />
            <div className="relative h-full w-[min(88vw,20rem)] bg-surface shadow-soft">
              <div className="flex h-16 items-center justify-between border-b border-border px-4">
                <Badge variant="secondary">Menu</Badge>
                <Button
                  aria-label="Đóng điều hướng"
                  onClick={() => setSidebarOpen(false)}
                  size="iconSm"
                  type="button"
                  variant="ghost"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <AppSidebar
                className="h-[calc(100%-4rem)] w-full border-r-0"
                onExampleSelect={selectExample}
                onNewChat={startNewChat}
              />
            </div>
          </div>
        ) : null}

        <main className="flex min-w-0 flex-1 flex-col">
          <MessageList
            error={error}
            isLoading={isLoading}
            messages={messages}
            onExampleSelect={selectExample}
            onRegenerate={() => void reload()}
          />
          <ChatInput
            input={input}
            isLoading={isLoading}
            onInputChange={handleInputChange}
            onStop={stop}
            onSubmit={handleSubmit}
          />
        </main>

        <aside className="hidden w-80 shrink-0 border-l border-border bg-surface xl:block">
          <div className="space-y-5 p-5">
            <div>
              <h2 className="text-sm font-semibold">Trạng thái truy xuất</h2>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">
                Backend thật có thể trả thêm confidence, legal basis và metadata
                nguồn để thay thế dữ liệu demo.
              </p>
            </div>
            <div className="grid gap-3">
              <StatusRow label="Nguồn đang index" value="2" />
              <StatusRow label="Chế độ trả lời" value="Có căn cứ" />
              <StatusRow label="Streaming" value="Bật" />
            </div>
            <div className="rounded-md border border-warning/40 bg-warning/10 px-3 py-3 text-sm leading-6 text-warning-foreground">
              Khi câu hỏi thiếu dữ kiện, hệ thống nên yêu cầu bổ sung thay vì
              suy đoán kết luận pháp lý.
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded-md border border-border bg-background px-3 py-2 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  )
}
