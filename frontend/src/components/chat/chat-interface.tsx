"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { X } from "lucide-react"
import { AppHeader } from "@/components/layout/app-header"
import { AppSidebar } from "@/components/layout/app-sidebar"
import { Button } from "@/components/ui/button"
import { ChatInput } from "@/components/chat/chat-input"
import { MessageList } from "@/components/chat/message-list"
import { useStreamingChat, type ChatMessage } from "@/hooks/use-streaming-chat"
import type { ConversationSummary } from "@/lib/types"

export function ChatInterface() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [conversations, setConversations] = useState<ConversationSummary[]>([])

  const loadConversations = useCallback(async () => {
    const response = await fetch("/api/conversations", { cache: "no-store" })
    if (!response.ok) return
    const payload = (await response.json()) as {
      conversations?: ConversationSummary[]
    }
    setConversations(payload.conversations ?? [])
  }, [])

  const chatBody = useMemo(
    () => ({
      mode: "legal_qa",
      language: "vi",
      includeCitations: true,
      ...(conversationId ? { conversationId } : {})
    }),
    [conversationId]
  )

  const handleResponseMetadata = useCallback(
    ({ conversationId: nextConversationId }: { conversationId?: string }) => {
      if (nextConversationId) {
        setConversationId(nextConversationId)
      }
      void loadConversations()
    },
    [loadConversations]
  )

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
    body: chatBody,
    onResponseMetadata: handleResponseMetadata
  })

  useEffect(() => {
    const question = new URLSearchParams(window.location.search).get("question")
    if (question) {
      setInput(question)
    }
  }, [setInput])

  useEffect(() => {
    void loadConversations()
  }, [loadConversations])

  function startNewChat() {
    stop()
    setConversationId(null)
    setInput("")
    setMessages([])
    setSidebarOpen(false)
  }

  async function selectConversation(nextConversationId: string) {
    stop()
    const response = await fetch(`/api/conversations/${nextConversationId}`, {
      cache: "no-store"
    })
    if (!response.ok) return

    const payload = (await response.json()) as {
      messages?: Array<ChatMessage & { created_at?: string }>
    }
    const nextMessages =
      payload.messages
        ?.filter((message) => message.role === "user" || message.role === "assistant")
        .map((message) => ({
          id: message.id,
          role: message.role,
          content: message.content
        })) ?? []

    setConversationId(nextConversationId)
    setMessages(nextMessages)
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
          activeConversationId={conversationId}
          className="hidden lg:flex"
          conversations={conversations}
          onConversationSelect={selectConversation}
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
                <span className="text-sm font-semibold">Điều hướng</span>
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
                activeConversationId={conversationId}
                className="h-[calc(100%-4rem)] w-full border-r-0"
                conversations={conversations}
                onConversationSelect={selectConversation}
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
      </div>
    </div>
  )
}
