"use client"

import { useCallback, useRef, useState } from "react"
import type { ChangeEvent, FormEvent } from "react"
import { sendChatRequest } from "@/lib/api/chat"
import type { ChatMessage } from "@/lib/types"

export type { ChatMessage }

type UseStreamingChatOptions = {
  api: string
  body?: Record<string, unknown>
  onResponseMetadata?: (metadata: { conversationId?: string }) => void
}

function createId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function useStreamingChat({
  api,
  body,
  onResponseMetadata
}: UseStreamingChatOptions) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<Error | undefined>()
  const abortRef = useRef<AbortController | null>(null)

  const stop = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setIsLoading(false)
  }, [])

  const streamAssistantResponse = useCallback(
    async (nextMessages: ChatMessage[]) => {
      const controller = new AbortController()
      abortRef.current = controller
      setIsLoading(true)
      setError(undefined)

      const assistantId = createId()
      setMessages([
        ...nextMessages,
        {
          id: assistantId,
          role: "assistant",
          content: ""
        }
      ])

      try {
        const response = await sendChatRequest(
          {
            messages: nextMessages.map(({ role, content }) => ({ role, content })),
            ...body
          },
          controller.signal,
          api
        )

        if (!response.ok || !response.body) {
          throw new Error(`Chat request failed with ${response.status}`)
        }

        onResponseMetadata?.({
          conversationId: response.headers.get("X-Conversation-Id") || undefined
        })

        const reader = response.body.getReader()
        const decoder = new TextDecoder()

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + chunk }
                : message
            )
          )
        }
      } catch (caughtError) {
        if ((caughtError as Error).name !== "AbortError") {
          setError(caughtError as Error)
          setMessages(nextMessages)
        }
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null
        }
        setIsLoading(false)
      }
    },
    [api, body, onResponseMetadata]
  )

  const handleInputChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => {
      setInput(event.target.value)
    },
    []
  )

  const handleSubmit = useCallback(
    async (event?: FormEvent<HTMLFormElement>) => {
      event?.preventDefault()
      const trimmed = input.trim()
      if (!trimmed || isLoading) return

      const nextMessages: ChatMessage[] = [
        ...messages,
        {
          id: createId(),
          role: "user",
          content: trimmed
        }
      ]

      setInput("")
      await streamAssistantResponse(nextMessages)
    },
    [input, isLoading, messages, streamAssistantResponse]
  )

  const reload = useCallback(async () => {
    if (isLoading || messages.length === 0) return

    const lastUserIndex = [...messages]
      .reverse()
      .findIndex((message) => message.role === "user")
    if (lastUserIndex === -1) return

    const userIndex = messages.length - 1 - lastUserIndex
    const nextMessages = messages.slice(0, userIndex + 1)
    await streamAssistantResponse(nextMessages)
  }, [isLoading, messages, streamAssistantResponse])

  return {
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
  }
}
