"use client"

import type { ChangeEvent, FormEvent, KeyboardEvent } from "react"
import { Send, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { LEGAL_DISCLAIMER } from "@/lib/constants"

type ChatInputProps = {
  input: string
  isLoading: boolean
  onInputChange: (event: ChangeEvent<HTMLTextAreaElement>) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  onStop: () => void
}

export function ChatInput({
  input,
  isLoading,
  onInputChange,
  onSubmit,
  onStop
}: ChatInputProps) {
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  return (
    <div className="border-t border-border bg-background/95 px-4 py-4 backdrop-blur sm:px-6">
      <form className="mx-auto max-w-5xl" onSubmit={onSubmit}>
        <div className="rounded-lg border border-border bg-surface p-2 shadow-soft">
          <Textarea
            aria-label="Câu hỏi pháp lý"
            maxLength={4000}
            onChange={onInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Nhập câu hỏi về chấm dứt hợp đồng, báo trước, bồi thường hoặc trợ cấp..."
            value={input}
          />
          <div className="flex flex-col gap-3 px-1 pt-2 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-xs leading-5 text-muted-foreground">
              {LEGAL_DISCLAIMER}
            </p>
            <div className="flex justify-end">
              {isLoading ? (
                <Button onClick={onStop} type="button" variant="secondary">
                  <Square className="h-4 w-4" />
                  Dừng
                </Button>
              ) : (
                <Button disabled={!input.trim()} type="submit">
                  <Send className="h-4 w-4" />
                  Gửi
                </Button>
              )}
            </div>
          </div>
        </div>
      </form>
    </div>
  )
}
