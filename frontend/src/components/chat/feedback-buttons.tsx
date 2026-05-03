"use client"

import { useState } from "react"
import { ThumbsDown, ThumbsUp } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const feedbackOptions = [
  { id: "helpful", label: "Hữu ích", icon: ThumbsUp },
  { id: "not-helpful", label: "Chưa đúng", icon: ThumbsDown }
]

export function FeedbackButtons() {
  const [selected, setSelected] = useState<string | null>(null)

  return (
    <div className="flex flex-wrap gap-2">
      {feedbackOptions.map((option) => {
        const Icon = option.icon
        const isSelected = selected === option.id
        return (
          <Button
            className={cn(isSelected && "border-primary bg-primary/10 text-primary")}
            key={option.id}
            onClick={() => setSelected(option.id)}
            size="sm"
            type="button"
            variant="outline"
          >
            <Icon className="h-3.5 w-3.5" />
            {option.label}
          </Button>
        )
      })}
    </div>
  )
}
