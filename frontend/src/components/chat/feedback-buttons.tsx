"use client"

import { useState } from "react"
import { AlertTriangle, Check, FileWarning, HelpCircle, ThumbsDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const feedbackOptions = [
  { id: "helpful", label: "Đúng", icon: Check },
  { id: "not-helpful", label: "Sai", icon: ThumbsDown },
  { id: "wrong-citation", label: "Sai nguồn", icon: FileWarning },
  { id: "missing-info", label: "Thiếu ý", icon: HelpCircle },
  { id: "hallucination", label: "Bịa nguồn", icon: AlertTriangle }
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
