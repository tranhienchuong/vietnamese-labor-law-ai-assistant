import { ChevronDown, ExternalLink } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import type { Citation } from "@/lib/types"

type CitationListProps = {
  citations: Citation[]
}

export function CitationList({ citations }: CitationListProps) {
  if (citations.length === 0) {
    return (
      <div className="rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-warning-foreground">
        Không tìm thấy nguồn pháp lý đủ chắc chắn cho câu hỏi này.
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Nguồn tham khảo</h3>
        <Badge variant="secondary">{citations.length} nguồn</Badge>
      </div>
      <div className="space-y-2">
        {citations.map((citation, index) => (
          <details
            className="group rounded-md border border-border bg-background"
            key={`${citation.title}-${citation.article}`}
            open={index === 0}
          >
            <summary className="flex cursor-pointer list-none items-center gap-3 px-3 py-2 text-sm font-medium">
              <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-180" />
              <span className="min-w-0 flex-1 truncate">
                {citation.title} · {citation.article}
              </span>
              {citation.relevance ? (
                <Badge variant={citation.relevance === "Cao" ? "success" : "outline"}>
                  {citation.relevance}
                </Badge>
              ) : null}
            </summary>
            <div className="border-t border-border px-3 py-3 text-sm leading-6 text-muted-foreground">
              <p>{citation.snippet}</p>
              {citation.url ? (
                <a
                  className="mt-2 inline-flex items-center gap-1 text-primary hover:underline"
                  href={citation.url}
                  rel="noreferrer"
                  target="_blank"
                >
                  Mở nguồn <ExternalLink className="h-3.5 w-3.5" />
                </a>
              ) : null}
            </div>
          </details>
        ))}
      </div>
    </div>
  )
}
