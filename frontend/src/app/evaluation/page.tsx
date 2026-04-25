import { Download, Filter, Upload } from "lucide-react"
import { AppPageShell } from "@/components/layout/app-page-shell"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { EVALUATION_ROWS } from "@/lib/constants"
import type { EvaluationRecord } from "@/lib/types"

function scoreClass(score: number) {
  if (score >= 8) return "text-success"
  if (score >= 6) return "text-warning-foreground"
  return "text-destructive"
}

function outcomeBadge(value: EvaluationRecord["answerCorrect"]) {
  if (value === "yes") return <Badge variant="success">Yes</Badge>
  if (value === "partial") return <Badge variant="warning">Partial</Badge>
  return <Badge variant="destructive">No</Badge>
}

export default function EvaluationPage() {
  return (
    <AppPageShell
      actions={
        <>
          <Button variant="outline">
            <Upload className="h-4 w-4" />
            Import
          </Button>
          <Button>
            <Download className="h-4 w-4" />
            Export CSV
          </Button>
        </>
      }
      description="Theo dõi benchmark, citation correctness, hallucination và điểm chất lượng câu trả lời."
      title="Evaluation"
    >
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <Button variant="outline">
          <Filter className="h-4 w-4" />
          Điểm thấp
        </Button>
        <Button variant="outline">Hallucination</Button>
        <Button variant="outline">Citation sai</Button>
        <Badge variant="secondary">{EVALUATION_ROWS.length} cases</Badge>
      </div>

      <div className="overflow-hidden rounded-lg border border-border bg-surface shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[980px] text-left text-sm">
            <thead className="bg-muted text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-4 py-3 font-semibold">ID</th>
                <th className="px-4 py-3 font-semibold">Question</th>
                <th className="px-4 py-3 font-semibold">Expected citation</th>
                <th className="px-4 py-3 font-semibold">Answer</th>
                <th className="px-4 py-3 font-semibold">Citation</th>
                <th className="px-4 py-3 font-semibold">Hallucination</th>
                <th className="px-4 py-3 font-semibold">Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {EVALUATION_ROWS.map((row) => (
                <tr className="align-top hover:bg-muted/50" key={row.id}>
                  <td className="px-4 py-4 font-medium">{row.id}</td>
                  <td className="max-w-xs px-4 py-4 leading-6">{row.question}</td>
                  <td className="px-4 py-4 text-muted-foreground">
                    {row.expectedCitation}
                  </td>
                  <td className="px-4 py-4">{outcomeBadge(row.answerCorrect)}</td>
                  <td className="px-4 py-4">{outcomeBadge(row.citationCorrect)}</td>
                  <td className="px-4 py-4">
                    {row.hallucination === "yes" ? (
                      <Badge variant="destructive">Yes</Badge>
                    ) : (
                      <Badge variant="success">No</Badge>
                    )}
                  </td>
                  <td className={`px-4 py-4 font-semibold ${scoreClass(row.finalScore)}`}>
                    {row.finalScore.toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </AppPageShell>
  )
}
