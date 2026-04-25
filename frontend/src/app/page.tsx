import Link from "next/link"
import {
  ArrowRight,
  BarChart3,
  BookOpenCheck,
  FileText,
  MessageSquareText,
  ShieldCheck
} from "lucide-react"
import { AppHeader } from "@/components/layout/app-header"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  APP_NAME,
  APP_TAGLINE,
  EXAMPLE_QUESTIONS,
  LEGAL_DISCLAIMER
} from "@/lib/constants"

const capabilities = [
  {
    title: "Chat có streaming",
    description: "Luồng trả lời xuất hiện dần, phù hợp trải nghiệm AI assistant.",
    icon: MessageSquareText
  },
  {
    title: "Nguồn pháp lý",
    description: "Thiết kế sẵn citation cards cho luật, điều khoản và đoạn trích.",
    icon: BookOpenCheck
  },
  {
    title: "Evaluation",
    description: "Có nền tảng để rà soát benchmark, hallucination và điểm cuối.",
    icon: BarChart3
  }
]

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <AppHeader />
      <main>
        <section className="border-b border-border bg-surface">
          <div className="mx-auto grid max-w-7xl gap-10 px-4 py-12 sm:px-6 lg:grid-cols-[1.05fr_0.95fr] lg:px-8 lg:py-16">
            <div className="flex flex-col justify-center">
              <div className="mb-5 flex flex-wrap items-center gap-2">
                <Badge variant="success">RAG enabled</Badge>
                <Badge variant="outline">Vietnam labor law</Badge>
              </div>
              <h1 className="max-w-3xl text-3xl font-semibold tracking-normal text-foreground sm:text-4xl lg:text-5xl">
                {APP_NAME}
              </h1>
              <p className="mt-5 max-w-2xl text-base leading-7 text-muted-foreground sm:text-lg">
                {APP_TAGLINE}
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Button asChild size="lg">
                  <Link href="/chat">
                    Bắt đầu chat
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
                <Button asChild size="lg" variant="outline">
                  <Link href="/evaluation">
                    Mở evaluation
                    <BarChart3 className="h-4 w-4" />
                  </Link>
                </Button>
              </div>
            </div>

            <div className="rounded-lg border border-border bg-background p-4 shadow-soft">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold">Câu hỏi mẫu</p>
                  <p className="text-sm text-muted-foreground">
                    Mở nhanh trong giao diện chat
                  </p>
                </div>
                <ShieldCheck className="h-5 w-5 text-success" />
              </div>
              <div className="space-y-3">
                {EXAMPLE_QUESTIONS.slice(0, 3).map((question) => (
                  <Link
                    className="block rounded-md border border-border bg-surface px-4 py-3 text-sm leading-6 transition-colors hover:border-primary/50 hover:bg-primary/5"
                    href={`/chat?question=${encodeURIComponent(question)}`}
                    key={question}
                  >
                    {question}
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
          <div className="grid gap-4 md:grid-cols-3">
            {capabilities.map((item) => {
              const Icon = item.icon
              return (
                <div
                  className="rounded-lg border border-border bg-surface p-5 shadow-sm"
                  key={item.title}
                >
                  <Icon className="h-5 w-5 text-primary" />
                  <h2 className="mt-4 text-base font-semibold">{item.title}</h2>
                  <p className="mt-2 text-sm leading-6 text-muted-foreground">
                    {item.description}
                  </p>
                </div>
              )
            })}
          </div>
        </section>

        <section className="border-t border-border bg-surface">
          <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-6 text-sm leading-6 text-muted-foreground sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
            <p>{LEGAL_DISCLAIMER}</p>
            <div className="flex gap-3">
              <Button asChild variant="outline">
                <Link href="/documents">
                  <FileText className="h-4 w-4" />
                  Documents
                </Link>
              </Button>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
