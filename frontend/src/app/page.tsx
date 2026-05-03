import Link from "next/link"
import {
  ArrowRight,
  BookOpenCheck,
  CheckCircle2,
  MessageSquareText,
  Scale,
  SearchCheck
} from "lucide-react"
import { AppHeader } from "@/components/layout/app-header"
import { Button } from "@/components/ui/button"
import {
  APP_NAME,
  APP_TAGLINE,
  EXAMPLE_QUESTIONS,
  LEGAL_DISCLAIMER,
  USER_BENEFITS
} from "@/lib/constants"

const benefitIcons = [MessageSquareText, BookOpenCheck, SearchCheck]

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <AppHeader />
      <main>
        <section className="border-b border-border bg-surface">
          <div className="mx-auto flex min-h-[calc(100vh-10rem)] max-w-7xl flex-col justify-center px-4 py-12 sm:px-6 lg:px-8">
            <div className="max-w-3xl">
              <div className="mb-5 flex items-center gap-3">
                <span className="flex h-11 w-11 items-center justify-center rounded-md bg-primary text-primary-foreground">
                  <Scale className="h-6 w-6" />
                </span>
                <span className="text-sm font-semibold text-muted-foreground">
                  Hỏi đáp pháp lý lao động
                </span>
              </div>

              <h1 className="max-w-4xl text-3xl font-semibold tracking-normal text-foreground sm:text-4xl lg:text-5xl">
                {APP_NAME}
              </h1>
              <p className="mt-5 max-w-2xl text-base leading-7 text-muted-foreground sm:text-lg">
                {APP_TAGLINE}
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Button asChild size="lg">
                  <Link href="/chat">
                    Bắt đầu hỏi
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
              </div>
            </div>

            <div className="mt-10 grid gap-3 lg:grid-cols-3">
              {EXAMPLE_QUESTIONS.slice(0, 3).map((question) => (
                <Link
                  className="rounded-lg border border-border bg-background px-4 py-4 text-sm leading-6 shadow-sm transition-colors hover:border-primary/50 hover:bg-primary/5"
                  href={`/chat?question=${encodeURIComponent(question)}`}
                  key={question}
                >
                  {question}
                </Link>
              ))}
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
          <div className="grid gap-4 md:grid-cols-3">
            {USER_BENEFITS.map((item, index) => {
              const Icon = benefitIcons[index] ?? CheckCircle2
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
          <div className="mx-auto max-w-7xl px-4 py-6 text-sm leading-6 text-muted-foreground sm:px-6 lg:px-8">
            {LEGAL_DISCLAIMER}
          </div>
        </section>
      </main>
    </div>
  )
}
