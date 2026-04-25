import { User } from "lucide-react"

type UserMessageProps = {
  content: string
}

export function UserMessage({ content }: UserMessageProps) {
  return (
    <article className="flex justify-end">
      <div className="flex max-w-[82%] items-start gap-3 sm:max-w-[70%]">
        <div className="rounded-lg bg-primary px-4 py-3 text-sm leading-6 text-primary-foreground shadow-sm">
          {content}
        </div>
        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground">
          <User className="h-4 w-4" />
        </div>
      </div>
    </article>
  )
}
