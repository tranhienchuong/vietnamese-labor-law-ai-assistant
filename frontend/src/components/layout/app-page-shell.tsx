import { AppHeader } from "@/components/layout/app-header"
import { AppSidebar } from "@/components/layout/app-sidebar"

type AppPageShellProps = {
  title: string
  description: string
  children: React.ReactNode
  actions?: React.ReactNode
}

export function AppPageShell({
  title,
  description,
  children,
  actions
}: AppPageShellProps) {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <AppHeader />
      <div className="mx-auto flex min-h-[calc(100vh-4rem)] max-w-[1600px]">
        <AppSidebar className="hidden lg:flex" />
        <main className="min-w-0 flex-1">
          <div className="border-b border-border bg-surface px-4 py-6 sm:px-6 lg:px-8">
            <div className="mx-auto flex max-w-6xl flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <h1 className="text-2xl font-semibold tracking-normal">{title}</h1>
                <p className="mt-2 max-w-3xl text-sm leading-6 text-muted-foreground">
                  {description}
                </p>
              </div>
              {actions ? <div className="flex flex-wrap gap-2">{actions}</div> : null}
            </div>
          </div>
          <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}
