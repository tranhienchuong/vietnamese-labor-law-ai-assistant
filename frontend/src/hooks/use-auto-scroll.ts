"use client"

import { useEffect, useRef } from "react"

export function useAutoScroll<TDependency>(dependency: TDependency) {
  const ref = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [dependency])

  return ref
}
