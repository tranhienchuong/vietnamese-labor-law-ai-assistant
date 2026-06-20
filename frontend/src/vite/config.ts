export const PRODUCT_NAME = "Vietnam Labor Law Assistant"

export const LEGAL_DISCLAIMER =
  "This system provides legal research support only and does not replace professional legal advice."

export const supabaseUrl = import.meta.env.VITE_SUPABASE_URL?.trim() ?? ""
export const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY?.trim() ?? ""
export const apiBaseUrl =
  import.meta.env.VITE_API_BASE_URL?.trim().replace(/\/$/, "") ?? "http://localhost:8000"

export const missingSupabaseConfig = !supabaseUrl || !supabaseAnonKey
