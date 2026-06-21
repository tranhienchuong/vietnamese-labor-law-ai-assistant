export const PRODUCT_NAME = "Trợ lý Luật lao động Việt Nam"

export const LEGAL_DISCLAIMER =
  "Hệ thống chỉ hỗ trợ tra cứu pháp luật và không thay thế tư vấn pháp lý chuyên nghiệp."

export const supabaseUrl = import.meta.env.VITE_SUPABASE_URL?.trim() ?? ""
export const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY?.trim() ?? ""
export const apiBaseUrl =
  import.meta.env.VITE_API_BASE_URL?.trim().replace(/\/$/, "") ?? "http://localhost:8000"

export const missingSupabaseConfig = !supabaseUrl || !supabaseAnonKey
