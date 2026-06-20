import { createClient } from "@supabase/supabase-js"
import { supabaseAnonKey, supabaseUrl } from "./config"

const fallbackUrl = "https://example.supabase.co"
const fallbackAnonKey = "missing-supabase-anon-key"

export const supabase = createClient(
  supabaseUrl || fallbackUrl,
  supabaseAnonKey || fallbackAnonKey,
  {
    auth: {
      flowType: "pkce",
      persistSession: true,
      autoRefreshToken: true,
      detectSessionInUrl: true
    }
  }
)
