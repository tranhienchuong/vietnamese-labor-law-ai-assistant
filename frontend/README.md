# Vietnam Labor Law Assistant Frontend

Vite + React + TypeScript frontend for Vietnam Labor Law Assistant.

The frontend is designed for Vercel deployment and uses Supabase Auth with
Google OAuth. The existing Python/FastAPI backend remains the API server.
`vercel.json` rewrites all routes to `index.html` so BrowserRouter pages such
as `/auth/callback`, `/app`, `/account`, and `/admin` work on direct refresh.

## Run locally

```powershell
npm install
npm run dev
```

The local development URL is:

```text
http://localhost:5173
```

Required environment variables:

```env
VITE_SUPABASE_URL=https://your-project-ref.supabase.co
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_BASE_URL=http://localhost:8000
```

Chat requests are sent to the FastAPI backend with the current Supabase access
token in the `Authorization` header.
