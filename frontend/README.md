# Vietnamese Labor Law AI Frontend

Next.js interface for the Vietnamese Labor Law AI Assistant.

## Run locally

```powershell
npm install
npm run dev
```

The chat route proxies to `BACKEND_URL` when configured. Without `BACKEND_URL`, it returns a local streaming demo answer so the UI can be reviewed immediately.

```env
BACKEND_URL=http://localhost:8000
```
