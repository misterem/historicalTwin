# frontend

The Historical Twin web app. It's a static Next.js site: the browser takes or picks a photo,
shrinks it to 800px, sends it to the matching API (`backend/app.py`), and shows the results.

## Develop

Start the API from the repo root, allowing this dev server's origin:

```bash
PAINTMATCH_ALLOWED_ORIGINS=http://localhost:3000 uv run uvicorn backend.app:app --port 8000
```

Then, in `frontend/`:

```bash
npm install
npm run dev        # http://localhost:3000
npm run lint
```

The camera works on `localhost` and over `https://` only. Browsers block it on plain `http://`
LAN addresses, so to test on a phone you need an https tunnel or a deployed copy. On plain
http the page tells the user to upload a photo instead.

## Configure

| env var | purpose |
|---|---|
| `NEXT_PUBLIC_API_URL` | API base URL (default `http://localhost:8000`). Baked into the bundle at build time, so rebuild after changing it. |

The API must list this site's origin in `PAINTMATCH_ALLOWED_ORIGINS`.

## Build and deploy

```bash
NEXT_PUBLIC_API_URL=https://api.example.com npm run build   # writes static files to out/
```

Upload `out/` to any static host, such as Cloudflare Pages or Vercel.
