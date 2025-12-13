Next.js frontend for InterPyApp (Node.js >= 18.17, Next 14).

## Run locally

```bash
npm install
cp .env.example .env.local   # adjust API URLs if needed
npm run dev   # http://localhost:3000
```

- The UI uses typed API helpers (`frontend/lib/api.ts`) and shows inline validation/server errors via a global error boundary.
- `/upload` responses include `stored_filename`; this is forwarded automatically when training from the UI.

Environment:
- `NEXT_PUBLIC_API_URL` / `API_URL`: point to the backend (e.g., `http://localhost:8000` locally, `http://backend:8000` in Docker).

## Docker

This frontend is included in the root `docker-compose.yml`.

```bash
cd ..
./scripts/docker_build.sh
./scripts/docker_up.sh   # frontend on :3000 (backend on :8000)
```

Tests:
- `npm run lint`
- `npm test` (Node test runner; see `tests/` for examples)
