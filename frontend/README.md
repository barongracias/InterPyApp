Next.js frontend for InterPyApp.

## Run locally

```bash
npm install
cp .env.example .env.local   # adjust API URLs if needed
npm install
npm run dev   # http://localhost:3000
```

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
