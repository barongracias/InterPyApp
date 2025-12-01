## InterPyApp Frontend

Next.js UI for interacting with the InterPyApp backend.

### Run locally

```bash
cd frontend
npm install
cp .env.example .env.local   # adjust API URLs if needed
npm run dev   # http://localhost:3000
```

### Environment
- `NEXT_PUBLIC_API_URL` / `API_URL`: point to the backend (e.g., `http://localhost:8000` locally, `http://backend:8000` in Docker).

### Tests

```bash
npm run lint
npm test   # runs Node test runner (see tests/ for examples)
```

### Docker
- Included in the root `docker-compose.yml`; build/run via `./scripts/docker_build.sh` and `./scripts/docker_up.sh`.
