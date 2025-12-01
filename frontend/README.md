Next.js frontend for InterPyApp.

## Run locally

```bash
npm install
npm run dev   # http://localhost:3000
```

## Docker

This frontend is included in the root `docker-compose.yml`.

```bash
cd ..
./scripts/docker_build.sh
./scripts/docker_up.sh   # frontend on :3000 (backend on :8000)
```
