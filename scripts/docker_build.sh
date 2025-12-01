#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND_TAG="interpyapp-backend:latest"
FRONTEND_TAG="interpyapp-frontend:latest"

cd "$ROOT_DIR"

echo "Building backend image: $BACKEND_TAG"
docker build -t "$BACKEND_TAG" -f backend/Dockerfile backend

echo "Building frontend image: $FRONTEND_TAG"
docker build -t "$FRONTEND_TAG" -f frontend/Dockerfile frontend

echo "Build complete."
