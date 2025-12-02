#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND_TAG="interpyapp-backend:latest"
FRONTEND_TAG="interpyapp-frontend:latest"
# Force amd64 builds so TensorFlow wheels resolve on Apple Silicon/M-series hosts.
PLATFORM="linux/amd64"

cd "$ROOT_DIR"

echo "Building backend image: $BACKEND_TAG"
docker build --platform "$PLATFORM" -t "$BACKEND_TAG" -f backend/Dockerfile backend

echo "Building frontend image: $FRONTEND_TAG"
docker build --platform "$PLATFORM" -t "$FRONTEND_TAG" -f frontend/Dockerfile frontend

echo "Build complete."
