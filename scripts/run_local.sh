#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/backend"
FRONTEND_DIR="${ROOT_DIR}/frontend"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

PY_VER="$(${PYTHON_BIN} - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:3])))
PY
)"

PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
  echo "TensorFlow CPU requires Python 3.11+. Detected ${PY_VER}. Please use PYTHON_BIN=python3.11 or upgrade your interpreter."
  exit 1
fi

echo "Setting up Python virtualenv with ${PYTHON_BIN}..."
if [ ! -d "${BACKEND_DIR}/.venv" ]; then
  "${PYTHON_BIN}" -m venv "${BACKEND_DIR}/.venv"
fi
source "${BACKEND_DIR}/.venv/bin/activate"
pip install --upgrade pip

echo "Installing backend dependencies (NumPy + TensorFlow)..."
# Base pinned deps
pip install fastapi==0.115.5 uvicorn[standard]==0.30.6 python-multipart==0.0.9 numpy==1.26.4 matplotlib==3.8.4 httpx==0.27.2 redis==5.1.1 rq==1.15.1
pip install -e "${BACKEND_DIR}/interpy_synth" --no-deps
pip install -e "${BACKEND_DIR}/interpy_bg" --no-deps
if [ "$(uname -s)" = "Darwin" ]; then
  echo "Detected macOS; installing tensorflow-macos..."
  pip install tensorflow-macos==2.15.0 || { echo "Failed to install tensorflow-macos==2.15.0"; exit 1; }
else
  echo "Installing tensorflow-cpu..."
  pip install tensorflow-cpu==2.15.0 || { echo "Failed to install tensorflow-cpu==2.15.0"; exit 1; }
fi
pip install -e "${BACKEND_DIR}/fivedreg_tf" --no-deps

echo "Installing frontend dependencies..."
cd "${FRONTEND_DIR}"
npm install

export NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL:-http://localhost:8000}
export API_URL=${API_URL:-${NEXT_PUBLIC_API_URL}}

echo "Starting backend (uvicorn) in background..."
cd "${BACKEND_DIR}"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

cd "${FRONTEND_DIR}"
echo "Starting frontend (Next.js) in foreground..."
trap "echo 'Stopping backend (PID ${BACKEND_PID})'; kill ${BACKEND_PID}" EXIT
npm run dev
