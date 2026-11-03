#!/usr/bin/env bash
set -e

# Path setup
COMPANION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$COMPANION_DIR")"

echo "=== RecaMem Companion Runner ==="

# Check/install companion npm dependencies
if [ ! -d "$COMPANION_DIR/node_modules" ]; then
  echo "Installing frontend dependencies..."
  pnpm --prefix "$COMPANION_DIR" install
fi

# Cleanup child processes on exit/ctrl+c
cleanup() {
  echo ""
  echo "Stopping RecaMem backend (PID: $BACKEND_PID)..."
  kill "$BACKEND_PID" 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Choose python interpreter: prefer conda 'ai' env, fallback to python3/python
PYTHON_BIN="python"
if [ -x "/home/asus/miniforge3/envs/ai/bin/python" ]; then
  PYTHON_BIN="/home/asus/miniforge3/envs/ai/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

# Fallback to mock anchor if ONNX cache/weights not present
if [ -z "$RECAMEM_ANCHOR" ]; then
  export RECAMEM_ANCHOR="mock"
fi

# Start backend
echo "Starting backend API (port 8000) using $PYTHON_BIN..."
(
  cd "$ROOT_DIR"
  PYTHONPATH=. "$PYTHON_BIN" recamem/cli.py serve --rest --port 8000
) &
BACKEND_PID=$!

# Wait briefly for backend to bind port
sleep 1.5

# Start frontend dev server and open browser
echo "Starting frontend dev server (port 5173)..."
pnpm --prefix "$COMPANION_DIR" dev -- --open
