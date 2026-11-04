#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="python3"
if [ -x "/home/asus/miniforge3/envs/ai/bin/python" ]; then
  PYTHON_BIN="/home/asus/miniforge3/envs/ai/bin/python"
elif [ -x "/home/asus/miniforge3/envs/geospatial/bin/python" ]; then
  PYTHON_BIN="/home/asus/miniforge3/envs/geospatial/bin/python"
fi

if [ -z "${RECAMEM_ANCHOR:-}" ]; then
  export RECAMEM_ANCHOR="mock"
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}/"

# Start companion server in background
"$PYTHON_BIN" -m recamem.cli companion --name companion-pet --port "$PORT" "$@" &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for server to be ready
for _ in $(seq 1 50); do
  if curl -sf "$URL" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

# Open in default browser
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
elif command -v python3 >/dev/null 2>&1; then
  python3 -c "import webbrowser; webbrowser.open('$URL')" >/dev/null 2>&1 || true
fi

# Keep running until interrupted
wait "$SERVER_PID"