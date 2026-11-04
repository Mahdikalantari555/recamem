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

exec "$PYTHON_BIN" -m recamem.cli companion --name companion-pet "$@"
