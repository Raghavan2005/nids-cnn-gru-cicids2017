#!/usr/bin/env bash
# Start the NIDS Command Center (PySide6 app + phone web server).
# Creates a virtualenv and installs dependencies on first run.
set -e
cd "$(dirname "$0")/T014_Project"

VENV="../.venv"
PY="${PYTHON:-python3}"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! python -c "import PySide6, segno, tensorflow, numpy, pandas" 2>/dev/null; then
    echo "Installing dependencies (first run only)..."
    pip install -r requirements.txt
    pip install tensorflow keras PySide6 segno
fi

# Optional: export NIDS_PORT=9000 to change the port (default 8000)
exec python webapp/app.py "$@"
