#!/usr/bin/env bash
# Set up a Python virtual environment and install requirements
set -e

if ! command -v python3 >/dev/null; then
  echo "python3 is required but not found" >&2
  exit 1
fi

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment created. Activate it with: source venv/bin/activate"
