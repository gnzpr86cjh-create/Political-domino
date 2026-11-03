#!/usr/bin/env bash
# Launch Political Domino (macOS / Linux). Create the venv and install requirements first (see README).
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
python -m streamlit run app.py
