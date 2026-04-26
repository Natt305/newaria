#!/bin/bash
set -euo pipefail

PYLIBS="/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages"
mkdir -p "$PYLIBS"

uv pip install --quiet --target "$PYLIBS" -r requirements.txt
