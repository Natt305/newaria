#!/bin/bash
set -e

PYLIBS="/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages"

uv pip install --quiet --target "$PYLIBS" -r requirements.txt
