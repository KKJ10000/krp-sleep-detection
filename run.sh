#!/usr/bin/env bash
# One command: ./run.sh <video.mp4> [gallery_dir]
# Creates a local virtualenv with pinned dependencies on first run, downloads two small models, runs the detector.
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:?usage: ./run.sh <video.mp4> [gallery_dir]}"
GALLERY="${2:-gallery}"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi
.venv/bin/python -m sleepdet --video "$VIDEO" --gallery "$GALLERY" --out output --annotate
