#!/usr/bin/env bash
set -euo pipefail

# TODO Edit these three paths to point to your local data directories
INPUT_DIR="$PWD/data/input/test"
OUTPUT_DIR="$PWD/data/output/test"

# TODO Optional: additional pipeline arguments
EXTRA_ARGS="--task segmentation \
            --workers 8 \
            --crop-length 0.1 \
            --cloth-resolution 0.05 \
            --class-threshold 0.02 \
            --clear-output"

docker run --rm -it \
  -v "$PWD/3dtrees_Smart_Tile:/workspace/3dtrees_Smart_Tile" \
  -v "$PWD/py-rct:/workspace/py-rct" \
  -v "$PWD/src:/src" \
  -v "$INPUT_DIR:/data/input" \
  -v "$OUTPUT_DIR:/data/output" \
  -v "$(dirname "$SHAPEFILE"):/data/aoi:ro" \
  tls-green-volume \
  python -u /src/run.py \
    --input /data/input \
    --output /data/output \
    $EXTRA_ARGS