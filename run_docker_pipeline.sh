#!/usr/bin/env bash
set -euo pipefail

# TODO Edit these three paths to point to your local data directories
INPUT_DIR="$PWD/data/input/test"
OUTPUT_DIR="$PWD/data/output/test"
CONFIG_FILE="$PWD/config/correction_factors.csv"
SHAPEFILE="$PWD/data/input/Gemeinschaftsgarten_Union/Gemeinschaftsgarten_Union.shp"

# TODO Optional: additional pipeline arguments
EXTRA_ARGS="--task all \
            --tiling \
            --tile-length 3 \
            --tile-buffer 1 \
            --workers 8 \
            --crop-length 0.1 \
            --cloth-resolution 0.05 \
            --class-threshold 0.02 \
            --clear-output"

docker run --rm -it \
  -v "$PWD/3dtrees_Smart_Tile:/3dtrees_Smart_Tile" \
  -v "$PWD/py-rct:/py-rct" \
  -v "$PWD/src:/src" \
  -v "$INPUT_DIR:/data/input" \
  -v "$OUTPUT_DIR:/data/output" \
  -v "$CONFIG_FILE:/config/correction_factors.csv" \
  -v "$(dirname "$SHAPEFILE"):/data/aoi:ro" \
  tls-green-volume \
  python -u /src/run.py \
    --input /data/input \
    --output /data/output \
    --shapefile "/data/aoi/$(basename "$SHAPEFILE")" \
    --correction-file "/config/correction_factors.csv" \
    $EXTRA_ARGS