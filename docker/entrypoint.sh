#!/bin/bash

# Activate conda environment for PDAL with Python
source /opt/conda/etc/profile.d/conda.sh

conda activate pdal-env

# Handle entrypoint for interactive and non-interactive
if [ $# -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi