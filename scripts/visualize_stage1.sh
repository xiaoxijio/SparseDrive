#!/bin/sh
set -eu

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

CONFIG=projects/configs/sparsedrive_small_stage1_4gpu_32g.py
RESULT_FILE=${STAGE1_RESULT_FILE:-work_dirs/sparsedrive_small_stage1_4gpu_32g/results.pkl}
OUT_DIR=${STAGE1_VIS_DIR:-vis/stage1}

PYTHONPATH=. MPLBACKEND=Agg python tools/visualization/visualize_stage1.py \
    "$CONFIG" \
    --result-path "$RESULT_FILE" \
    --out-dir "$OUT_DIR" \
    "$@"
