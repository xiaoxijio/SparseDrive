#!/bin/sh
set -eu
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
CONFIG=projects/configs/sparsedrive_small_stage1_4gpu_32g.py
CHECKPOINT=${STAGE1_CHECKPOINT:-work_dirs/sparsedrive_small_stage1_4gpu_32g/iter_175800.pth}
RESULT_FILE=${STAGE1_RESULT_FILE:-work_dirs/sparsedrive_small_stage1_4gpu_32g/results.pkl}
GPU=${STAGE1_GPU:-0}
EVAL_ONLY=0
if [ "${1:-}" = "--eval-only" ]; then
    EVAL_ONLY=1
    shift
fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ "$EVAL_ONLY" -eq 0 ]; then
    test -f "$CHECKPOINT" || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 1; }
    mkdir -p "$(dirname "$RESULT_FILE")"
    CUDA_VISIBLE_DEVICES="$GPU" bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" 1 \
        --deterministic --out "$RESULT_FILE" --cfg-options data.workers_per_gpu=0 "$@"
else
    test "$#" -eq 0 || { echo '--eval-only accepts no additional arguments' >&2; exit 1; }
fi
# Run tasks sequentially so each process releases all its memory before the next.
for TASK in detection tracking map; do
    CUDA_VISIBLE_DEVICES="" python tools/eval_stage1.py "$CONFIG" \
        --result-path "$RESULT_FILE" --task "$TASK"
done
