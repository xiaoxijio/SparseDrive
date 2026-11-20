#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
# Bound each process: rental shells may otherwise inherit 48 BLAS threads.
export OMP_NUM_THREADS="${SPARSEDRIVE_OMP_THREADS:-1}"
export MKL_NUM_THREADS="${SPARSEDRIVE_MKL_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Activate sparsedrive before calling this script. Override PORT if occupied.
# Usage: bash scripts/train_ep1_s1.sh [config.py] [tools/train.py arguments]
CONFIG=projects/configs/ep1/s1_det_occ_b0_nose.py
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"
  shift
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi
python tools/prepare/prepare_ep1_anchors.py
python -m torch.distributed.run \
  --nproc_per_node=4 --master_port="${PORT:-29501}" \
  tools/train.py "$CONFIG" --launcher pytorch --no-validate "$@"
