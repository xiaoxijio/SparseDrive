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
CONFIG=projects/configs/ep1/s1_det_occ_b0_nose.py
python tools/prepare_ep1_anchors.py
python -m torch.distributed.run \
  --nproc_per_node=4 --master_port="${PORT:-29501}" \
  tools/train.py "$CONFIG" --launcher pytorch "$@"
