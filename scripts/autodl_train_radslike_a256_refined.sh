#!/usr/bin/env bash
set -euo pipefail

CODE=${CODE:-/root/autodl-fs/projects/grt}
GRT_ENV=${GRT_ENV:-/root/autodl-tmp/envs/grt}
COMMON_CFG=${COMMON_CFG:-"grt/grt.yaml grt/small.yaml data/outdoor.yaml repr/rads_like.yaml data/iq1m_radslike_precomputed.yaml splits/codex_iq1m_radslike_model_selection_autodl.yaml optim/radslike_adapt.yaml repr/rads_like_a256_refined.yaml"}
OUT=${OUT:-/root/autodl-fs/outputs/grt/iq1m_radslike_a256_refined_autodl_5090}
ACCUMULATE=${ACCUMULATE:-1}
OMP_NUM_THREADS=${A256_OMP_NUM_THREADS:-1}
MKL_NUM_THREADS=${A256_MKL_NUM_THREADS:-1}

if [[ -z "${BASE_MODEL:-}" ]]; then
  echo "BASE_MODEL must point to a completed A256 baseline experiment." >&2
  exit 2
fi

LOAD_FULL_DECODER=${LOAD_FULL_DECODER:-1}
TRAIN_REFINER_ONLY=${TRAIN_REFINER_ONLY:-1}
PATH="${GRT_ENV}/bin:${PATH}"
export CODE GRT_ENV COMMON_CFG OUT ACCUMULATE OMP_NUM_THREADS MKL_NUM_THREADS
export BASE_MODEL LOAD_FULL_DECODER TRAIN_REFINER_ONLY PATH
exec bash "${CODE}/scripts/autodl_train_radslike.sh" "${1:-map}"
