#!/usr/bin/env bash
set -euo pipefail

CODE=${CODE:-/root/autodl-fs/projects/grt}
GRT_ENV=${GRT_ENV:-/root/autodl-tmp/envs/grt}
COMMON_CFG=${COMMON_CFG:-"grt/grt.yaml grt/small.yaml data/outdoor.yaml repr/rads_like.yaml data/iq1m_radslike_precomputed.yaml splits/codex_iq1m_radslike_model_selection_autodl.yaml optim/radslike_adapt.yaml repr/rads_like_a256.yaml"}
if [[ -n "${A256_EXTRA_CFG:-}" ]]; then
  COMMON_CFG="${COMMON_CFG} ${A256_EXTRA_CFG}"
fi
OUT=${OUT:-/root/autodl-fs/outputs/grt/iq1m_radslike_a256_autodl_5090}
ACCUMULATE=${ACCUMULATE:-1}
OMP_NUM_THREADS=${A256_OMP_NUM_THREADS:-1}
MKL_NUM_THREADS=${A256_MKL_NUM_THREADS:-1}

PATH="${GRT_ENV}/bin:${PATH}"
export CODE GRT_ENV COMMON_CFG OUT ACCUMULATE OMP_NUM_THREADS MKL_NUM_THREADS PATH
exec bash "${CODE}/scripts/autodl_train_radslike.sh" "${1:-map}"
