#!/usr/bin/env bash
set -euo pipefail

CODE=${CODE:-/root/autodl-fs/projects/grt}
: "${BASE_MODEL:?Set BASE_MODEL to the selected A256/E1 experiment directory}"

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-/root/autodl-fs/outputs/grt/iq1m_radslike_a256_full_continue}
A256_EXTRA_CFG=${A256_EXTRA_CFG:-optim/radslike_full_continue.yaml}
LOAD_FULL_DECODER=${LOAD_FULL_DECODER:-1}
EPOCHS=${EPOCHS:-3}
PATIENCE=${PATIENCE:-2}
LIMIT_TRAIN=${LIMIT_TRAIN:-}
VAL_INTERVAL=${VAL_INTERVAL:-1.0}
MAX_TIME=${MAX_TIME:-04:00:00}
WORKERS=${WORKERS:-8}
ACCUMULATE=${ACCUMULATE:-1}
PRECISION=${PRECISION:-bf16-mixed}

export CODE BASE_MODEL RUN_TS OUT A256_EXTRA_CFG LOAD_FULL_DECODER
export EPOCHS PATIENCE LIMIT_TRAIN VAL_INTERVAL MAX_TIME WORKERS ACCUMULATE
export PRECISION

exec bash "${CODE}/scripts/autodl_train_radslike_a256.sh" map
