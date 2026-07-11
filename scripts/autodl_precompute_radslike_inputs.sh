#!/usr/bin/env bash
set -euo pipefail

CODE=${CODE:-/root/autodl-fs/projects/grt}
DATA=${DATA:-/root/autodl-tmp/data/iq1m}
CACHE=${CACHE:-/root/autodl-tmp/data/iq1m_radslike_precomputed_single_elev_f16_v1}

BATCH_SIZE=${BATCH_SIZE:-32}
SHARD_SAMPLES=${SHARD_SAMPLES:-256}
WORKERS_PER_PROCESS=${WORKERS_PER_PROCESS:-8}
NUM_PROCESSES=${NUM_PROCESSES:-1}
CACHE_DTYPE=${CACHE_DTYPE:-float16}
PRECOMPUTE_SCRIPT=${PRECOMPUTE_SCRIPT:-scripts/precompute_radslike_inputs.py}
if [[ "${PRECOMPUTE_SCRIPT}" == *"_cpu.py" ]]; then
  DEFAULT_DEVICE_PREFIX=cpu
else
  DEFAULT_DEVICE_PREFIX=cuda
fi
DEVICE_PREFIX=${DEVICE_PREFIX:-${DEFAULT_DEVICE_PREFIX}}

export HF_HOME=${HF_HOME:-/root/autodl-tmp/cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/root/autodl-tmp/cache/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/root/autodl-tmp/cache/xdg}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTHONPATH="${CODE}:${PYTHONPATH:-}"

mkdir -p "${CACHE}" /root/autodl-fs/outputs/grt
cd "${CODE}"

echo "CODE=${CODE}"
echo "DATA=${DATA}"
echo "CACHE=${CACHE}"
echo "PRECOMPUTE_SCRIPT=${PRECOMPUTE_SCRIPT}"
echo "BATCH_SIZE=${BATCH_SIZE} SHARD_SAMPLES=${SHARD_SAMPLES}"
echo "NUM_PROCESSES=${NUM_PROCESSES} WORKERS_PER_PROCESS=${WORKERS_PER_PROCESS}"
echo "CACHE_DTYPE=${CACHE_DTYPE}"
date -Is
df -h /root/autodl-tmp /root/autodl-fs /
nvidia-smi || true

python -u "${PRECOMPUTE_SCRIPT}" \
  --repo "${CODE}" \
  --data-root "${DATA}" \
  --out-root "${CACHE}" \
  --batch-size "${BATCH_SIZE}" \
  --shard-samples "${SHARD_SAMPLES}" \
  --workers-per-process "${WORKERS_PER_PROCESS}" \
  --num-processes "${NUM_PROCESSES}" \
  --device-prefix "${DEVICE_PREFIX}" \
  --cache-dtype "${CACHE_DTYPE}" \
  "$@"

if [[ "${RUN_CHECK:-1}" == "1" ]]; then
  python -u scripts/check_precomputed_channel.py \
    --repo "${CODE}" \
    --data-root "${DATA}" \
    --cache-root "${CACHE}" \
    --trace "${CHECK_TRACE:-outdoor/baum}"
fi

echo "DONE_PRECOMPUTE $(date -Is)"
