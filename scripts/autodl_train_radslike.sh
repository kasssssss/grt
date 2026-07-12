#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-map}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}

CODE=${CODE:-/root/autodl-fs/projects/grt}
DATA=${DATA:-/root/autodl-tmp/data/iq1m}
CKPT_ROOT=${CKPT_ROOT:-/root/autodl-tmp/checkpoints/iq1m-checkpoints}
OUT=${OUT:-/root/autodl-fs/outputs/grt/iq1m_radslike_autodl_5090_fixed_20260710}

COMMON_CFG=${COMMON_CFG:-"grt/grt.yaml grt/small.yaml data/outdoor.yaml repr/rads_like.yaml data/iq1m_radslike_precomputed.yaml splits/codex_iq1m_radslike_model_selection_autodl.yaml optim/radslike_adapt.yaml"}
EPOCHS=${EPOCHS:-200}
PATIENCE=${PATIENCE:-12}
WORKERS=${WORKERS:-8}
ACCUMULATE=${ACCUMULATE:-4}
LOG_INTERVAL=${LOG_INTERVAL:-25}
LOG_EXAMPLE_INTERVAL=${LOG_EXAMPLE_INTERVAL:-250}
NUM_CHECKPOINTS=${NUM_CHECKPOINTS:-1}
VAL_INTERVAL=${VAL_INTERVAL:-1.0}
# Full-data training is the default. Model-selection jobs can still set an
# explicit batch cap, for example LIMIT_TRAIN=500.
LIMIT_TRAIN=${LIMIT_TRAIN:-}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export CUDA_VISIBLE_DEVICES
export HF_HOME=${HF_HOME:-/root/autodl-tmp/cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/root/autodl-tmp/cache/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/root/autodl-tmp/cache/xdg}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTHONPATH="${CODE}:${PYTHONPATH:-}"

mkdir -p "${OUT}"
cd "${CODE}"

case "${MODE}" in
  map)
    OBJ=${MAP_OBJ:-"obj/map_radslike_balanced.yaml"}
    OFFICIAL="${CKPT_ROOT}/base/small"
    DECODER_HEAD="occ3d"
    METRIC=${MAP_METRIC:-"map_f1/val"}
    METRIC_MODE=${MAP_METRIC_MODE:-"max"}
    NAME="iq1m_radslike_map_autodl"
    VERSION="map_${RUN_TS}"
    EXTRA_ARGS=()
    EXTRA_CKPT_ARGS=(
      --extra_checkpoint_metric "loss/val:min"
      --extra_checkpoint_metric "map_depth/val:min"
      --extra_checkpoint_metric "map_m1_f1/val:max"
      --extra_checkpoint_metric "map_p1_f1/val:max"
    )
    ;;
  depth)
    # Paper-style depth is the first hit of the same 3D occupancy volume used
    # for BEV. Select a checkpoint by depth without using the stale direct head.
    OBJ=${MAP_OBJ:-"obj/map_radslike_balanced.yaml"}
    OFFICIAL="${CKPT_ROOT}/base/small"
    DECODER_HEAD="occ3d"
    METRIC="map_depth/val"
    METRIC_MODE="min"
    NAME="iq1m_radslike_map_depth_select_autodl"
    VERSION="map_depth_${RUN_TS}"
    EXTRA_ARGS=()
    EXTRA_CKPT_ARGS=(
      --extra_checkpoint_metric "loss/val:min"
      --extra_checkpoint_metric "map_f1/val:max"
    )
    ;;
  semseg|semantic)
    OBJ="obj/segment_lidar_index_balanced.yaml"
    OFFICIAL="${CKPT_ROOT}/semseg/small"
    DECODER_HEAD="semseg"
    METRIC="seg_miou/val"
    METRIC_MODE="max"
    NAME="iq1m_radslike_semseg_autodl"
    VERSION="semseg_${RUN_TS}"
    EXTRA_ARGS=()
    EXTRA_CKPT_ARGS=(--extra_checkpoint_metric "loss/val:min")
    if [[ "${FREEZE_DECODER:-0}" == "1" ]]; then
      EXTRA_ARGS+=(--freeze_decoder)
    fi
    ;;
  *)
    echo "Unknown MODE=${MODE}; use map, depth, or semseg." >&2
    exit 2
    ;;
esac

if [[ "${TRAIN_REFINER_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--train_refiner_only)
fi
if [[ "${TRAIN_UNPATCH_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--train_unpatch_only)
fi
if [[ "${FREEZE_ENCODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--freeze)
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT_TRAIN:-}" ]]; then
  LIMIT_ARGS+=(--limit_train_batches "${LIMIT_TRAIN}")
fi

SEED_ARGS=()
if [[ -n "${SEED:-}" ]]; then
  SEED_ARGS+=(--seed "${SEED}")
fi
if [[ -n "${LIMIT_VAL:-}" ]]; then
  LIMIT_ARGS+=(--limit_val_batches "${LIMIT_VAL}")
fi
if [[ -n "${MAX_TIME:-}" ]]; then
  if [[ "${MAX_TIME}" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    MAX_TIME="00:${MAX_TIME}"
  fi
  LIMIT_ARGS+=(--max_time "${MAX_TIME}")
fi

LOG="${OUT}/train_${MODE}_${RUN_TS}.log"

INIT_ARGS=()
if [[ "${FROM_SCRATCH:-0}" == "1" ]]; then
  if [[ -n "${BASE_MODEL:-}" ]]; then
    echo "FROM_SCRATCH=1 cannot be combined with BASE_MODEL." >&2
    exit 2
  fi
elif [[ -n "${BASE_MODEL:-}" ]]; then
  INIT_ARGS+=(--base_model "${BASE_MODEL}")
  if [[ "${LOAD_FULL_DECODER:-0}" == "1" ]]; then
    INIT_ARGS+=(--load_full_decoder)
  elif [[ "${LOAD_DECODER:-0}" == "1" ]]; then
    INIT_ARGS+=(--load_decoder)
  fi
else
  INIT_ARGS+=(
    --official_base_model "${OFFICIAL}"
    --official_decoder_head "${DECODER_HEAD}"
    --official_min_loaded_fraction "${OFFICIAL_MIN_LOADED_FRACTION:-0.99}"
    --official_elevation_index "${OFFICIAL_ELEVATION_INDEX:-0}"
  )
fi

{
  echo "RUN_TS=${RUN_TS}"
  echo "MODE=${MODE}"
  echo "CODE=${CODE}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}"
  echo "COMMON_CFG=${COMMON_CFG}"
  echo "OBJECTIVE=${OBJ}"
  echo "OFFICIAL=${OFFICIAL}"
  echo "BASE_MODEL=${BASE_MODEL:-}"
  echo "FROM_SCRATCH=${FROM_SCRATCH:-0}"
  echo "INIT_ARGS=${INIT_ARGS[*]}"
  echo "DECODER_HEAD=${DECODER_HEAD}"
  echo "EPOCHS=${EPOCHS} PATIENCE=${PATIENCE} WORKERS=${WORKERS}"
  echo "ACCUMULATE=${ACCUMULATE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "SEED=${SEED:-}"
  echo "LIMIT_ARGS=${LIMIT_ARGS[*]:-}"
  date -Is
  df -h /root/autodl-tmp /root/autodl-fs /
  nvidia-smi || true
} | tee "${LOG}"

python -u train.py \
  -p "${DATA}" \
  -o "${OUT}" \
  -c ${COMMON_CFG} "${OBJ}" \
  "${INIT_ARGS[@]}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --metric "${METRIC}" \
  --metric_mode "${METRIC_MODE}" \
  --num_checkpoints "${NUM_CHECKPOINTS}" \
  --val_interval "${VAL_INTERVAL}" \
  --log_interval "${LOG_INTERVAL}" \
  --log_example_interval "${LOG_EXAMPLE_INTERVAL}" \
  --no_progress_bar \
  --workers "${WORKERS}" \
  --accumulate_grad_batches "${ACCUMULATE}" \
  --precision "${PRECISION:-bf16-mixed}" \
  "${SEED_ARGS[@]}" \
  --name "${NAME}" \
  --version "${VERSION}" \
  "${LIMIT_ARGS[@]}" \
  "${EXTRA_CKPT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG}"

echo "DONE_TRAIN ${OUT}/${NAME}/${VERSION}" | tee -a "${LOG}"
