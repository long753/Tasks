#!/usr/bin/env bash
# Standalone planning evals behind the bottom row of the depth figure: LeWM with flat
# planning, H-JEPA and HWM with 2/3/4-level planning. Model and planner seeds are paired.
# Training already runs the same evals at the end; this script re-runs them from saved
# checkpoints under --checkpoint-root (models/ by default). --outputs is forwarded to
# eval.py and controls only evaluation artifacts, never checkpoint lookup.
set -e

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
ENVS=${ENVS:-"ant fourroom cube pusht"}
SEEDS=${SEEDS:-"42 43 44"}

CHECKPOINT_ROOT="$ROOT/models"
RUN_NAME_PREFIX="eval/$(date +%Y%m%d-%H%M%S)-$$"
ARGS=()
while (($#)); do
  case $1 in
    --checkpoint-root)
      (($# > 1)) || { echo "missing value for --checkpoint-root" >&2; exit 2; }
      CHECKPOINT_ROOT=$2
      shift 2
      ;;
    --checkpoint-root=*)
      CHECKPOINT_ROOT=${1#*=}
      shift
      ;;
    --run-name)
      (($# > 1)) || { echo "missing value for --run-name" >&2; exit 2; }
      RUN_NAME_PREFIX=$2
      shift 2
      ;;
    --run-name=*)
      RUN_NAME_PREFIX=${1#*=}
      shift
      ;;
    --models|--models=*)
      echo "eval_depth.sh uses --checkpoint-root for saved checkpoints; --outputs selects evaluation artifacts" >&2
      exit 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done
[[ $CHECKPOINT_ROOT = /* ]] || CHECKPOINT_ROOT="$ROOT/$CHECKPOINT_ROOT"

eval_model() {
  local env=$1 model=$2 plan=$3 seed=$4
  local run="$CHECKPOINT_ROOT/${env}/${env}_${model}/seed${seed}"
  local run_name="${env}/${env}_${model}/seed${seed}/eval_${plan}"
  [[ -z $RUN_NAME_PREFIX ]] || run_name="${RUN_NAME_PREFIX%/}/$run_name"
  "$PYTHON" "$ROOT/scripts/eval/eval.py" "${ARGS[@]}" --config "${env}_${plan}" \
    --checkpoint "$run/${env}_${model}_object.ckpt" --run-name "$run_name" "seed=$seed"
}

for env in $ENVS; do
  for seed in $SEEDS; do
    eval_model "$env" lewm flat "$seed"
    for n in 2 3 4; do
      eval_model "$env" "hjepa_l$n" "l$n" "$seed"
      eval_model "$env" "hwm_l$n" "l$n" "$seed"
    done
  done
done
