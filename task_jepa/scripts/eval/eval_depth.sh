#!/usr/bin/env bash
# Standalone planning evals behind the bottom row of the depth figure: LeWM with flat
# planning, H-JEPA and HWM with 2/3/4-level planning. Model and planner seeds are paired.
# Training already runs the same evals at the end; this script re-runs them from saved
# checkpoints in models/<env>/<env>_<model>/seed<seed>/.
set -e

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
ENVS=${ENVS:-"ant fourroom cube pusht"}
SEEDS=${SEEDS:-"42 43 44"}

storage_path() {
  local flag=$1 default=$2 value
  shift 2
  value=$default
  while (($#)); do
    case $1 in
      "$flag")
        (($# > 1)) || { echo "missing value for $flag" >&2; exit 2; }
        value=$2
        shift 2
        ;;
      "$flag"=*)
        value=${1#*=}
        shift
        ;;
      *)
        shift
        ;;
    esac
  done
  [[ $value = /* ]] || value="$ROOT/$value"
  printf '%s\n' "$value"
}

MODELS_ROOT="$(storage_path --models models "$@")"
OUTPUTS_ROOT="$(storage_path --outputs outputs "$@")"

eval_model() {
  local env=$1 model=$2 plan=$3 seed=$4
  local run="$MODELS_ROOT/${env}/${env}_${model}/seed${seed}"
  local out="$OUTPUTS_ROOT/eval/${env}/${env}_${model}/seed${seed}/eval_${plan}"
  "$PYTHON" "$ROOT/scripts/eval/eval.py" "$@" --config-name "${env}_${plan}" "seed=$seed" \
    "policy=$run/${env}_${model}_object.ckpt" "output.dir=$out" "hydra.run.dir=$out/hydra"
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
