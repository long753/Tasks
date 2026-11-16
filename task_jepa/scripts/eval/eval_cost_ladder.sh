#!/usr/bin/env bash
# Level-1 planning of the H-JEPA models with the cost measured in each upper level's
# latent (the cost-ladder tables). An n-level model is evaluated with the l2..ln
# projections. The native level-1 column uses the flat planner (<env>_flat).
# Saved checkpoints come from --checkpoint-root (models/ by default); --outputs is
# forwarded to eval.py and controls only evaluation artifacts.

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
      echo "eval_cost_ladder.sh uses --checkpoint-root for saved checkpoints; --outputs selects evaluation artifacts" >&2
      exit 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done
[[ $CHECKPOINT_ROOT = /* ]] || CHECKPOINT_ROOT="$ROOT/$CHECKPOINT_ROOT"

for env in $ENVS; do
  for seed in $SEEDS; do
    for n in 2 3 4; do
      run="$CHECKPOINT_ROOT/${env}/${env}_hjepa_l$n/seed${seed}"
      for plan in flat $(seq -f "l%g_project" 2 "$n"); do
        run_name="${env}/${env}_hjepa_l${n}/seed${seed}/eval_${plan}"
        [[ -z $RUN_NAME_PREFIX ]] || run_name="${RUN_NAME_PREFIX%/}/$run_name"
        "$PYTHON" "$ROOT/scripts/eval/eval.py" "${ARGS[@]}" --config "${env}_${plan}" \
          --checkpoint "$run/${env}_hjepa_l${n}_object.ckpt" --run-name "$run_name" "seed=$seed"
      done
    done
  done
done
