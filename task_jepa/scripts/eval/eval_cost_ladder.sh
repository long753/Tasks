#!/usr/bin/env bash
# Level-1 planning of the H-JEPA models with the cost measured in each upper level's
# latent (the cost-ladder tables). An n-level model is evaluated with the l2..ln
# projections. The native level-1 column uses the flat planner (<env>_flat).
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

for env in $ENVS; do
  for seed in $SEEDS; do
    for n in 2 3 4; do
      run="$MODELS_ROOT/${env}/${env}_hjepa_l$n/seed${seed}"
      for plan in flat $(seq -f "l%g_project" 2 "$n"); do
        out="$OUTPUTS_ROOT/eval/${env}/${env}_hjepa_l${n}/seed${seed}/eval_${plan}"
        "$PYTHON" "$ROOT/scripts/eval/eval.py" "$@" --config-name "${env}_${plan}" "seed=$seed" \
          "policy=$run/${env}_hjepa_l${n}_object.ckpt" "output.dir=$out" "hydra.run.dir=$out/hydra"
      done
    done
  done
done
