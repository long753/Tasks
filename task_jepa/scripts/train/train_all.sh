#!/usr/bin/env bash
# Train every model behind the depth figure and the cost-ladder tables: 4 environments
# x 7 models x 3 seeds. Each run ends with its planning eval and final probing/decoding
# eval. The runs are independent; on a cluster, submit each line as its own job.
# ENVS=droid trains the DROID models (3 models x 3 seeds) instead; each DROID run takes one
# process per GPU (srun task), so run it inside a SLURM allocation with 2 GPUs and 2 tasks per node.
set -e

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
ENVS=${ENVS:-"ant fourroom cube pusht"}

RUN_NAME_PREFIX=
ARGS=()
while (($#)); do
  case $1 in
    --run-name)
      (($# > 1)) || { echo "missing value for --run-name" >&2; exit 2; }
      RUN_NAME_PREFIX=$2
      shift 2
      ;;
    --run-name=*)
      RUN_NAME_PREFIX=${1#*=}
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done


for env in $ENVS; do
  if [[ $env == droid ]]; then
    models=${MODELS:-"lewm hwm_l2 hjepa_l2"}
    seeds=${SEEDS:-"1 1000 10000"}
    launch=(srun --ntasks-per-node=2)
  else
    models=${MODELS:-"lewm hjepa_l2 hjepa_l3 hjepa_l4 hwm_l2 hwm_l3 hwm_l4"}
    seeds=${SEEDS:-"42 43 44"}
    launch=()
  fi
  for model in $models; do
    for seed in $seeds; do
      run_name="${env}/${env}_${model}/seed${seed}"
      [[ -z $RUN_NAME_PREFIX ]] || run_name="${RUN_NAME_PREFIX%/}/$run_name"
      "${launch[@]}" "$PYTHON" "$ROOT/scripts/train/train.py" "${ARGS[@]}" \
        --config "${env}_${model}" --run-name "$run_name" "seed=$seed"
    done
  done
done
