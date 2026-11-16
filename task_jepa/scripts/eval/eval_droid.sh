#!/usr/bin/env bash
# Offline DROID planning evals behind Figure 9(b): the release planner cell of
# each model (the config defaults) on the 16 evaluation clips, with planner seeds
# 1/2/3 for every trained model, then the mean Frechet fidelity per model.
# --checkpoint-root selects saved checkpoints (models/ by default); --outputs is for
# evaluation artifacts and is forwarded to eval.py.
set -e

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
MODELS=${MODELS:-"lewm hwm_l2 hjepa_l2"}
SEEDS=${SEEDS:-"1 1000 10000"}
PLAN_SEEDS=${PLAN_SEEDS:-"1 2 3"}
EPOCHS=${EPOCHS:-""}

CHECKPOINT_ROOT="$ROOT/models"
OUTPUTS_ROOT="$ROOT/outputs"
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
    --outputs)
      (($# > 1)) || { echo "missing value for --outputs" >&2; exit 2; }
      OUTPUTS_ROOT=$2
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --outputs=*)
      OUTPUTS_ROOT=${1#*=}
      ARGS+=("$1")
      shift
      ;;
    --models|--models=*)
      echo "eval_droid.sh uses --checkpoint-root for saved checkpoints; --outputs selects evaluation artifacts" >&2
      exit 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done
[[ $CHECKPOINT_ROOT = /* ]] || CHECKPOINT_ROOT="$ROOT/$CHECKPOINT_ROOT"
[[ $OUTPUTS_ROOT = /* ]] || OUTPUTS_ROOT="$ROOT/$OUTPUTS_ROOT"
RUNS=${RUNS:-"$CHECKPOINT_ROOT/droid"}
[[ $RUNS = /* ]] || RUNS="$ROOT/$RUNS"
if [[ -n $RUN_NAME_PREFIX ]]; then
  EVAL_OUTPUTS="$OUTPUTS_ROOT/${RUN_NAME_PREFIX%/}/droid"
else
  EVAL_OUTPUTS="$OUTPUTS_ROOT/droid"
fi

for model in $MODELS; do
  plan=flat
  [[ $model =~ _l([0-9]) ]] && plan=l${BASH_REMATCH[1]}
  for seed in $SEEDS; do
    run="$RUNS/droid_${model}/seed${seed}"
    for epoch in ${EPOCHS:-final}; do
      if [[ $epoch == final ]]; then
        ckpt="$run/droid_${model}_object.ckpt"
      else
        ckpt="$run/droid_${model}_epoch_${epoch}_object.ckpt"
      fi
      if [[ ! -f $ckpt ]]; then
        echo "skip: missing $ckpt"
        continue
      fi
      for ps in $PLAN_SEEDS; do
        run_name="droid/droid_${model}/seed${seed}/eval_${plan}/epoch_${epoch}/plan_seed${ps}"
        [[ -z $RUN_NAME_PREFIX ]] || run_name="${RUN_NAME_PREFIX%/}/$run_name"
        "$PYTHON" "$ROOT/scripts/eval/eval.py" "${ARGS[@]}" --config "droid_${plan}" \
          --checkpoint "$ckpt" --run-name "$run_name" "seed=$ps"
      done
    done
  done
done

"$PYTHON" - "$MODELS" "$SEEDS" "$PLAN_SEEDS" "${EPOCHS:-final}" "$EVAL_OUTPUTS" <<'PY'
import csv
import os
import re
import sys

import numpy as np

models, seeds, plan_seeds, epochs = (arg.split() for arg in sys.argv[1:5])
for model in models:
    match = re.search(r"_l(\d)", model)
    plan = f"l{match.group(1)}" if match else "flat"
    for epoch in epochs:
        sub = f"/epoch_{epoch}"
        per_seed = []
        for seed in seeds:
            run = f"{sys.argv[5]}/droid_{model}/seed{seed}/eval_{plan}{sub}"
            paths = [f"{run}/plan_seed{plan_seed}/eval.csv" for plan_seed in plan_seeds]
            if all(os.path.exists(path) for path in paths):
                rows = []
                for path in paths:
                    with open(path, newline="") as file:
                        rows.append(next(csv.DictReader(file)))
                per_seed.append(100 * np.mean([float(row["frechet/skill_mean"]) for row in rows]))
        if not per_seed:
            continue
        se = np.std(per_seed, ddof=1) / np.sqrt(len(per_seed))
        print(
            f"droid_{model} {epoch}: Frechet fidelity {np.mean(per_seed):.2f} +- {se:.2f} "
            f"(%, mean +- SE over {len(per_seed)} train seeds)"
        )
PY
