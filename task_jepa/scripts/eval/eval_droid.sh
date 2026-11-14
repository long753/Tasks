#!/usr/bin/env bash
# Offline DROID planning evals behind Figure 9(b): the release planner cell of
# each model (the config defaults) on the 16 evaluation clips, with planner seeds
# 1/2/3 for every trained model, then the mean Frechet fidelity per model.
# EPOCHS="50 100 150" evaluates droid_<model>_epoch_<N>_object.ckpt instead.
set -e

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
MODELS=${MODELS:-"lewm hwm_l2 hjepa_l2"}
SEEDS=${SEEDS:-"1 1000 10000"}
PLAN_SEEDS=${PLAN_SEEDS:-"1 2 3"}
EPOCHS=${EPOCHS:-""}

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
RUNS=${RUNS:-"$MODELS_ROOT/droid"}
EVAL_OUTPUTS="$OUTPUTS_ROOT/eval/droid"

for model in $MODELS; do
  plan=flat
  [[ $model =~ _l([0-9]) ]] && plan=l${BASH_REMATCH[1]}
  for seed in $SEEDS; do
    run="$RUNS/droid_${model}/seed${seed}"
    for epoch in ${EPOCHS:-final}; do
      if [[ $epoch == final ]]; then
        ckpt="$run/droid_${model}_object.ckpt"
        out="$EVAL_OUTPUTS/droid_${model}/seed${seed}/eval_${plan}"
      else
        ckpt="$run/droid_${model}_epoch_${epoch}_object.ckpt"
        out="$EVAL_OUTPUTS/droid_${model}/seed${seed}/eval_${plan}/epoch_${epoch}"
      fi
      if [[ ! -f $ckpt ]]; then
        echo "skip: missing $ckpt"
        continue
      fi
      for ps in $PLAN_SEEDS; do
        plan_out="$out/plan_seed${ps}"
        "$PYTHON" "$ROOT/scripts/eval/eval.py" "$@" --config-name "droid_${plan}" "policy=$ckpt" "seed=$ps" \
          "output.dir=$plan_out" "hydra.run.dir=$plan_out/hydra"
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
        sub = "" if epoch == "final" else f"/epoch_{epoch}"
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
