#!/usr/bin/env bash
set -euo pipefail

# 标准 H-JEPA L4
echo "========== Evaluating H-JEPA L4 =========="
python scripts/eval/eval.py \
  --config fourroom_l4 \
  --checkpoint ./models/h_jepa/fourroom/fourroom_hjepa_l4/seed42/fourroom_hjepa_l4_object.ckpt \
  --run-name fourroom_prefix_hjepa_l4 \
  --prefix-levels 1,2,3


# e2e H-JEPA L4
echo "========== Evaluating e2e H-JEPA L4 =========="
python scripts/eval/eval.py \
  --config fourroom_l4 \
  --checkpoint ./models/e2e_l4_seed42/fourroom_hjepa_l4_object.ckpt \
  --run-name fourroom_prefix_e2e_l4 \
  --prefix-levels 1,2,3


# SG H-JEPA L4
echo "========== Evaluating SG H-JEPA L4 =========="
python scripts/eval/eval.py \
  --config fourroom_l4 \
  --checkpoint ./models/sg_l4_seed42/fourroom_hjepa_l4_epoch_1_object.ckpt \
  --run-name fourroom_prefix_sg_l4 \
  --prefix-levels 1,2,3

echo "========== All evaluations completed =========="