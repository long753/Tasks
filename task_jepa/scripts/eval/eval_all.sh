#!/usr/bin/env bash
set -euo pipefail

# 标准 H-JEPA L3
echo "========== Evaluating H-JEPA L3 =========="
python scripts/eval/eval.py \
  --config fourroom_l3 \
  --checkpoint ./models/e2e_l3_seed42/fourroom_hjepa_l3_object.ckpt \
  --run-name fourroom_e2e_l3

# 标准 H-JEPA
# echo "========== Evaluating H-JEPA L4 =========="
python scripts/eval/eval.py \
  --config fourroom_l4 \
  --checkpoint ./models/e2e_l4_seed42/fourroom_hjepa_l4_object.ckpt \
  --run-name fourroom_e2e_l4

# SG H-JEPA L3
echo "========== Evaluating SG H-JEPA L3 =========="
python scripts/eval/eval.py \
  --config fourroom_l3 \
  --checkpoint ./models/sg_l3_seed42/fourroom_hjepa_l3_object.ckpt \
  --run-name fourroom_sg_l3

echo "========== All evaluations completed =========="