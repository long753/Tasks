#!/usr/bin/env bash
for env in pusht cube; do
    for model in lewm hjepa_l2 hjepa_l3 hjepa_l4; do
        hf download jepa-world-models/h-jepa \
            --local-dir "models/h_jepa" \
            --include "${env}/${env}_${model}/seed42/*"
    done
done
