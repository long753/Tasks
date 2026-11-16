"""Train Script for H-JEPA.

python scripts/train/train.py \
    --config ant_hjepa_l4 \
    --training-mode sg \
    --run-name ant_l4_sg

training-mode: e2e (default), sg, layerwise
"""

from h_jepa.main_hjepa import run
from stable_worldmodel.paths import run_hydra

if __name__ == "__main__":
    run_hydra(entrypoint=run, command="train")
