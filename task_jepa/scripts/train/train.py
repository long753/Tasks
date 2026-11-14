"""Command entry point; implementation lives in h_jepa.main_hjepa."""

from h_jepa.main_hjepa import run
from stable_worldmodel.paths import run_hydra

if __name__ == "__main__":
    run_hydra(run)
