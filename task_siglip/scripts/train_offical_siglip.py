"""Train the official SigLIP architecture from scratch with Tiny parameters.

uv run python -m scripts.train_offical_siglip --dataset coco --num_epochs 20 --batch_size 16
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(object=PROJECT_ROOT))

from scripts.train import parse_args, run_training
from src.model.official_siglip import OfficialSigLIPModel


def main(argv=None):
    args = parse_args(argv, default_config="configs/tiny.yaml", description=__doc__)
    run_training(args, model_factory=OfficialSigLIPModel, model_backend="official")


if __name__ == "__main__":
    main()
