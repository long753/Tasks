"""Project-relative storage locations shared by commands and library callers."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class StoragePaths:
    datasets: Path
    models: Path
    outputs: Path

    @classmethod
    def resolve(
        cls,
        datasets: str | Path = "datasets",
        models: str | Path = "models",
        outputs: str | Path = "outputs",
    ) -> "StoragePaths":
        return cls(*(resolve_path(value) for value in (datasets, models, outputs)))

    def as_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in ("datasets", "models", "outputs")}


def add_storage_arguments(parser: argparse.ArgumentParser) -> None:
    for name in ("datasets", "models", "outputs"):
        parser.add_argument(f"--{name}", default=name, help=f"{name} directory (relative to the project root)")


def run_hydra(entrypoint: Callable[..., Any], argv: Sequence[str] | None = None) -> None:
    """Accept ordinary path flags alongside Hydra's experiment overrides."""
    parser = argparse.ArgumentParser(add_help=False)
    add_storage_arguments(parser)
    args, overrides = parser.parse_known_args(argv)
    paths = StoragePaths.resolve(args.datasets, args.models, args.outputs)
    # Keep Hydra's key=value form available without adding a paths YAML/schema.
    for index, item in enumerate(overrides):
        key, separator, value = item.lstrip("+").partition("=")
        if separator and key in paths.as_dict():
            overrides[index] = f"++{key}={json.dumps(str(resolve_path(value)))}"
    injected = [f"++{key}={json.dumps(value)}" for key, value in paths.as_dict().items() if not any(
        item.lstrip("+").startswith(f"{key}=") for item in overrides
    )]
    output_override = next((item.split("=", 1)[1] for item in overrides if item.startswith("++outputs=")), None)
    output_root = Path(json.loads(output_override)) if output_override is not None else paths.outputs
    if not any(item.lstrip("+").startswith("hydra.run.dir=") for item in overrides):
        log_dir = output_root / "hydra" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        injected.append(f"hydra.run.dir={json.dumps(str(log_dir))}")
    if any(item in ("-h", "--help") for item in overrides):
        print(parser.format_help())
    original = sys.argv
    sys.argv = [original[0], *overrides, *injected]
    try:
        entrypoint()
    finally:
        sys.argv = original
