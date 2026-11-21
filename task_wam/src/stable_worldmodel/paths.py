"""Project-relative storage locations shared by commands and library callers."""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config"


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


def add_storage_arguments(parser: argparse.ArgumentParser, *, include_models: bool = True) -> None:
    names = ("datasets", "models", "outputs") if include_models else ("datasets", "outputs")
    for name in names:
        parser.add_argument(f"--{name}", default=name, help=f"{name} directory (relative to the project root)")


def resolve_run_name(value: str | None = None, *, prefix: str = "run") -> str:
    """Create a unique default name or validate a relative experiment namespace."""
    name = value if value is not None else f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    path = Path(name)
    if not name.strip() or path.is_absolute() or not path.parts or ".." in path.parts or path == Path("."):
        raise ValueError("--run-name must be a nonempty relative name without '..'")
    return str(path)
