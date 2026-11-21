from pathlib import Path

from stable_worldmodel.paths import StoragePaths


def get_cache_dir() -> Path:
    """Return the project dataset directory, creating it when needed."""
    cache_dir = StoragePaths.resolve().datasets
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
