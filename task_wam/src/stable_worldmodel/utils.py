"""Utilities shared by the FourRoom environment and wrappers."""

from collections.abc import Iterable
from typing import Any


def get_in(mapping: Any, path: Iterable[str]) -> Any:
    for key in path:
        mapping = mapping[key]
    return mapping
