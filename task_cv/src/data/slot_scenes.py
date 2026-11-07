"""Deterministic random multi-object scenes for Slot Attention experiments."""

from __future__ import annotations

import colorsys
import random
from numbers import Integral

import torch
from torch import Tensor


class RandomShapeScenes:
    """Sample varied, nonoverlapping circle and square scenes on the CPU.

    The generator owns a :class:`random.Random` instance, so recreating it with
    the same seed reproduces its stream without depending on Torch's global RNG.
    Instance labels are returned for evaluation only; RGB colours are shuffled
    independently for every scene and therefore do not encode those labels.
    """

    _BACKGROUND = 0.08
    _SCENE_ATTEMPTS = 64
    _PLACEMENT_ATTEMPTS = 128

    def __init__(self, image_size: int, max_objects: int = 3, seed: int = 42) -> None:
        if not _is_integer(image_size) or image_size < 16:
            raise ValueError("image_size must be an integer of at least 16")
        if not _is_integer(max_objects) or not 1 <= max_objects <= 3:
            raise ValueError("max_objects must be an integer between 1 and 3")
        if not _is_integer(seed):
            raise ValueError("seed must be an integer")

        self.image_size = int(image_size)
        self.max_objects = int(max_objects)
        self._rng = random.Random(int(seed))
        coordinates = torch.arange(self.image_size, dtype=torch.float32)
        self._rows, self._columns = torch.meshgrid(coordinates, coordinates, indexing="ij")

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Return RGB images and instance labels for a freshly sampled batch.

        Images have shape ``[B, 3, H, W]`` and ``float32`` values in ``[0, 1]``.
        Labels have shape ``[B, H, W]`` and ``int64`` values: zero is background
        and every sampled scene contains contiguous object labels from one.
        """
        if not _is_integer(batch_size) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        count = int(batch_size)
        images = torch.full(
            (count, 3, self.image_size, self.image_size),
            self._BACKGROUND,
            dtype=torch.float32,
        )
        labels = torch.zeros((count, self.image_size, self.image_size), dtype=torch.int64)

        for batch_index in range(count):
            object_count = self._rng.randint(1, self.max_objects)
            masks = self._scene_masks(object_count)
            colours = self._scene_colours(object_count)
            for object_id, (mask, colour) in enumerate(zip(masks, colours), start=1):
                labels[batch_index][mask] = object_id
                for channel, value in enumerate(colour):
                    images[batch_index, channel][mask] = value

        return images, labels

    def _scene_masks(self, object_count: int) -> list[Tensor]:
        """Place every requested object, retrying whole scenes after a conflict."""
        minimum_extent = self.image_size * 0.09
        maximum_extent = self.image_size * 0.16

        for _ in range(self._SCENE_ATTEMPTS):
            occupied = torch.zeros((self.image_size, self.image_size), dtype=torch.bool)
            masks: list[Tensor] = []
            for _ in range(object_count):
                shape = self._rng.randrange(2)
                extent = self._rng.uniform(minimum_extent, maximum_extent)
                mask = self._place_shape(shape, extent, occupied)
                if mask is None:
                    break
                occupied |= mask
                masks.append(mask)
            if len(masks) == object_count:
                return masks

        raise RuntimeError(
            f"could not place {object_count} nonoverlapping shapes after "
            f"{self._SCENE_ATTEMPTS} scene attempts"
        )

    def _place_shape(self, shape: int, extent: float, occupied: Tensor) -> Tensor | None:
        low = extent
        high = (self.image_size - 1) - extent
        for _ in range(self._PLACEMENT_ATTEMPTS):
            center_x = self._rng.uniform(low, high)
            center_y = self._rng.uniform(low, high)
            horizontal = self._columns - center_x
            vertical = self._rows - center_y
            if shape == 0:
                mask = horizontal.square() + vertical.square() <= extent * extent
            else:
                mask = torch.maximum(horizontal.abs(), vertical.abs()) <= extent
            if not torch.any(mask & occupied):
                return mask
        return None

    def _scene_colours(self, object_count: int) -> list[tuple[float, float, float]]:
        """Create a shuffled, well-separated vibrant palette for one scene."""
        starting_hue = self._rng.random()
        hues = [
            (starting_hue + index / object_count + self._rng.uniform(-0.08, 0.08)) % 1.0
            for index in range(object_count)
        ]
        self._rng.shuffle(hues)
        return [
            colorsys.hsv_to_rgb(
                hue,
                self._rng.uniform(0.70, 0.92),
                self._rng.uniform(0.80, 0.98),
            )
            for hue in hues
        ]


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)
