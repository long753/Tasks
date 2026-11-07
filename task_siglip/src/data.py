"""Unified on-disk manifest dataset for Flickr8k and COCO."""

import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        root: str | Path = ".",
        *,
        random_caption: bool = True,
    ):
        self.root = Path(root)
        self.random_caption = random_caption
        with Path(manifest).open(encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]
        if not self.items:
            raise ValueError(f"Manifest has no samples: {manifest}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        with Image.open(self.root / item["image_path"]) as image:
            image = image.convert("RGB")
        captions = item["captions"]
        text = random.choice(captions) if self.random_caption else captions[0]
        return {"image": image, "text": text}


def make_collate_fn(processor, max_text_length: int = 64):
    def collate_fn(batch):
        return processor(
            images=[item["image"] for item in batch],
            text=[item["text"] for item in batch],
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )

    return collate_fn
