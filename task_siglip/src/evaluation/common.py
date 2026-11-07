"""Checkpoint loading and feature encoding shared by evaluation tasks."""

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from src.model.official_siglip import OfficialSigLIPModel
from src.model.siglip import SigLIPModel

PROCESSOR_NAME = "google/siglip-base-patch16-224"


def load_model(checkpoint_path: Path, device, *, random_init=False, seed=42):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_backend = checkpoint.get("model_backend", "custom")
    if random_init:
        torch.manual_seed(seed)
    if model_backend == "frozen_dino_bert":
        # Lazy import: the standalone trainer reuses evaluation statistics.
        from scripts.train_pretrained_siglip import RUN_NAME, create_model

        model_seed = seed if random_init else checkpoint["args"]["seed"]
        model, processor = create_model(model_seed, checkpoint, load_weights=not random_init)
        model_name = RUN_NAME
    else:
        config = checkpoint["config"]
        model_factories = {"custom": SigLIPModel, "official": OfficialSigLIPModel}
        try:
            model_factory = model_factories[model_backend]
        except KeyError as error:
            raise ValueError(f"Unsupported model backend: {model_backend!r}") from error
        tokenizer = AutoTokenizer.from_pretrained(PROCESSOR_NAME)
        model = model_factory(config=config, tokenizer=tokenizer)
        if not random_init:
            model.load_state_dict(checkpoint["model_state_dict"])
        processor = AutoImageProcessor.from_pretrained(
            PROCESSOR_NAME,
            use_fast=False,
            size={"height": config["model"]["image_size"], "width": config["model"]["image_size"]},
        )
        model_name = config["name"]
    metadata = {"checkpoint": str(checkpoint_path), "model": model_name, "epoch": checkpoint["epoch"]}
    if model_backend != "custom":
        metadata["model_backend"] = model_backend
    if random_init:
        metadata.pop("checkpoint")
        metadata.update(config_source_checkpoint=str(checkpoint_path), epoch=0, initialization="random", seed=seed)
    model = model.to(device).eval()
    del checkpoint
    return model, processor, metadata


def _is_frozen_model(model):
    from scripts.train_pretrained_siglip import FrozenSigLIP

    return isinstance(model, FrozenSigLIP)


@torch.inference_mode()
def encode_image_batch(model, pixel_values, *, normalize=True):
    if isinstance(model, SigLIPModel):
        features = model.vision_encoder(pixel_values)
    elif isinstance(model, OfficialSigLIPModel):
        features = model.get_image_features(pixel_values)
    elif _is_frozen_model(model):
        features = model.image_features(pixel_values, normalize=False)
    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")
    return F.normalize(features, dim=-1) if normalize else features


@torch.inference_mode()
def encode_text_batch(model, texts, device, *, normalize=True):
    if isinstance(model, SigLIPModel):
        input_ids = model.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=model.text_encoder.max_length,
            return_tensors="pt",
        )["input_ids"].to(device)
        features = model.text_encoder(input_ids)
    elif isinstance(model, OfficialSigLIPModel):
        features = model.get_text_features(texts, device)
    elif _is_frozen_model(model):
        features = model.text_features(texts, device, normalize=False)
    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")
    return F.normalize(features, dim=-1) if normalize else features


@torch.inference_mode()
def encode_texts(model, texts, batch_size, device):
    """Collect CPU features for full-pool retrieval; classifiers cache batches on device."""
    features = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Texts", unit="batch"):
        features.append(encode_text_batch(model, texts[start : start + batch_size], device).cpu())
    return torch.cat(features)
