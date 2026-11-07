import torch
from torch import nn
from transformers import SiglipConfig, SiglipModel


class OfficialSigLIPModel(nn.Module):
    """Official SigLIP architecture, trained from scratch with the project's config."""

    def __init__(self, config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        model_config = config["model"]
        vision_config = model_config["vision"]
        text_config = model_config["text"]
        self.max_text_length = model_config["max_text_length"]

        self.backbone = SiglipModel(
            SiglipConfig(
                text_config={
                    "vocab_size": len(tokenizer),
                    "hidden_size": text_config["hidden_size"],
                    "intermediate_size": text_config["intermediate_size"],
                    "num_hidden_layers": text_config["num_hidden_layers"],
                    "num_attention_heads": text_config["num_attention_heads"],
                    "max_position_embeddings": self.max_text_length,
                    "projection_size": text_config["projection_size"],
                    "hidden_act": "gelu_pytorch_tanh",
                    "layer_norm_eps": 1e-6,
                },
                vision_config={
                    "hidden_size": vision_config["hidden_size"],
                    "intermediate_size": vision_config["intermediate_size"],
                    "num_hidden_layers": vision_config["num_hidden_layers"],
                    "num_attention_heads": vision_config["num_attention_heads"],
                    "image_size": model_config["image_size"],
                    "patch_size": model_config["patch_size"],
                    "hidden_act": "gelu_pytorch_tanh",
                    "layer_norm_eps": 1e-6,
                },
            )
        )
        with torch.no_grad():
            self.backbone.logit_scale.fill_(model_config["init_logit_scale"])
            self.backbone.logit_bias.fill_(model_config["init_logit_bias"])

    def _tokenize(self, texts, device):
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(device)

    def forward(self, texts, pixel_values):
        input_ids = self._tokenize(texts, pixel_values.device)
        return self.backbone(
            input_ids=input_ids,
            pixel_values=pixel_values,
        ).logits_per_image

    def get_image_features(self, pixel_values):
        return self.backbone.get_image_features(pixel_values=pixel_values)

    def get_text_features(self, texts, device):
        input_ids = self._tokenize(texts, device)
        return self.backbone.get_text_features(input_ids=input_ids)
