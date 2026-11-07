import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from src.model.encoder import TextEncoder, VisionEncoder


class SigLIPModel(nn.Module):
    def __init__(self, config=None, tokenizer=None):
        super().__init__()
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(pretrained_model_name_or_path="google/siglip-base-patch16-224")
        )

        if config is None:
            init_logit_scale = 2.3
            init_logit_bias = -10.0
            self.text_encoder = TextEncoder(vocab_size=len(self.tokenizer))
            self.vision_encoder = VisionEncoder()
        else:
            model_config = config["model"]
            vision_config = model_config["vision"]
            text_config = model_config["text"]
            vision_output_dim = vision_config["hidden_size"]
            text_output_dim = text_config["projection_size"]
            init_logit_scale = model_config["init_logit_scale"]
            init_logit_bias = model_config["init_logit_bias"]
            if vision_output_dim != text_output_dim:
                raise ValueError("Vision output dimension must match text projection dimension")

            self.text_encoder = TextEncoder(
                vocab_size=len(self.tokenizer),
                embed_dim=text_config["hidden_size"],
                num_layers=text_config["num_hidden_layers"],
                num_heads=text_config["num_attention_heads"],
                mlp_hidden_dim=text_config["intermediate_size"],
                max_length=model_config["max_text_length"],
                output_dim=text_output_dim,
            )
            self.vision_encoder = VisionEncoder(
                image_size=model_config["image_size"],
                patch_size=model_config["patch_size"],
                embed_dim=vision_output_dim,
                num_layers=vision_config["num_hidden_layers"],
                num_heads=vision_config["num_attention_heads"],
                mlp_hidden_dim=vision_config["intermediate_size"],
            )

        # Logit scale and bias parameters for similarity computation
        self.logit_scale = nn.Parameter(torch.tensor(data=[init_logit_scale]))
        self.logit_bias = nn.Parameter(torch.tensor(data=[init_logit_bias]))

    def _calculate_similarity(self, input_ids, pixel_values):
        # 计算文本的归一化特征向量
        text_features = self.text_encoder(input_ids)
        text_features = F.normalize(input=text_features, p=2, dim=-1)

        # 计算图像的归一化特征向量
        image_features = self.vision_encoder(pixel_values)
        image_features = F.normalize(input=image_features, p=2, dim=-1)

        # 计算文本和图像特征向量之间的相似度
        similarity = image_features @ text_features.T
        logits = similarity * self.logit_scale.exp() + self.logit_bias

        return logits

    def forward(self, texts, pixel_values, positive_mask=None):
        # 合理性检验
        if not texts or pixel_values is None:
            raise ValueError("Inputs are invalid")

        # 文本的 tokenized
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.text_encoder.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device=next(self.text_encoder.parameters()).device)

        # 计算 logits
        logits = self._calculate_similarity(input_ids, pixel_values)

        return logits
