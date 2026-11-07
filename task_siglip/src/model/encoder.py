import math
from typing import cast

import torch
from torch import nn

from src.model.transformer import TransformerBlock


def _lecun_normal_(weight):
    """Use SigLIP's standard-normal truncation and variance correction."""
    fan_in = weight[0].numel()
    std = math.sqrt(1.0 / fan_in) / 0.87962566103423978
    with torch.no_grad():
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-2.0, b=2.0)
        weight.mul_(std)


class VisionEncoder(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        embed_dim=192,
        num_layers=4,
        num_heads=3,
        mlp_hidden_dim=768,
    ):
        super().__init__()

        if image_size <= 0 or patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch size")

        self.image_size = image_size
        self.patch_projection = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        _lecun_normal_(self.patch_projection.weight)
        nn.init.zeros_(cast(torch.Tensor, self.patch_projection.bias))

        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.empty(1, num_patches, embed_dim))
        nn.init.normal_(
            self.position_embedding,
            std=embed_dim**-0.5,
        )

        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_hidden_dim) for _ in range(num_layers)])

        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.pool_query = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.pool_query)
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        # MultiheadAttention already uses Xavier for in_proj and zeros for both biases.
        _lecun_normal_(self.pool_attention.out_proj.weight)
        self.pool_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        fc1 = nn.Linear(embed_dim, mlp_hidden_dim)
        fc2 = nn.Linear(mlp_hidden_dim, embed_dim)
        for layer in (fc1, fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias, std=1e-6)
        self.pool_mlp = nn.Sequential(fc1, nn.GELU(approximate="tanh"), fc2)

    def _embed_patches(self, pixel_values):
        """Embed the input images into a sequence of patch embeddings."""
        x = self.patch_projection(pixel_values)
        return x.flatten(2).transpose(1, 2)

    def _pool_tokens(self, tokens):
        """Pool the sequence of tokens into a single vector using attention."""
        batch_size = tokens.shape[0]
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attention(
            query=query,
            key=tokens,
            value=tokens,
            need_weights=False,
        )
        pooled = pooled + self.pool_mlp(self.pool_norm(pooled))
        return pooled[:, 0, :]

    def forward(self, pixel_values):
        # patch embedding and position encoding
        x = self._embed_patches(pixel_values)
        x = x + self.position_embedding

        # transformer blocks
        for block in self.blocks:
            x = block(x)

        # pool the tokens into a single vector
        tokens = self.final_norm(x)
        vector = self._pool_tokens(tokens)

        return vector


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=192,
        num_layers=4,
        num_heads=3,
        mlp_hidden_dim=768,
        max_length=64,
        output_dim=192,
    ):
        super().__init__()

        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        # Initialize embeddings with a normal distribution
        std = embed_dim**-0.5
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=std)

        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_hidden_dim) for _ in range(num_layers)])

        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.projection = nn.Linear(embed_dim, output_dim)
        _lecun_normal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, input_ids):
        # Validate input shape
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, L]")
        if input_ids.shape[1] != self.max_length:
            raise ValueError(f"Expected text length {self.max_length}, got {input_ids.shape[1]}")

        # Embed tokens and add positional embeddings
        positions = torch.arange(
            self.max_length,
            device=input_ids.device,
        )

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        pooled = x[:, -1, :]

        return self.projection(pooled)
