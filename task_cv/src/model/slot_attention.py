
import torch
import torch.nn as nn


class SlotAttention(nn.Module):
    """Iteratively group token features into a fixed number of latent slots."""

    def __init__(self, num_slots: int, slot_size: int, num_iterations: int) -> None:
        super().__init__()
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if slot_size <= 0:
            raise ValueError("slot_size must be positive")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")

        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_iterations = num_iterations
        self.epsilon = 1e-8
        self.scale = slot_size**-0.5

        # Each forward pass draws a fresh set of slots from this learned Gaussian.
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_size))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))

        self.norm_inputs = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_pre_mlp = nn.LayerNorm(slot_size)

        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(slot_size, slot_size, bias=False)
        self.project_v = nn.Linear(slot_size, slot_size, bias=False)

        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, 2 * slot_size),
            nn.ReLU(),
            nn.Linear(2 * slot_size, slot_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return slots with shape ``[batch, num_slots, slot_size]``.

        ``inputs`` must have shape ``[batch, num_tokens, slot_size]``. Attention
        first competes across slots for each token, then is renormalized across
        tokens so every slot receives a weighted summary.
        """
        if inputs.ndim != 3 or inputs.size(-1) != self.slot_size:
            raise ValueError(
                "inputs must have shape [batch, num_tokens, "
                f"{self.slot_size}]"
            )

        batch_size = inputs.size(0)
        inputs = self.norm_inputs(inputs)
        keys = self.project_k(inputs)
        values = self.project_v(inputs)

        noise = torch.randn(
            batch_size,
            self.num_slots,
            self.slot_size,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        slots = self.slots_mu + self.slots_log_sigma.exp() * noise

        for _ in range(self.num_iterations):
            previous_slots = slots
            queries = self.project_q(self.norm_slots(slots))
            logits = torch.einsum("bkd,bnd->bkn", queries, keys) * self.scale

            # Normalize over slots first, then make each slot's weights sum to one.
            attention = torch.softmax(logits, dim=1)
            attention = attention + self.epsilon
            attention = attention / attention.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bkn,bnd->bkd", attention, values)
            slots = self.gru(
                updates.reshape(-1, self.slot_size),
                previous_slots.reshape(-1, self.slot_size),
            ).reshape(batch_size, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_pre_mlp(slots))

        return slots


class VisualEncoder(nn.Module):
    """Encode images into position-aware feature tokens."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")

        # Extract local features while reducing each spatial dimension by four.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.position_proj = nn.Linear(2, dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return tokens with shape ``[batch, height * width, dim]``."""
        features = self.cnn(images)
        _, _, height, width = features.shape

        y = torch.linspace(
            -1,
            1,
            height,
            device=features.device,
            dtype=features.dtype,
        )
        x = torch.linspace(
            -1,
            1,
            width,
            device=features.device,
            dtype=features.dtype,
        )
        y_coordinates, x_coordinates = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([x_coordinates, y_coordinates], dim=-1)
        positions = self.position_proj(grid).permute(2, 0, 1).unsqueeze(0)

        tokens = (features + positions).flatten(2).transpose(1, 2)
        return self.mlp(self.norm(tokens))


class SpatialBroadcastDecoder(nn.Module):
    """Decode slots into RGB images and per-slot mask logits.

    The shared decoder receives every slot independently, so its RGB output has
    shape ``[batch, slots, 3, image_size, image_size]`` and mask logits have
    shape ``[batch, slots, 1, image_size, image_size]``.
    """

    def __init__(
        self,
        image_size: int,
        slot_dim: int,
        hidden_dim: int = 64,
        constant_color: bool = False,
    ) -> None:
        super().__init__()
        if image_size < 16 or image_size % 16 != 0:
            raise ValueError("image_size must be a multiple of 16 and at least 16")
        if slot_dim <= 0:
            raise ValueError("slot_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.image_size = image_size
        self.slot_dim = slot_dim
        self.grid_size = image_size // 16
        self.position_proj = nn.Linear(2, slot_dim)
        # A solid-colour scene needs one colour per object, not a whole RGB image.
        # This explicit prior is optional; textured images use the general decoder.
        self.constant_color = constant_color
        self.output_channels = 1 if constant_color else 4
        self.color_proj = nn.Linear(slot_dim, 3) if constant_color else None

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(
                slot_dim,
                hidden_dim,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, self.output_channels, kernel_size=3, padding=1),
        )

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unmasked RGB reconstructions and unnormalized mask logits."""
        if slots.ndim != 3 or slots.size(-1) != self.slot_dim:
            raise ValueError(
                f"slots must have shape [batch, num_slots, {self.slot_dim}]"
            )

        batch_size, num_slots, _ = slots.shape
        coordinates = torch.linspace(
            -1,
            1,
            self.grid_size,
            device=slots.device,
            dtype=slots.dtype,
        )
        y_coordinates, x_coordinates = torch.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )
        grid = torch.stack([x_coordinates, y_coordinates], dim=-1)
        positions = self.position_proj(grid).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

        broadcast_slots = slots[:, :, :, None, None] + positions
        decoded = self.cnn(
            broadcast_slots.reshape(
                batch_size * num_slots,
                self.slot_dim,
                self.grid_size,
                self.grid_size,
            )
        )
        decoded = decoded.reshape(
            batch_size,
            num_slots,
            self.output_channels,
            self.image_size,
            self.image_size,
        )
        if self.constant_color:
            colors = self.color_proj(slots).sigmoid()
            rgb = colors[:, :, :, None, None].expand(
                -1, -1, -1, self.image_size, self.image_size
            )
            return rgb, decoded
        # Keep RGB linear for the MSE objective; sigmoid can saturate on white backgrounds.
        # Clamp only when displaying results, never in the training path.
        return decoded[:, :, :3], decoded[:, :, 3:]


class SlotAttentionAutoEncoder(nn.Module):
    """Slot Attention autoencoder with compositional per-slot reconstructions."""

    def __init__(
        self,
        image_size: int = 128,
        slot_count: int = 4,
        slot_dim: int = 64,
        num_iterations: int = 3,
        constant_color: bool = False,
    ) -> None:
        super().__init__()
        if image_size < 16 or image_size % 16 != 0:
            raise ValueError("image_size must be a multiple of 16 and at least 16")
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        if slot_dim <= 0:
            raise ValueError("slot_dim must be positive")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")

        self.image_size = image_size
        self.encoder = VisualEncoder(dim=slot_dim)
        self.slot_attention = SlotAttention(
            num_slots=slot_count,
            slot_size=slot_dim,
            num_iterations=num_iterations,
        )
        self.decoder = SpatialBroadcastDecoder(
            image_size=image_size,
            slot_dim=slot_dim,
            constant_color=constant_color,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return composited reconstruction, masks, per-slot RGB, slots, and tokens."""
        if (
            images.ndim != 4
            or images.size(1) != 3
            or images.size(2) != self.image_size
            or images.size(3) != self.image_size
        ):
            raise ValueError(
                "images must have shape [batch, 3, "
                f"{self.image_size}, {self.image_size}]"
            )

        tokens = self.encoder(images)
        slots = self.slot_attention(tokens)
        slot_reconstructions, mask_logits = self.decoder(slots)
        masks = torch.softmax(mask_logits, dim=1)
        reconstruction = (masks * slot_reconstructions).sum(dim=1)

        return {
            "reconstruction": reconstruction,
            "masks": masks,
            "slot_reconstructions": slot_reconstructions,
            "slots": slots,
            "tokens": tokens,
        }
