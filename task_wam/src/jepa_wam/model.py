"""Single-level visual WAM: goal-conditioned latent flow and action flow."""

import math
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.nn.functional import all_reduce
from transformers import ViTConfig, ViTModel

LATENT_DIM = 192
LATENT_TOKENS = 4


class ObservationEncoder(nn.Module):
    image_mean: Tensor
    image_std: Tensor
    state_mean: Tensor
    state_std: Tensor

    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.vit = ViTModel(
            ViTConfig(image_size=224, patch_size=14, hidden_size=LATENT_DIM,
                      intermediate_size=768, num_hidden_layers=12, num_attention_heads=3),
            add_pooling_layer=False, use_mask_token=False,
        )
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.state_projection = nn.Linear(state_dim, LATENT_DIM) if state_dim else None
        self.fusion = (
            nn.TransformerEncoderLayer(LATENT_DIM, 3, 4 * LATENT_DIM, dropout=0.0, activation='gelu',
                                       batch_first=True, norm_first=True)
            if state_dim else nn.Identity()
        )

    def forward(self, pixels: Tensor, state: Tensor | None = None) -> Tensor:
        x = pixels.to(dtype=torch.float32, copy=True).div_(255).sub_(self.image_mean).div_(self.image_std)
        tokens = self.vit(x).last_hidden_state
        if self.state_projection is None:
            return tokens[:, 0]
        if state is None:
            raise ValueError('This encoder requires proprioception for the observation and the goal')
        state_token = self.state_projection((state - self.state_mean) / self.state_std).unsqueeze(1)
        return self.fusion(torch.cat((tokens, state_token), dim=1)).mean(dim=1)


class FlowBlock(nn.Module):
    """DiT-style self-attention and MLP, each modulated by the condition."""

    def __init__(self, width: int):
        super().__init__()
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 4 * width))
        self.attn_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift_a, scale_a, shift_m, scale_m = self.modulation(condition).unsqueeze(1).chunk(4, dim=-1)
        query = self.attn_norm(x) * (1 + scale_a) + shift_a
        x = x + self.attention(query, query, query, need_weights=False)[0]
        return x + self.mlp(self.mlp_norm(x) * (1 + scale_m) + shift_m)


class VelocityField(nn.Module):
    frequencies: Tensor

    def __init__(self, length: int, features: int, width: int = 128, depth: int = 3):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, length, width) * 0.02)
        self.register_buffer('frequencies', torch.exp(torch.linspace(0.0, math.log(1000), width // 2)))
        self.time_embedding = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.condition = nn.Linear(2 * LATENT_DIM, width)
        self.input = nn.Linear(features, width)
        self.blocks = nn.ModuleList(FlowBlock(width) for _ in range(depth))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, features))

    def forward(self, x: Tensor, time: Tensor, current: Tensor, goal: Tensor) -> Tensor:
        phase = time[:, None] * self.frequencies
        time_features = torch.cat((phase.sin(), phase.cos()), dim=-1)
        condition = self.condition(torch.cat((current, goal), dim=-1)) + self.time_embedding(time_features)
        x = self.input(x) + self.position
        for block in self.blocks:
            x = block(x, condition)
        return self.output(x)

    def sample(self, noise: Tensor, current: Tensor, goal: Tensor, steps: int) -> Tensor:
        x = noise
        for i in range(steps):
            time = x.new_full((x.size(0),), i / steps)
            x = x + self(x, time, current, goal) / steps
        return x


def noisy_target(target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    noise = torch.randn_like(target)
    time = torch.rand(target.size(0), device=target.device)
    return torch.lerp(noise, target, time[:, None, None]), time, target - noise


class SIGReg(nn.Module):
    """Compare the empirical characteristic functions of random unit projections to N(0, 1)."""

    grid: Tensor
    gaussian: Tensor

    def __init__(self):
        super().__init__()
        grid = torch.linspace(0, 3, 17)
        gaussian = torch.exp(-0.5 * grid.square())
        self.register_buffer('grid', grid)
        self.register_buffer('gaussian', gaussian)

    def forward(self, latents: Tensor) -> Tensor:
        with torch.autocast(latents.device.type, enabled=False):
            axes = F.normalize(torch.randn(latents.size(-1), 256, device=latents.device), dim=0)
            distributed = torch.distributed.is_initialized()
            if distributed:
                torch.distributed.broadcast(axes, src=0)
            angles = (latents.float() @ axes).unsqueeze(-1) * self.grid
            ecf = torch.stack((angles.cos().mean(0), angles.sin().mean(0)))
            samples = latents.size(0)
            if distributed:
                world_size = torch.distributed.get_world_size()
                # Equal-sized rank batches; DDP averages the resulting parameter gradients.
                ecf = all_reduce(ecf) / world_size
                samples *= world_size
            cos, sin = ecf.unbind()
            error = (cos - self.gaussian).square() + sin.square()
            return 2 * samples * torch.trapezoid(error * self.gaussian, self.grid, dim=-1).mean()


class WAM(nn.Module):
    def __init__(self, horizon: int, action_dim: int, state_dim: int):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.encoder = ObservationEncoder(state_dim)
        self.target_encoder = deepcopy(self.encoder).requires_grad_(False).eval()
        self.world = VelocityField(LATENT_TOKENS, LATENT_DIM // LATENT_TOKENS)
        self.expert = VelocityField(horizon, action_dim)
        self.sigreg = SIGReg()

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def initialize_encoder(self, weights: dict[str, Tensor], stats: tuple[Tensor, Tensor] | None) -> None:
        self.encoder.vit.load_state_dict(weights)
        if stats is not None:
            self.encoder.state_mean.copy_(stats[0])
            self.encoder.state_std.copy_(stats[1])
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    @torch.no_grad()
    def update_target(self, decay: float) -> None:
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target.lerp_(online, 1 - decay)

    def forward(self, batch: dict[str, Tensor], *, use_prediction: bool) -> dict[str, Tensor]:
        current = self.encoder(batch['current'], batch.get('current_state'))
        goal = self.encoder(batch['goal'], batch.get('goal_state'))
        with torch.no_grad():
            future = self.target_encoder(batch['future'], batch.get('future_state')).float()
        target = future.reshape(-1, LATENT_TOKENS, LATENT_DIM // LATENT_TOKENS)
        x, time, velocity = noisy_target(target)
        world_velocity = self.world(x, time, current, goal).float()
        endpoint = (x + (1 - time[:, None, None]) * world_velocity).flatten(1)
        subgoal = endpoint if use_prediction else future
        actions, action_time, action_velocity = noisy_target(batch['actions'].float())
        return {
            'world': F.mse_loss(world_velocity, velocity),
            'action': F.mse_loss(self.expert(actions, action_time, current, subgoal).float(), action_velocity),
            'sigreg': self.sigreg(torch.cat((current, goal), dim=0)),
            'subgoal_rmse': F.mse_loss(endpoint.detach(), future).sqrt(),
        }

    @torch.no_grad()
    def plan(self, current: Tensor, goal: Tensor, steps: int, generator: torch.Generator | None = None) -> Tensor:
        batch_size = current.size(0)
        latent_noise = torch.randn(batch_size, LATENT_TOKENS, LATENT_DIM // LATENT_TOKENS,
                                   device=current.device, generator=generator)
        subgoal = self.world.sample(latent_noise, current, goal, steps).flatten(1)
        action_noise = torch.randn(batch_size, self.horizon, self.action_dim,
                                   device=current.device, generator=generator)
        return self.expert.sample(action_noise, current, subgoal, steps)
