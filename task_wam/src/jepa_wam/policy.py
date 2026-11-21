"""Receding-horizon WAM policy for stable_worldmodel.World."""

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from jepa_wam.model import WAM
from stable_worldmodel.policy import BasePolicy


class WAMPolicy(BasePolicy):
    def __init__(self, model: WAM, *, replan_steps: int = 2, ode_steps: int = 8, seed: int = 42):
        super().__init__()
        if not 1 <= replan_steps <= model.horizon or ode_steps < 1:
            raise ValueError('Use 1 <= replan_steps <= horizon and at least one ODE step')
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.replan_steps = replan_steps
        self.ode_steps = ode_steps
        self.goal: Tensor | None = None
        self.pending: deque[np.ndarray] = deque()

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: str = 'cuda', **kwargs) -> 'WAMPolicy':
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = WAM(**saved['model_config'])
        model.load_state_dict(saved['model'])
        return cls(model.to(device), **kwargs)

    def reset(self) -> None:
        self.goal = None
        self.pending.clear()

    @property
    def needs_plan(self) -> bool:
        """Whether the next action requires sampling a new chunk."""
        return not self.pending

    def _encode(self, obs: dict, pixels: str, proprio: str) -> Tensor:
        images = torch.as_tensor(obs[pixels][:, -1], device=self.device).permute(0, 3, 1, 2)
        state = (
            torch.as_tensor(obs[proprio][:, -1], device=self.device, dtype=torch.float32)
            if self.model.encoder.state_dim else None
        )
        return self.model.encoder(images, state)

    @torch.no_grad()
    def get_action(self, obs: dict, **kwargs) -> np.ndarray:
        if self.needs_plan:
            if self.goal is None:
                self.goal = self._encode(obs, 'goal', 'goal_proprio')
            current = self._encode(obs, 'pixels', 'proprio')
            chunk = self.model.plan(current, self.goal, self.ode_steps, self.generator)
            self.pending.extend(chunk[:, :self.replan_steps].transpose(0, 1).cpu().numpy())
        action_space = self.env.action_space
        return np.clip(self.pending.popleft(), action_space.low, action_space.high)
