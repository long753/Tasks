"""Policy boundary used by the world and FourRoom controllers."""

from typing import Any

import numpy as np


class BasePolicy:
    env: Any

    def __init__(self, **kwargs: Any) -> None:
        self.env = None
        self.type = 'base'
        for name, value in kwargs.items():
            setattr(self, name, value)

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def set_env(self, env: Any) -> None:
        self.env = env

    def reset(self) -> None:
        """Clear state before a new rollout if the policy holds any."""


class RandomPolicy(BasePolicy):
    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.type = 'random'
        self.seed = seed

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        return self.env.action_space.sample()

    def set_seed(self, seed: int) -> None:
        if self.env is not None:
            self.env.action_space.seed(seed)
