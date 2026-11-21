"""Fixed FourRoom task protocol and policy timing shared by benchmark modes."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from stable_worldmodel.paths import CONFIG_ROOT
from stable_worldmodel.policy import BasePolicy
from stable_worldmodel.world import World


def load_fourroom_protocol(datasets: Path) -> DictConfig:
    protocol = cast(DictConfig, OmegaConf.load(CONFIG_ROOT / 'eval/fourroom_l4.yaml'))
    protocol.datasets = str(datasets)
    protocol.world.num_envs = 1
    protocol.world.max_episode_steps = 2 * int(protocol.eval.eval_budget)
    # Success must use the position in the goal image, not the collector's target.
    goal_call = next(call for call in protocol.eval.callables if call.method == '_set_goal_state')
    goal_call.args.goal_state.value = 'goal_xy'
    return protocol


def fixed_task_payload(tasks: dict[str, Any], task_index: int) -> dict[str, Any]:
    episode = dict(tasks['data'][task_index])
    env_seed = int(tasks['task_info']['tasks'][task_index]['seed'])
    # World reads seed from episode columns. A scalar tensor row survives its column conversion.
    episode['seed'] = torch.full((len(episode['pixels']),), env_seed, dtype=torch.int64)
    return {
        'data': [episode], 'episodes_idx': [int(tasks['episodes_idx'][task_index])],
        'columns': [*tasks['columns'], 'seed'],
    }


def replay_observations(tasks: dict[str, Any], task_index: int) -> list[dict[str, np.ndarray]]:
    episode = tasks['data'][task_index]
    columns = {key: episode[key].numpy() for key in tasks['columns'] if not key.startswith('goal')}
    columns['pixels'] = np.moveaxis(columns['pixels'], 1, -1)
    # Match World's raw [env, history, ...] input (FourRoom history_size=1).
    goal = {'goal' if key == 'pixels' else f'goal_{key}': values[None, -1:] for key, values in columns.items()}
    actions = columns.pop('action')
    return [
        goal | {key: values[None, frame:frame + 1] for key, values in columns.items()}
        | {'step_idx': np.array([[frame]]), 'action': actions[None, max(frame - 1, 0):max(frame, 1)]}
        for frame in range(len(columns['pixels']))
    ]


class PlanningPolicy(Protocol):
    @property
    def needs_plan(self) -> bool: ...

    def set_env(self, env: Any) -> None: ...
    def reset(self) -> None: ...
    def get_action(self, obs: dict[str, Any], **kwargs: Any) -> np.ndarray: ...


class CountingPolicy(BasePolicy):
    """Count planning calls and measure synchronized observation-to-action latency."""

    def __init__(self, policy: PlanningPolicy, sync: Callable[[], None]) -> None:
        super().__init__()
        self.policy = policy
        self.sync = sync
        self.actions = 0
        self.plans = 0
        self.inference_s = self.last_inference_s = 0.0

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        self.policy.set_env(env)

    def reset(self) -> None:
        self.actions = self.plans = 0
        self.inference_s = self.last_inference_s = 0.0
        self.policy.reset()

    def get_action(self, obs: dict[str, Any], **kwargs: Any) -> np.ndarray:
        needs_plan = self.policy.needs_plan
        self.sync()
        start = perf_counter()
        action = self.policy.get_action(obs, **kwargs)
        self.sync()
        self.last_inference_s = perf_counter() - start
        self.inference_s += self.last_inference_s
        self.plans += needs_plan
        self.actions += 1
        return action


@dataclass(frozen=True)
class ClosedLoopResult:
    model: str
    task_index: int
    episode_index: int
    environment_seed: int
    policy_seed: int
    success: bool
    steps_to_success: int | None
    env_steps: int
    planning_calls: int
    inference_s: float
    inference_ms_per_step: float
    elapsed_s: float


@dataclass(frozen=True)
class ReplayResult:
    model: str
    task_index: int
    frame: int
    kind: str
    elapsed_ms: float


def closed_loop(
    world: World, policy: CountingPolicy, tasks: dict[str, Any], task_index: int,
    *, model: str, budget: int, policy_seed: int, callables: list[dict[str, Any]],
) -> ClosedLoopResult:
    payload = fixed_task_payload(tasks, task_index)
    policy.sync()
    start = perf_counter()
    metrics = world.evaluate_from_dataset(
        None, None, None, goal_offset_steps=int(tasks['goal_offset_steps']),
        eval_budget=budget, callables=callables, load_eval_trajs_path=payload,
        stop_on_success=True,
    )
    policy.sync()
    elapsed_s = perf_counter() - start
    success = bool(metrics['episode_successes'][0])
    return ClosedLoopResult(
        model, task_index, payload['episodes_idx'][0], int(metrics['seeds'][0]), policy_seed, success,
        int(metrics['steps_to_success'][0]) if success else None, policy.actions, policy.plans,
        policy.inference_s, 1000 * policy.inference_s / policy.actions, elapsed_s,
    )


def time_replay(
    policy: CountingPolicy, observations: Sequence[dict[str, np.ndarray]],
    *, model: str, task_index: int,
) -> Iterator[ReplayResult]:
    policy.reset()
    for frame, obs in enumerate(observations):
        # WorldModelPolicy transforms this dictionary in place; preserve it for the next model.
        inputs = obs.copy()
        previous_plans = policy.plans
        policy.get_action(inputs)
        if policy.plans == previous_plans:
            kind = 'cached_action'
        else:
            kind = 'first_plan' if previous_plans == 0 else 'replan'
        yield ReplayResult(model, task_index, frame, kind, policy.last_inference_s * 1000)
