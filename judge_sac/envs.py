"""Environment definitions for the required partial-observation tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    max_episode_steps: int


class _PendulumPartialObservation(gym.Wrapper):
    def __init__(self, indices: tuple[int, ...], obs_low: np.ndarray, obs_high: np.ndarray) -> None:
        super().__init__(gym.make("Pendulum-v1"))
        self._indices = np.asarray(indices, dtype=np.int64)
        self.observation_space = spaces.Box(
            low=obs_low.astype(np.float32),
            high=obs_high.astype(np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._project(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._project(obs), float(reward), terminated, truncated, info

    def _project(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)[self._indices]


class PendulumP(_PendulumPartialObservation):
    def __init__(self) -> None:
        super().__init__(
            indices=(0, 1),
            obs_low=np.array([-1.0, -1.0], dtype=np.float32),
            obs_high=np.array([1.0, 1.0], dtype=np.float32),
        )


class PendulumV(_PendulumPartialObservation):
    def __init__(self) -> None:
        super().__init__(
            indices=(2,),
            obs_low=np.array([-8.0], dtype=np.float32),
            obs_high=np.array([8.0], dtype=np.float32),
        )


class _ContinuousCartPoleBase(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, indices: tuple[int, ...], obs_high: np.ndarray) -> None:
        super().__init__()
        self._indices = np.asarray(indices, dtype=np.int64)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 30.0
        self.tau = 0.02
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.min_action = -1.0
        self.max_action = 1.0
        self.action_space = spaces.Box(
            low=np.array([self.min_action], dtype=np.float32),
            high=np.array([self.max_action], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-obs_high.astype(np.float32),
            high=obs_high.astype(np.float32),
            dtype=np.float32,
        )
        self.state: np.ndarray | None = None
        self.steps_beyond_done: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        del options
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.steps_beyond_done = None
        return self._project(self.state), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        action_array = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action_array = np.clip(action_array, self.action_space.low, self.action_space.high)
        force = self.force_mag * float(action_array.item())

        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.asarray([x, x_dot, theta, theta_dot], dtype=np.float32)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        reward = 1.0
        if terminated and self.steps_beyond_done is not None:
            reward = 0.0
            self.steps_beyond_done += 1
        elif terminated:
            self.steps_beyond_done = 0

        return self._project(self.state), float(reward), terminated, False, {}

    def _project(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)[self._indices]

    def render(self) -> None:
        return None


class CartPoleP(_ContinuousCartPoleBase):
    def __init__(self) -> None:
        super().__init__(
            indices=(0, 2),
            obs_high=np.array(
                [self_x_threshold() * 2, self_theta_threshold() * 2],
                dtype=np.float32,
            ),
        )


class CartPoleV(_ContinuousCartPoleBase):
    def __init__(self) -> None:
        super().__init__(
            indices=(1, 3),
            obs_high=np.array(
                [np.finfo(np.float32).max, np.finfo(np.float32).max],
                dtype=np.float32,
            ),
        )


def self_x_threshold() -> float:
    return 2.4


def self_theta_threshold() -> float:
    return 12 * 2 * math.pi / 360


_ENV_SPECS: dict[str, tuple[type[gym.Env], EnvironmentSpec]] = {
    "PendulumP": (PendulumP, EnvironmentSpec(name="PendulumP", max_episode_steps=200)),
    "PendulumV": (PendulumV, EnvironmentSpec(name="PendulumV", max_episode_steps=200)),
    "CartPoleP": (CartPoleP, EnvironmentSpec(name="CartPoleP", max_episode_steps=1000)),
    "CartPoleV": (CartPoleV, EnvironmentSpec(name="CartPoleV", max_episode_steps=1000)),
}


def available_envs() -> tuple[str, ...]:
    return tuple(_ENV_SPECS.keys())


def make_env(name: str) -> gym.Env:
    try:
        env_cls, spec = _ENV_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown environment: {name}") from exc
    return gym.wrappers.TimeLimit(env_cls(), max_episode_steps=spec.max_episode_steps)


def get_env_spec(name: str) -> EnvironmentSpec:
    try:
        _, spec = _ENV_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown environment: {name}") from exc
    return spec
