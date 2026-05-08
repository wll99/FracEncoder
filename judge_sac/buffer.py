"""Trajectory replay for sequence-based Judge training."""

from __future__ import annotations

import pickle
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EpisodeTrajectory:
    obs: np.ndarray
    actions: np.ndarray
    rewards_env: np.ndarray
    terminateds: np.ndarray
    truncateds: np.ndarray
    returns_mc: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def total_return(self) -> float:
        return float(self.rewards_env.sum())


@dataclass
class TrajectoryBatch:
    obs: np.ndarray
    next_obs: np.ndarray
    actions: np.ndarray
    rewards_env: np.ndarray
    prev_rewards: np.ndarray
    dones: np.ndarray
    truncateds: np.ndarray
    returns_mc: np.ndarray
    mask: np.ndarray
    loss_mask: np.ndarray
    prev_actions: np.ndarray


@dataclass
class TransitionBatch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    truncateds: np.ndarray


def compute_mc_returns(
    rewards_env: np.ndarray,
    terminateds: np.ndarray,
    discount_gamma: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    rewards = np.asarray(rewards_env, dtype=np.float32)
    terminals = np.asarray(terminateds, dtype=np.float32)
    valid_mask = (
        np.ones_like(rewards, dtype=np.float32)
        if mask is None
        else np.asarray(mask, dtype=np.float32)
    )
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for index in range(rewards.shape[0] - 1, -1, -1):
        if valid_mask[index] == 0.0:
            running = 0.0
            continue
        if terminals[index] > 0.0:
            running = rewards[index]
        else:
            running = rewards[index] + discount_gamma * running
        returns[index] = running
    return returns


class TrajectoryBuffer:
    def __init__(self, max_episodes: int, discount_gamma: float) -> None:
        self.max_episodes = max_episodes
        self.discount_gamma = discount_gamma
        self.episodes: deque[EpisodeTrajectory] = deque(maxlen=max_episodes)
        self.total_transitions = 0

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def add_episode(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards_env: np.ndarray,
        dones: np.ndarray | None = None,
        *,
        terminateds: np.ndarray | None = None,
        truncateds: np.ndarray | None = None,
    ) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards_env = np.asarray(rewards_env, dtype=np.float32).reshape(-1)
        if terminateds is None:
            if dones is None:
                raise ValueError("Either dones or terminateds must be provided.")
            terminateds = dones
        terminateds = np.asarray(terminateds, dtype=np.float32).reshape(-1)
        if truncateds is None:
            truncateds = np.zeros_like(terminateds, dtype=np.float32)
        truncateds = np.asarray(truncateds, dtype=np.float32).reshape(-1)

        if obs.shape[0] != actions.shape[0] + 1:
            raise ValueError("obs must have one more timestep than actions.")
        if (
            actions.shape[0] != rewards_env.shape[0]
            or actions.shape[0] != terminateds.shape[0]
            or actions.shape[0] != truncateds.shape[0]
        ):
            raise ValueError(
                "actions, rewards_env, terminateds, and truncateds must share the same length."
            )

        returns_mc = compute_mc_returns(rewards_env, terminateds, self.discount_gamma)
        if len(self.episodes) == self.max_episodes:
            self.total_transitions -= self.episodes[0].length
        episode = EpisodeTrajectory(
            obs=obs,
            actions=actions,
            rewards_env=rewards_env,
            terminateds=terminateds,
            truncateds=truncateds,
            returns_mc=returns_mc,
        )
        self.episodes.append(episode)
        self.total_transitions += episode.length

    def state_dict(self) -> dict[str, object]:
        return {
            "max_episodes": self.max_episodes,
            "discount_gamma": self.discount_gamma,
            "total_transitions": self.total_transitions,
            "episodes": [
                {
                    "obs": episode.obs,
                    "actions": episode.actions,
                    "rewards_env": episode.rewards_env,
                    "terminateds": episode.terminateds,
                    "truncateds": episode.truncateds,
                    "returns_mc": episode.returns_mc,
                }
                for episode in self.episodes
            ],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.max_episodes = int(state["max_episodes"])
        self.discount_gamma = float(state["discount_gamma"])
        self.total_transitions = int(state["total_transitions"])
        self.episodes = deque(
            (
                EpisodeTrajectory(
                    obs=np.asarray(item["obs"], dtype=np.float32),
                    actions=np.asarray(item["actions"], dtype=np.float32),
                    rewards_env=np.asarray(item["rewards_env"], dtype=np.float32),
                    terminateds=np.asarray(
                        item["terminateds"] if "terminateds" in item else item["dones"],
                        dtype=np.float32,
                    ),
                    truncateds=np.asarray(
                        item["truncateds"]
                        if "truncateds" in item
                        else np.zeros_like(
                            item["terminateds"] if "terminateds" in item else item["dones"],
                            dtype=np.float32,
                        ),
                        dtype=np.float32,
                    ),
                    returns_mc=np.asarray(item["returns_mc"], dtype=np.float32),
                )
                for item in state["episodes"]
            ),
            maxlen=self.max_episodes,
        )

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(self.state_dict(), handle)
        return output_path

    def load(self, path: str | Path) -> None:
        input_path = Path(path)
        with input_path.open("rb") as handle:
            state = pickle.load(handle)
        self.load_state_dict(state)

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        burn_in: int = 0,
        sampling_mode: str = "uniform",
    ) -> TrajectoryBatch:
        if not self.episodes:
            raise ValueError("Cannot sample from an empty TrajectoryBuffer.")

        window_len = seq_len + burn_in
        sampled_episodes = random.choices(
            list(self.episodes),
            weights=self._episode_sampling_weights(sampling_mode),
            k=batch_size,
        )
        obs_dim = sampled_episodes[0].obs.shape[-1]
        action_dim = sampled_episodes[0].actions.shape[-1]

        obs_batch = np.zeros((batch_size, window_len + 1, obs_dim), dtype=np.float32)
        actions_batch = np.zeros((batch_size, window_len, action_dim), dtype=np.float32)
        rewards_batch = np.zeros((batch_size, window_len), dtype=np.float32)
        prev_rewards_batch = np.zeros((batch_size, window_len), dtype=np.float32)
        dones_batch = np.zeros((batch_size, window_len), dtype=np.float32)
        truncateds_batch = np.zeros((batch_size, window_len), dtype=np.float32)
        returns_batch = np.zeros((batch_size, window_len), dtype=np.float32)
        mask_batch = np.zeros((batch_size, window_len), dtype=np.float32)
        loss_mask_batch = np.zeros((batch_size, window_len), dtype=np.float32)
        prev_actions_batch = np.zeros((batch_size, window_len, action_dim), dtype=np.float32)

        for batch_index, episode in enumerate(sampled_episodes):
            if episode.length <= 0:
                continue
            train_start = random.randint(0, episode.length - 1)
            prefix_start = max(0, train_start - burn_in)
            train_end = min(episode.length, train_start + seq_len)
            window_end = train_end

            local_len = window_end - prefix_start
            obs_slice = episode.obs[prefix_start : window_end + 1]
            actions_slice = episode.actions[prefix_start:window_end]
            rewards_slice = episode.rewards_env[prefix_start:window_end]
            dones_slice = episode.terminateds[prefix_start:window_end]
            truncateds_slice = episode.truncateds[prefix_start:window_end]
            returns_slice = episode.returns_mc[prefix_start:window_end]

            prev_actions = np.zeros_like(actions_slice, dtype=np.float32)
            prev_rewards = np.zeros_like(rewards_slice, dtype=np.float32)
            absolute_indices = np.arange(prefix_start, window_end)
            prev_valid = absolute_indices - 1
            for local_index, prev_index in enumerate(prev_valid):
                if prev_index >= 0:
                    prev_actions[local_index] = episode.actions[prev_index]
                    prev_rewards[local_index] = episode.rewards_env[prev_index]

            obs_batch[batch_index, : local_len + 1] = obs_slice
            actions_batch[batch_index, :local_len] = actions_slice
            rewards_batch[batch_index, :local_len] = rewards_slice
            prev_rewards_batch[batch_index, :local_len] = prev_rewards
            dones_batch[batch_index, :local_len] = dones_slice
            truncateds_batch[batch_index, :local_len] = truncateds_slice
            returns_batch[batch_index, :local_len] = returns_slice
            mask_batch[batch_index, :local_len] = 1.0
            loss_start = train_start - prefix_start
            loss_end = loss_start + (train_end - train_start)
            loss_mask_batch[batch_index, loss_start:loss_end] = 1.0
            prev_actions_batch[batch_index, :local_len] = prev_actions

        return TrajectoryBatch(
            obs=obs_batch[:, :-1],
            next_obs=obs_batch[:, 1:],
            actions=actions_batch,
            rewards_env=rewards_batch,
            prev_rewards=prev_rewards_batch,
            dones=dones_batch,
            truncateds=truncateds_batch,
            returns_mc=returns_batch,
            mask=mask_batch,
            loss_mask=loss_mask_batch,
            prev_actions=prev_actions_batch,
        )

    def _episode_sampling_weights(self, sampling_mode: str) -> list[float] | None:
        if sampling_mode == "uniform":
            return None
        episodes = list(self.episodes)
        if sampling_mode == "recent":
            return [float(index + 1) for index in range(len(episodes))]
        if sampling_mode == "return":
            returns = np.asarray([episode.total_return for episode in episodes], dtype=np.float32)
            shifted = returns - float(returns.min()) + 1e-3
            return shifted.tolist()
        raise ValueError(f"Unsupported trajectory sampling_mode: {sampling_mode}")


class FlatReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.truncateds = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
        truncated: float = 0.0,
    ) -> None:
        self.obs[self.position] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.actions[self.position] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[self.position] = float(reward)
        self.next_obs[self.position] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.dones[self.position] = float(done)
        self.truncateds[self.position] = float(truncated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        if self.size < batch_size:
            raise ValueError("Not enough transitions in FlatReplayBuffer.")
        indices = np.random.randint(0, self.size, size=batch_size)
        return TransitionBatch(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            dones=self.dones[indices],
            truncateds=self.truncateds[indices],
        )
