"""Training entry point for Judge-SAC experiments."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .buffer import FlatReplayBuffer, TrajectoryBatch, TrajectoryBuffer, TransitionBatch
from .envs import available_envs, make_env
from .judge import HistoryRewardJudge, build_context_inputs, build_judge_inputs
from .sac import SACAgent


@dataclass
class TrainingConfig:
    variant: str = "custom"
    env_name: str = "PendulumP"
    steps: int = 5_000
    seed: int = 0
    trajectory_buffer_episodes: int = 256
    flat_buffer_capacity: int = 100_000
    batch_size: int = 8
    seq_len: int = 64
    burn_in: int = 0
    start_steps: int = 1_000
    update_after: int = 1_000
    update_every: int = 1
    eval_interval: int = 2_000
    eval_episodes: int = 5
    reward_mode: str = "mix"
    mix_alpha: float = 0.5
    discount_gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    hidden_size: int = 256
    hidden_layers: int = 2
    frac_alpha: float = 0.6
    freeze_judge_frac_alpha: bool = False
    judge_memory_dim: int = 32
    judge_hidden_dim: int = 64
    judge_lr: float = 1e-3
    judge_enabled: bool = True
    judge_input_includes_env_reward: bool = True
    judge_reward_scale: float | None = None
    judge_reward_clip: float | None = None
    judge_mix_warmup_steps: int | None = None
    replay_mode: str = "auto"
    trajectory_sampling_mode: str = "uniform"
    actor_context_mode: str = "fade"
    actor_context_dim: int = 128
    actor_context_memory_dim: int = 128
    actor_context_hidden_dim: int = 128
    actor_context_frac_alpha: float = 0.6
    freeze_actor_context_frac_alpha: bool = False
    critic_context_mode: str = "fade"
    critic_context_dim: int = 128
    critic_context_memory_dim: int = 128
    critic_context_hidden_dim: int = 128
    critic_context_frac_alpha: float = 0.6
    freeze_critic_context_frac_alpha: bool = False
    actor_context_deterministic: bool = False
    critic_context_deterministic: bool = False
    context_aux_loss_weight: float = 0.0
    judge_kl_weight: float = 0.0
    actor_context_kl_weight: float = 0.0
    critic_context_kl_weight: float = 0.0
    save_best_checkpoint: bool = False
    best_checkpoint_filename: str = "best_checkpoint.pt"
    early_stop_patience_evals: int | None = None
    early_stop_min_delta: float = 0.0
    preload_trajectory_buffer_path: str | None = None
    preload_total_steps_from_buffer: bool = False
    preload_checkpoint_path: str | None = None
    save_trajectory_buffer_at_step: int | None = None
    trajectory_buffer_save_path: str | None = None
    force_random_actions_until_step: int | None = None
    result_root: str = "result"
    run_name: str | None = None
    device: str = "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAC with a history-based Judge reward module.")
    parser.add_argument("--variant", default="custom")
    parser.add_argument("--env", dest="env_name", choices=available_envs(), default="PendulumP")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory-buffer-episodes", type=int, default=256)
    parser.add_argument("--flat-buffer-capacity", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--start-steps", type=int, default=1_000)
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--reward-mode", choices=("env", "replace", "mix"), default="mix")
    parser.add_argument("--mix-alpha", type=float, default=0.5)
    parser.add_argument("--discount-gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--frac-alpha", type=float, default=0.6)
    parser.add_argument("--freeze-judge-frac-alpha", action="store_true")
    parser.add_argument("--judge-memory-dim", type=int, default=32)
    parser.add_argument("--judge-hidden-dim", type=int, default=64)
    parser.add_argument("--judge-lr", type=float, default=1e-3)
    parser.add_argument("--disable-judge", action="store_true")
    parser.add_argument("--no-judge-input-env-reward", action="store_true")
    parser.add_argument("--judge-reward-scale", type=float, default=None)
    parser.add_argument("--judge-reward-clip", type=float, default=None)
    parser.add_argument("--judge-mix-warmup-steps", type=int, default=None)
    parser.add_argument("--replay-mode", choices=("auto", "sequence", "flat"), default="auto")
    parser.add_argument("--trajectory-sampling-mode", choices=("uniform", "recent", "return"), default="uniform")
    parser.add_argument("--actor-context-mode", choices=("none", "fade"), default="fade")
    parser.add_argument("--actor-context-dim", type=int, default=128)
    parser.add_argument("--actor-context-memory-dim", type=int, default=128)
    parser.add_argument("--actor-context-hidden-dim", type=int, default=128)
    parser.add_argument("--actor-context-frac-alpha", type=float, default=0.6)
    parser.add_argument("--freeze-actor-context-frac-alpha", action="store_true")
    parser.add_argument("--actor-context-deterministic", action="store_true")
    parser.add_argument("--critic-context-mode", choices=("none", "fade"), default="fade")
    parser.add_argument("--critic-context-dim", type=int, default=128)
    parser.add_argument("--critic-context-memory-dim", type=int, default=128)
    parser.add_argument("--critic-context-hidden-dim", type=int, default=128)
    parser.add_argument("--critic-context-frac-alpha", type=float, default=0.6)
    parser.add_argument("--freeze-critic-context-frac-alpha", action="store_true")
    parser.add_argument("--critic-context-deterministic", action="store_true")
    parser.add_argument("--context-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--judge-kl-weight", type=float, default=0.0)
    parser.add_argument("--actor-context-kl-weight", type=float, default=0.0)
    parser.add_argument("--critic-context-kl-weight", type=float, default=0.0)
    parser.add_argument("--save-best-checkpoint", action="store_true")
    parser.add_argument("--best-checkpoint-filename", default="best_checkpoint.pt")
    parser.add_argument("--early-stop-patience-evals", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--preload-trajectory-buffer-path", default=None)
    parser.add_argument("--preload-total-steps-from-buffer", action="store_true")
    parser.add_argument("--preload-checkpoint-path", default=None)
    parser.add_argument("--save-trajectory-buffer-at-step", type=int, default=None)
    parser.add_argument("--trajectory-buffer-save-path", default=None)
    parser.add_argument("--force-random-actions-until-step", type=int, default=None)
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cpu")
    return parser


def parse_args() -> TrainingConfig:
    namespace = build_parser().parse_args()
    namespace.judge_enabled = not namespace.disable_judge
    del namespace.disable_judge
    namespace.judge_input_includes_env_reward = not namespace.no_judge_input_env_reward
    del namespace.no_judge_input_env_reward
    return TrainingConfig(**vars(namespace))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _validate_config(config: TrainingConfig) -> None:
    if not config.judge_enabled and config.reward_mode != "env":
        raise ValueError("reward_mode must be 'env' when judge_enabled is False.")
    if config.replay_mode == "flat" and (
        config.judge_enabled
        or config.actor_context_mode != "none"
        or config.critic_context_mode != "none"
    ):
        raise ValueError(
            "flat replay_mode only supports judge_disabled + actor_context_mode='none' + critic_context_mode='none'."
        )
    if config.judge_reward_scale is not None and config.judge_reward_scale <= 0.0:
        raise ValueError("judge_reward_scale must be positive when provided.")
    if config.judge_reward_clip is not None and config.judge_reward_clip <= 0.0:
        raise ValueError("judge_reward_clip must be positive when provided.")
    if config.judge_mix_warmup_steps is not None and config.judge_mix_warmup_steps < 0:
        raise ValueError("judge_mix_warmup_steps must be non-negative when provided.")
    if config.early_stop_patience_evals is not None and config.early_stop_patience_evals < 0:
        raise ValueError("early_stop_patience_evals must be non-negative when provided.")
    if config.early_stop_min_delta < 0.0:
        raise ValueError("early_stop_min_delta must be non-negative.")
    if config.save_trajectory_buffer_at_step is not None and config.save_trajectory_buffer_at_step < 0:
        raise ValueError("save_trajectory_buffer_at_step must be non-negative when provided.")
    if config.force_random_actions_until_step is not None and config.force_random_actions_until_step < 0:
        raise ValueError("force_random_actions_until_step must be non-negative when provided.")
    if config.context_aux_loss_weight < 0.0:
        raise ValueError("context_aux_loss_weight must be non-negative.")
    if config.hidden_layers < 1:
        raise ValueError("hidden_layers must be at least 1.")


def _resolve_hidden_sizes(config: TrainingConfig) -> tuple[int, ...]:
    return tuple([config.hidden_size] * config.hidden_layers)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _nan_to_none(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def _resolve_replay_mode(config: TrainingConfig) -> str:
    if config.replay_mode != "auto":
        return config.replay_mode
    if (
        not config.judge_enabled
        and config.actor_context_mode == "none"
        and config.critic_context_mode == "none"
    ):
        return "flat"
    return "sequence"


def _resolve_runtime_config(config: TrainingConfig) -> TrainingConfig:
    resolved = replace(config)
    pendulum_judge = resolved.judge_enabled and resolved.env_name.startswith("Pendulum")

    if resolved.judge_reward_scale is None:
        resolved.judge_reward_scale = 1.0
    if resolved.judge_reward_clip is None and pendulum_judge:
        resolved.judge_reward_clip = 20.0
    if resolved.judge_mix_warmup_steps is None:
        resolved.judge_mix_warmup_steps = 4_000 if pendulum_judge else 0
    return resolved


def evaluate_policy(
    agent: SACAgent,
    env_name: str,
    episodes: int,
    seed: int,
) -> float:
    returns: list[float] = []
    eval_env = make_env(env_name)
    for episode_index in range(episodes):
        obs, _ = eval_env.reset(seed=seed + episode_index)
        if hasattr(eval_env.action_space, "seed"):
            eval_env.action_space.seed(seed + episode_index)
        agent.reset_rollout_state()
        prev_action: np.ndarray | None = None
        prev_reward = 0.0
        episode_return = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = agent.select_action(
                obs,
                prev_action=prev_action,
                prev_reward=prev_reward,
                deterministic=True,
            )
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_return += float(reward)
            prev_action = np.asarray(action, dtype=np.float32).reshape(-1)
            prev_reward = float(reward)
        returns.append(episode_return)
    eval_env.close()
    return float(np.mean(returns))


def run_training(config: TrainingConfig) -> dict[str, object]:
    config = _resolve_runtime_config(config)
    _validate_config(config)
    replay_mode = _resolve_replay_mode(config)
    judge_input_uses_env_reward = config.judge_input_includes_env_reward
    start_time = perf_counter()
    run_name = config.run_name or _default_run_name(config)
    run_dir = Path(config.result_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)
    env = make_env(config.env_name)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

    judge_input_dim = obs_dim + action_dim + obs_dim + 1
    actor_context_input_dim = obs_dim + action_dim + 1
    critic_context_input_dim = obs_dim + action_dim + 1
    agent = SACAgent(
        obs_dim,
        action_dim,
        action_low,
        action_high,
        gamma=config.discount_gamma,
        tau=config.tau,
        lr=config.lr,
        hidden_sizes=_resolve_hidden_sizes(config),
        device=config.device,
        actor_rollout_context_len=config.seq_len,
        actor_context_input_dim=actor_context_input_dim
        if config.actor_context_mode == "fade"
        else None,
        actor_context_dim=config.actor_context_dim,
        actor_context_memory_dim=config.actor_context_memory_dim,
        actor_context_hidden_dim=config.actor_context_hidden_dim,
        actor_context_frac_alpha=config.actor_context_frac_alpha,
        actor_context_learnable_frac_alpha=not config.freeze_actor_context_frac_alpha,
        critic_context_input_dim=critic_context_input_dim
        if config.critic_context_mode == "fade"
        else None,
        critic_context_dim=config.critic_context_dim,
        critic_context_memory_dim=config.critic_context_memory_dim,
        critic_context_hidden_dim=config.critic_context_hidden_dim,
        critic_context_frac_alpha=config.critic_context_frac_alpha,
        critic_context_learnable_frac_alpha=not config.freeze_critic_context_frac_alpha,
        actor_context_deterministic=config.actor_context_deterministic,
        critic_context_deterministic=config.critic_context_deterministic,
        context_aux_prediction_dim=obs_dim + 1 if config.context_aux_loss_weight > 0.0 else 0,
        context_aux_loss_weight=config.context_aux_loss_weight,
        context_kl_weight_actor=config.actor_context_kl_weight,
        context_kl_weight_critic=config.critic_context_kl_weight,
    )
    trajectory_buffer = (
        TrajectoryBuffer(
            max_episodes=config.trajectory_buffer_episodes,
            discount_gamma=config.discount_gamma,
        )
        if replay_mode == "sequence"
        else None
    )
    flat_buffer = (
        FlatReplayBuffer(
            capacity=config.flat_buffer_capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        if replay_mode == "flat"
        else None
    )
    judge = (
        HistoryRewardJudge(
            input_dim=judge_input_dim,
            memory_dim=config.judge_memory_dim,
            hidden_dim=config.judge_hidden_dim,
            frac_alpha=config.frac_alpha,
            learnable_frac_alpha=not config.freeze_judge_frac_alpha,
            lr=config.judge_lr,
            device=config.device,
        )
        if config.judge_enabled
        else None
    )

    if config.preload_checkpoint_path is not None:
        checkpoint_blob = torch.load(config.preload_checkpoint_path, map_location=agent.device)
        agent.load_state_dict(checkpoint_blob["agent"])
        if judge is not None and checkpoint_blob.get("judge") is not None:
            judge.load_state_dict(checkpoint_blob["judge"])

    if trajectory_buffer is not None and config.preload_trajectory_buffer_path is not None:
        trajectory_buffer.load(config.preload_trajectory_buffer_path)

    total_steps = 0
    episode_index = 0
    if trajectory_buffer is not None and config.preload_total_steps_from_buffer:
        total_steps = min(config.steps, trajectory_buffer.total_transitions)
        episode_index = trajectory_buffer.num_episodes
    eval_steps: list[int] = []
    eval_returns: list[float] = []
    best_checkpoint_eval_step: int | None = None
    best_checkpoint_eval_return: float | None = None
    best_checkpoint_path: str | None = None
    best_eval_so_far = -float("inf")
    no_improve_evals = 0
    stop_requested = False
    buffer_snapshot_saved = False
    episode_rows: list[dict[str, float | int | bool | None]] = []
    update_rows: list[dict[str, float | int | None]] = []
    last_update_stats: dict[str, float | None] = {
        "judge_loss": None,
        "judge_decomp_loss": None,
        "judge_kl_loss": None,
        "frac_alpha": config.frac_alpha if config.judge_enabled else None,
        "mean_r_env": None,
        "mean_r_tilde": None,
        "mean_r_train": None,
        "actor_context_std": None,
        "actor_frac_alpha": None,
        "critic_context_std": None,
        "critic_frac_alpha": None,
        "actor_kl_loss": None,
        "critic_kl_loss": None,
    }

    def build_checkpoint_payload() -> dict[str, object]:
        return {
            "agent": agent.state_dict(),
            "judge": None if judge is None else judge.state_dict(),
            "config": asdict(config),
        }

    def record_eval(eval_return: float) -> None:
        nonlocal best_checkpoint_eval_step
        nonlocal best_checkpoint_eval_return
        nonlocal best_checkpoint_path
        nonlocal best_eval_so_far
        nonlocal no_improve_evals
        nonlocal stop_requested

        eval_steps.append(total_steps)
        eval_returns.append(eval_return)
        improved = eval_return > best_eval_so_far + config.early_stop_min_delta
        if improved:
            best_eval_so_far = eval_return
            no_improve_evals = 0
            if config.save_best_checkpoint:
                checkpoint_path = run_dir / config.best_checkpoint_filename
                torch.save(build_checkpoint_payload(), checkpoint_path)
                best_checkpoint_eval_step = total_steps
                best_checkpoint_eval_return = eval_return
                best_checkpoint_path = str(checkpoint_path)
        else:
            no_improve_evals += 1
            if config.early_stop_patience_evals is not None and no_improve_evals >= config.early_stop_patience_evals:
                stop_requested = True

    while total_steps < config.steps and not stop_requested:
        obs, _ = env.reset(seed=config.seed + episode_index)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(config.seed + episode_index)
        agent.reset_rollout_state()
        prev_action_for_actor: np.ndarray | None = None
        prev_reward_for_actor = 0.0

        episode_obs = [np.asarray(obs, dtype=np.float32)]
        episode_actions: list[np.ndarray] = []
        episode_rewards_env: list[float] = []
        episode_terminateds: list[float] = []
        episode_truncateds: list[float] = []
        episode_env_return = 0.0
        episode_length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and total_steps < config.steps and not stop_requested:
            random_action_phase = total_steps < config.start_steps or (
                config.force_random_actions_until_step is not None
                and total_steps < config.force_random_actions_until_step
            )
            if random_action_phase:
                if config.actor_context_mode == "fade":
                    agent.append_rollout_context(obs, prev_action_for_actor, prev_reward_for_actor)
                action = np.asarray(env.action_space.sample(), dtype=np.float32)
            else:
                action = agent.select_action(
                    obs,
                    prev_action=prev_action_for_actor,
                    prev_reward=prev_reward_for_actor,
                    deterministic=False,
                )

            next_obs, env_reward, terminated, truncated, _ = env.step(action)

            episode_actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            episode_rewards_env.append(float(env_reward))
            episode_terminateds.append(float(terminated))
            episode_truncateds.append(float(truncated))
            episode_obs.append(np.asarray(next_obs, dtype=np.float32))
            if flat_buffer is not None:
                flat_buffer.add(obs, action, env_reward, next_obs, float(terminated), float(truncated))
            obs = next_obs
            prev_action_for_actor = np.asarray(action, dtype=np.float32).reshape(-1)
            prev_reward_for_actor = float(env_reward)
            episode_env_return += float(env_reward)
            episode_length += 1
            total_steps += 1

        if trajectory_buffer is not None:
            trajectory_buffer.add_episode(
                np.asarray(episode_obs, dtype=np.float32),
                np.asarray(episode_actions, dtype=np.float32),
                np.asarray(episode_rewards_env, dtype=np.float32),
                terminateds=np.asarray(episode_terminateds, dtype=np.float32),
                truncateds=np.asarray(episode_truncateds, dtype=np.float32),
            )
            if (
                config.save_trajectory_buffer_at_step is not None
                and not buffer_snapshot_saved
                and total_steps >= config.save_trajectory_buffer_at_step
            ):
                snapshot_path = (
                    Path(config.trajectory_buffer_save_path)
                    if config.trajectory_buffer_save_path is not None
                    else run_dir / f"trajectory_buffer_step{config.save_trajectory_buffer_at_step}.pkl"
                )
                trajectory_buffer.save(snapshot_path)
                buffer_snapshot_saved = True

        judge_return_mean = float("nan")
        train_reward_mean = float("nan")

        if (
            trajectory_buffer is not None
            and trajectory_buffer.total_transitions >= max(config.update_after, config.seq_len)
            and trajectory_buffer.num_episodes > 0
        ):
            update_count = max(1, episode_length * config.update_every)
            for update_index in range(update_count):
                batch = trajectory_buffer.sample_sequences(
                    config.batch_size,
                    config.seq_len,
                    burn_in=config.burn_in,
                    sampling_mode=config.trajectory_sampling_mode,
                )
                tensor_batch = _to_torch_batch(batch, device=agent.device)
                judge_inputs = build_judge_inputs(
                    tensor_batch["obs"],
                    tensor_batch["actions"],
                    tensor_batch["next_obs"],
                    tensor_batch["rewards_env"],
                    include_env_reward=judge_input_uses_env_reward,
                )
                critic_context_inputs = build_context_inputs(
                    tensor_batch["obs"],
                    tensor_batch["prev_actions"],
                    tensor_batch["prev_rewards"],
                )
                next_actor_context_inputs = build_context_inputs(
                    tensor_batch["next_obs"],
                    tensor_batch["actions"],
                    tensor_batch["rewards_env"],
                )
                if judge is not None:
                    judge_stats = judge.update_from_batch(
                        judge_inputs=judge_inputs,
                        rewards_env=tensor_batch["rewards_env"],
                        mask=tensor_batch["mask"],
                        loss_mask=tensor_batch["loss_mask"],
                        kl_weight=config.judge_kl_weight,
                    )
                    with torch.no_grad():
                        judge_outputs = judge.forward_sequence(judge_inputs, tensor_batch["mask"])
                        r_tilde = _prepare_judge_rewards(
                            judge_outputs["r_tilde"],
                            scale=config.judge_reward_scale,
                            clip=config.judge_reward_clip,
                        )
                else:
                    judge_stats = {"judge_loss": None, "judge_decomp_loss": None, "judge_kl_loss": None, "frac_alpha": None}
                    r_tilde = torch.zeros_like(tensor_batch["rewards_env"])

                r_train = _fuse_reward_tensor(
                    tensor_batch["rewards_env"],
                    r_tilde,
                    reward_mode=config.reward_mode,
                    mix_alpha=config.mix_alpha,
                    global_step=total_steps,
                    mix_warmup_steps=config.judge_mix_warmup_steps,
                )
                valid = tensor_batch["loss_mask"] > 0
                if valid.sum() == 0:
                    continue
                sac_batch = {
                    "obs": tensor_batch["obs"],
                    "actions": tensor_batch["actions"],
                    "rewards": r_train,
                    "rewards_env": tensor_batch["rewards_env"],
                    "next_obs": tensor_batch["next_obs"],
                    "dones": tensor_batch["dones"],
                    "truncateds": tensor_batch["truncateds"],
                    "mask": tensor_batch["mask"],
                    "loss_mask": tensor_batch["loss_mask"],
                    "actor_context_inputs": critic_context_inputs,
                    "next_actor_context_inputs": next_actor_context_inputs,
                    "critic_context_inputs": critic_context_inputs,
                    "next_critic_context_inputs": next_actor_context_inputs,
                }
                sac_stats = agent.update_sequence(sac_batch)
                judge_return_mean = float(r_tilde[valid].mean().item()) if judge is not None else float("nan")
                train_reward_mean = float(r_train[valid].mean().item())
                last_update_stats = {
                    "critic_loss": sac_stats.critic_loss,
                    "actor_loss": sac_stats.actor_loss,
                    "alpha_loss": sac_stats.alpha_loss,
                    "entropy_alpha": sac_stats.alpha,
                    "judge_loss": _optional_float(judge_stats["judge_loss"]),
                    "judge_decomp_loss": _optional_float(judge_stats["judge_decomp_loss"]),
                    "judge_kl_loss": _optional_float(judge_stats["judge_kl_loss"]),
                    "frac_alpha": _optional_float(judge_stats["frac_alpha"]),
                    "mean_r_env": float(tensor_batch["rewards_env"][valid].mean().item()),
                    "mean_r_tilde": _nan_to_none(judge_return_mean),
                    "mean_r_train": train_reward_mean,
                    "actor_context_std": sac_stats.actor_context_std,
                    "actor_frac_alpha": sac_stats.actor_frac_alpha,
                    "actor_aux_loss": sac_stats.actor_aux_loss,
                    "critic_context_std": sac_stats.critic_context_std,
                    "critic_frac_alpha": sac_stats.critic_frac_alpha,
                    "critic_aux_loss": sac_stats.critic_aux_loss,
                    "actor_kl_loss": sac_stats.actor_kl_loss,
                    "critic_kl_loss": sac_stats.critic_kl_loss,
                }
                update_rows.append(
                    {
                        "global_step": total_steps,
                        "update_index": update_index,
                        "critic_loss": sac_stats.critic_loss,
                        "actor_loss": sac_stats.actor_loss,
                        "alpha_loss": sac_stats.alpha_loss,
                        "entropy_alpha": sac_stats.alpha,
                        "judge_loss": _optional_float(judge_stats["judge_loss"]),
                        "judge_decomp_loss": _optional_float(judge_stats["judge_decomp_loss"]),
                        "judge_kl_loss": _optional_float(judge_stats["judge_kl_loss"]),
                        "frac_alpha": _optional_float(judge_stats["frac_alpha"]),
                        "mean_r_env": float(tensor_batch["rewards_env"][valid].mean().item()),
                        "mean_r_tilde": _nan_to_none(judge_return_mean),
                        "mean_r_train": train_reward_mean,
                        "actor_context_std": sac_stats.actor_context_std,
                        "actor_frac_alpha": sac_stats.actor_frac_alpha,
                        "actor_aux_loss": sac_stats.actor_aux_loss,
                        "critic_context_std": sac_stats.critic_context_std,
                        "critic_frac_alpha": sac_stats.critic_frac_alpha,
                        "critic_aux_loss": sac_stats.critic_aux_loss,
                        "actor_kl_loss": sac_stats.actor_kl_loss,
                        "critic_kl_loss": sac_stats.critic_kl_loss,
                    }
                )
        elif (
            flat_buffer is not None
            and total_steps >= config.update_after
            and flat_buffer.size >= config.batch_size
        ):
            update_count = max(1, episode_length * config.update_every)
            for update_index in range(update_count):
                batch = flat_buffer.sample(config.batch_size)
                tensor_batch = _to_torch_transition_batch(batch, device=agent.device)
                sac_stats = agent.update(tensor_batch)
                train_reward_mean = float(tensor_batch["rewards"].mean().item())
                last_update_stats = {
                    "critic_loss": sac_stats.critic_loss,
                    "actor_loss": sac_stats.actor_loss,
                    "alpha_loss": sac_stats.alpha_loss,
                    "entropy_alpha": sac_stats.alpha,
                    "judge_loss": None,
                    "judge_decomp_loss": None,
                    "judge_kl_loss": None,
                    "frac_alpha": None,
                    "mean_r_env": float(tensor_batch["rewards"].mean().item()),
                    "mean_r_tilde": None,
                    "mean_r_train": train_reward_mean,
                    "actor_context_std": sac_stats.actor_context_std,
                    "actor_frac_alpha": sac_stats.actor_frac_alpha,
                    "actor_aux_loss": sac_stats.actor_aux_loss,
                    "critic_context_std": sac_stats.critic_context_std,
                    "critic_frac_alpha": sac_stats.critic_frac_alpha,
                    "critic_aux_loss": sac_stats.critic_aux_loss,
                    "actor_kl_loss": sac_stats.actor_kl_loss,
                    "critic_kl_loss": sac_stats.critic_kl_loss,
                }
                update_rows.append(
                    {
                        "global_step": total_steps,
                        "update_index": update_index,
                        "critic_loss": sac_stats.critic_loss,
                        "actor_loss": sac_stats.actor_loss,
                        "alpha_loss": sac_stats.alpha_loss,
                        "entropy_alpha": sac_stats.alpha,
                        "judge_loss": None,
                        "judge_decomp_loss": None,
                        "judge_kl_loss": None,
                        "frac_alpha": None,
                        "mean_r_env": float(tensor_batch["rewards"].mean().item()),
                        "mean_r_tilde": None,
                        "mean_r_train": train_reward_mean,
                        "actor_context_std": sac_stats.actor_context_std,
                        "actor_frac_alpha": sac_stats.actor_frac_alpha,
                        "actor_aux_loss": sac_stats.actor_aux_loss,
                        "critic_context_std": sac_stats.critic_context_std,
                        "critic_frac_alpha": sac_stats.critic_frac_alpha,
                        "critic_aux_loss": sac_stats.critic_aux_loss,
                        "actor_kl_loss": sac_stats.actor_kl_loss,
                        "critic_kl_loss": sac_stats.critic_kl_loss,
                    }
                )

        while config.eval_interval > 0 and (
            len(eval_steps) == 0 and total_steps >= config.eval_interval
            or (eval_steps and total_steps >= eval_steps[-1] + config.eval_interval)
        ):
            eval_return = evaluate_policy(
                agent,
                config.env_name,
                config.eval_episodes,
                seed=config.seed + 10_000 + len(eval_steps) * 100,
            )
            record_eval(eval_return)
            _elapsed = perf_counter() - start_time
            print(
                f"[eval] step={total_steps}/{config.steps}"
                f"  eval_return={eval_return:.1f}"
                f"  best={best_eval_so_far:.1f}"
                f"  time={_elapsed:.0f}s",
                flush=True,
            )

        episode_rows.append(
            {
                "episode": episode_index,
                "steps": episode_length,
                "env_return": episode_env_return,
                "judge_return_mean": _nan_to_none(judge_return_mean),
                "train_reward_mean": _nan_to_none(train_reward_mean),
                "frac_alpha": last_update_stats.get("frac_alpha"),
                "trajectory_buffer_episodes": trajectory_buffer.num_episodes if trajectory_buffer is not None else 0,
            }
        )
        if episode_index % 10 == 0 or episode_length >= 100:
            _elapsed = perf_counter() - start_time
            print(
                f"[ep {episode_index}] step={total_steps}/{config.steps}"
                f"  len={episode_length}"
                f"  env_return={episode_env_return:.1f}"
                f"  time={_elapsed:.0f}s",
                flush=True,
            )
        episode_index += 1

    if not eval_steps or eval_steps[-1] != total_steps:
        eval_return = evaluate_policy(
            agent,
            config.env_name,
            config.eval_episodes,
            seed=config.seed + 20_000,
        )
        record_eval(eval_return)

    best_eval_index = int(np.argmax(np.asarray(eval_returns, dtype=np.float32)))

    episode_array = {
        "episode": np.asarray([row["episode"] for row in episode_rows], dtype=np.int32),
        "steps": np.asarray([row["steps"] for row in episode_rows], dtype=np.int32),
        "env_return": np.asarray([row["env_return"] for row in episode_rows], dtype=np.float32),
        "judge_return_mean": np.asarray([np.nan if row["judge_return_mean"] is None else row["judge_return_mean"] for row in episode_rows], dtype=np.float32),
        "train_reward_mean": np.asarray([np.nan if row["train_reward_mean"] is None else row["train_reward_mean"] for row in episode_rows], dtype=np.float32),
    }
    np.savez(
        run_dir / "metrics.npz",
        eval_steps=np.asarray(eval_steps, dtype=np.int32),
        eval_returns=np.asarray(eval_returns, dtype=np.float32),
        **episode_array,
    )
    _write_episode_csv(run_dir / "episode_metrics.csv", episode_rows)
    _write_episode_csv(run_dir / "update_metrics.csv", update_rows)

    torch.save(build_checkpoint_payload(), run_dir / "checkpoint.pt")

    summary = {
        **asdict(config),
        "run_dir": str(run_dir),
        "episodes": episode_index,
        "total_steps": total_steps,
        "stopped_early": stop_requested,
        "judge_input_uses_env_reward": judge_input_uses_env_reward,
        "wall_time_seconds": perf_counter() - start_time,
        "best_eval_step": eval_steps[best_eval_index],
        "best_eval_return": eval_returns[best_eval_index],
        "final_eval_return": eval_returns[-1],
        "best_checkpoint_eval_step": best_checkpoint_eval_step,
        "best_checkpoint_eval_return": best_checkpoint_eval_return,
        "best_checkpoint_path": best_checkpoint_path,
        "last_update": last_update_stats,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    env.close()
    return summary


def _default_run_name(config: TrainingConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config.env_name}_{config.reward_mode}_seed{config.seed}_{timestamp}"


def _write_episode_csv(path: Path, rows: list[dict[str, float | int | bool | None]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _to_torch_batch(batch: TrajectoryBatch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.as_tensor(batch.obs, dtype=torch.float32, device=device),
        "next_obs": torch.as_tensor(batch.next_obs, dtype=torch.float32, device=device),
        "actions": torch.as_tensor(batch.actions, dtype=torch.float32, device=device),
        "rewards_env": torch.as_tensor(batch.rewards_env, dtype=torch.float32, device=device),
        "prev_rewards": torch.as_tensor(batch.prev_rewards, dtype=torch.float32, device=device),
        "dones": torch.as_tensor(batch.dones, dtype=torch.float32, device=device),
        "truncateds": torch.as_tensor(batch.truncateds, dtype=torch.float32, device=device),
        "returns_mc": torch.as_tensor(batch.returns_mc, dtype=torch.float32, device=device),
        "mask": torch.as_tensor(batch.mask, dtype=torch.float32, device=device),
        "loss_mask": torch.as_tensor(batch.loss_mask, dtype=torch.float32, device=device),
        "prev_actions": torch.as_tensor(batch.prev_actions, dtype=torch.float32, device=device),
    }


def _prepare_judge_rewards(
    rewards_tilde: torch.Tensor,
    *,
    scale: float | None,
    clip: float | None,
) -> torch.Tensor:
    return _scale_and_clip_tensor(rewards_tilde, scale=scale, clip=clip)


def _to_torch_transition_batch(batch: TransitionBatch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.as_tensor(batch.obs, dtype=torch.float32, device=device),
        "actions": torch.as_tensor(batch.actions, dtype=torch.float32, device=device),
        "rewards": torch.as_tensor(batch.rewards, dtype=torch.float32, device=device).unsqueeze(-1),
        "next_obs": torch.as_tensor(batch.next_obs, dtype=torch.float32, device=device),
        "dones": torch.as_tensor(batch.dones, dtype=torch.float32, device=device).unsqueeze(-1),
        "truncateds": torch.as_tensor(batch.truncateds, dtype=torch.float32, device=device).unsqueeze(-1),
    }


def _fuse_reward_tensor(
    rewards_env: torch.Tensor,
    rewards_tilde: torch.Tensor,
    *,
    reward_mode: str,
    mix_alpha: float,
    global_step: int,
    mix_warmup_steps: int,
) -> torch.Tensor:
    judge_progress = _judge_warmup_progress(global_step, mix_warmup_steps)
    if reward_mode == "env":
        return rewards_env
    if reward_mode == "replace":
        return (1.0 - judge_progress) * rewards_env + judge_progress * rewards_tilde
    if reward_mode == "mix":
        effective_mix_alpha = 1.0 - judge_progress * (1.0 - mix_alpha)
        return effective_mix_alpha * rewards_env + (1.0 - effective_mix_alpha) * rewards_tilde
    raise ValueError(f"Unsupported reward_mode: {reward_mode}")


def _scale_and_clip_tensor(
    values: torch.Tensor,
    *,
    scale: float | None,
    clip: float | None,
) -> torch.Tensor:
    scaled = values if scale is None else values * scale
    if clip is None:
        return scaled
    return torch.clamp(scaled, min=-clip, max=clip)


def _judge_warmup_progress(global_step: int, mix_warmup_steps: int) -> float:
    if mix_warmup_steps <= 0:
        return 1.0
    return float(min(max(global_step / mix_warmup_steps, 0.0), 1.0))


def main() -> None:
    config = parse_args()
    summary = run_training(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
