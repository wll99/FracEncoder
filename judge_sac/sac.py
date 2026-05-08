"""SAC implementation with fractional context encoders and temporal KL regularization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from .judge import GaussianContextEncoder, temporal_kl_loss


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def _mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_size))
        layers.append(nn.ReLU())
        prev_dim = hidden_size
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
        action_low: np.ndarray,
        action_high: np.ndarray,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.backbone = _mlp(obs_dim + context_dim, hidden_sizes, hidden_sizes[-1])
        self.mean_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def _distribution(
        self,
        obs: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.context_dim > 0:
            if context is None:
                raise ValueError("context tensor is required when context_dim > 0.")
            inputs = torch.cat([obs, context], dim=-1)
        else:
            inputs = obs
        hidden = self.backbone(inputs)
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        context: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._distribution(obs, context)
        std = log_std.exp()
        normal = Normal(mean, std)
        pre_tanh = mean if deterministic else normal.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias

        log_prob = normal.log_prob(pre_tanh) - torch.log(
            self.action_scale * (1.0 - squashed.pow(2)) + EPS
        )
        return action, log_prob.sum(dim=-1, keepdim=True)

    def act(
        self,
        obs: np.ndarray,
        device: torch.device,
        deterministic: bool = False,
        context: np.ndarray | None = None,
    ) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        context_tensor = None
        if context is not None:
            context_tensor = torch.as_tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.sample(obs_tensor, context_tensor, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.net = _mlp(obs_dim + action_dim + context_dim, hidden_sizes, 1)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.context_dim > 0:
            if context is None:
                raise ValueError("context tensor is required when context_dim > 0.")
            inputs = [obs, action, context]
        else:
            inputs = [obs, action]
        return self.net(torch.cat(inputs, dim=-1))


@dataclass
class SACUpdateStats:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    actor_aux_loss: float | None = None
    critic_aux_loss: float | None = None
    actor_context_std: float | None = None
    actor_frac_alpha: float | None = None
    critic_context_std: float | None = None
    critic_frac_alpha: float | None = None
    actor_kl_loss: float | None = None
    critic_kl_loss: float | None = None


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        *,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        hidden_sizes: tuple[int, ...] = (256, 256),
        device: str = "cpu",
        actor_rollout_context_len: int | None = None,
        actor_context_input_dim: int | None = None,
        actor_context_dim: int = 128,
        actor_context_memory_dim: int = 128,
        actor_context_hidden_dim: int = 128,
        actor_context_frac_alpha: float = 0.6,
        actor_context_learnable_frac_alpha: bool = True,
        critic_context_input_dim: int | None = None,
        critic_context_dim: int = 128,
        critic_context_memory_dim: int = 128,
        critic_context_hidden_dim: int = 128,
        critic_context_frac_alpha: float = 0.6,
        critic_context_learnable_frac_alpha: bool = True,
        actor_context_deterministic: bool = False,
        critic_context_deterministic: bool = False,
        context_aux_prediction_dim: int = 0,
        context_aux_loss_weight: float = 0.0,
        context_kl_weight_actor: float = 0.0,
        context_kl_weight_critic: float = 0.0,
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -float(action_dim)
        self.actor_rollout_context_len = actor_rollout_context_len
        self.use_actor_context_encoder = actor_context_input_dim is not None and actor_context_dim > 0
        self.use_critic_context_encoder = critic_context_input_dim is not None and critic_context_dim > 0
        self.context_aux_loss_weight = context_aux_loss_weight
        self.context_kl_weight_actor = context_kl_weight_actor
        self.context_kl_weight_critic = context_kl_weight_critic
        self.actor_context_deterministic = actor_context_deterministic
        self.critic_context_deterministic = critic_context_deterministic
        self.actor_context_dim = actor_context_dim if self.use_actor_context_encoder else 0
        self.critic_context_dim = critic_context_dim if self.use_critic_context_encoder else 0
        self._rollout_actor_inputs: list[np.ndarray] = []

        self.actor = SquashedGaussianActor(
            obs_dim,
            action_dim,
            hidden_sizes,
            action_low,
            action_high,
            context_dim=self.actor_context_dim,
        ).to(self.device)
        self.q1 = QNetwork(obs_dim, action_dim, hidden_sizes, context_dim=self.critic_context_dim).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_sizes, context_dim=self.critic_context_dim).to(self.device)
        self.target_q1 = QNetwork(
            obs_dim,
            action_dim,
            hidden_sizes,
            context_dim=self.critic_context_dim,
        ).to(self.device)
        self.target_q2 = QNetwork(
            obs_dim,
            action_dim,
            hidden_sizes,
            context_dim=self.critic_context_dim,
        ).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        actor_parameters: list[nn.Parameter] = list(self.actor.parameters())
        if self.use_actor_context_encoder:
            self.actor_encoder = GaussianContextEncoder(
                input_dim=actor_context_input_dim,
                memory_dim=actor_context_memory_dim,
                hidden_dim=actor_context_hidden_dim,
                latent_dim=actor_context_dim,
                prediction_dim=context_aux_prediction_dim,
                frac_alpha=actor_context_frac_alpha,
                learnable_frac_alpha=actor_context_learnable_frac_alpha,
                device=device,
            ).to(self.device)
            self.target_actor_encoder = GaussianContextEncoder(
                input_dim=actor_context_input_dim,
                memory_dim=actor_context_memory_dim,
                hidden_dim=actor_context_hidden_dim,
                latent_dim=actor_context_dim,
                prediction_dim=context_aux_prediction_dim,
                frac_alpha=actor_context_frac_alpha,
                learnable_frac_alpha=actor_context_learnable_frac_alpha,
                device=device,
            ).to(self.device)
            self.target_actor_encoder.load_state_dict(self.actor_encoder.state_dict())
            actor_parameters.extend(self.actor_encoder.parameters())
        else:
            self.actor_encoder = None
            self.target_actor_encoder = None

        critic_parameters: list[nn.Parameter] = list(self.q1.parameters()) + list(self.q2.parameters())
        if self.use_critic_context_encoder:
            self.critic_encoder = GaussianContextEncoder(
                input_dim=critic_context_input_dim,
                memory_dim=critic_context_memory_dim,
                hidden_dim=critic_context_hidden_dim,
                latent_dim=critic_context_dim,
                prediction_dim=context_aux_prediction_dim,
                frac_alpha=critic_context_frac_alpha,
                learnable_frac_alpha=critic_context_learnable_frac_alpha,
                device=device,
            ).to(self.device)
            self.target_critic_encoder = GaussianContextEncoder(
                input_dim=critic_context_input_dim,
                memory_dim=critic_context_memory_dim,
                hidden_dim=critic_context_hidden_dim,
                latent_dim=critic_context_dim,
                prediction_dim=context_aux_prediction_dim,
                frac_alpha=critic_context_frac_alpha,
                learnable_frac_alpha=critic_context_learnable_frac_alpha,
                device=device,
            ).to(self.device)
            self.target_critic_encoder.load_state_dict(self.critic_encoder.state_dict())
            critic_parameters.extend(self.critic_encoder.parameters())
        else:
            self.critic_encoder = None
            self.target_critic_encoder = None

        self.actor_optimizer = torch.optim.Adam(actor_parameters, lr=lr)
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=lr)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def reset_rollout_state(self) -> None:
        self._rollout_actor_inputs.clear()

    def append_rollout_context(
        self,
        obs: np.ndarray,
        prev_action: np.ndarray | None,
        prev_reward: float,
    ) -> None:
        if not self.use_actor_context_encoder:
            return
        obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
        if prev_action is None:
            prev_action_array = np.zeros((self.actor.action_scale.shape[0],), dtype=np.float32)
        else:
            prev_action_array = np.asarray(prev_action, dtype=np.float32).reshape(-1)
        step_input = np.concatenate(
            [obs_array, prev_action_array, np.asarray([prev_reward], dtype=np.float32)],
            axis=0,
        )
        self._rollout_actor_inputs.append(step_input)
        if self.actor_rollout_context_len is not None and len(self._rollout_actor_inputs) > self.actor_rollout_context_len:
            self._rollout_actor_inputs = self._rollout_actor_inputs[-self.actor_rollout_context_len :]

    def _rollout_actor_context(self, deterministic: bool) -> np.ndarray | None:
        if not self.use_actor_context_encoder:
            return None
        if not self._rollout_actor_inputs:
            raise ValueError("Actor rollout context is empty. Append the current step before selecting an action.")
        sequence = torch.as_tensor(
            np.asarray(self._rollout_actor_inputs, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask = torch.ones((1, sequence.shape[1]), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs = self.actor_encoder.forward_sequence(
                sequence,
                mask,
                deterministic=deterministic or self.actor_context_deterministic,
            )
        return outputs["z"][:, -1].squeeze(0).cpu().numpy()

    def select_action(
        self,
        obs: np.ndarray,
        prev_action: np.ndarray | None = None,
        prev_reward: float = 0.0,
        deterministic: bool = False,
    ) -> np.ndarray:
        if self.use_actor_context_encoder:
            self.append_rollout_context(obs, prev_action, prev_reward)
            actor_context = self._rollout_actor_context(deterministic=deterministic)
        else:
            actor_context = None
        return self.actor.act(
            obs,
            self.device,
            deterministic=deterministic,
            context=actor_context,
        ).astype(np.float32)

    def update(self, batch: dict[str, torch.Tensor]) -> SACUpdateStats:
        return self._update_flat_batch(
            obs=batch["obs"],
            actions=batch["actions"],
            rewards=batch["rewards"],
            next_obs=batch["next_obs"],
            dones=batch["dones"],
        )

    def update_sequence(self, batch: dict[str, torch.Tensor]) -> SACUpdateStats:
        valid = batch["loss_mask"] > 0
        if valid.sum() == 0:
            raise ValueError("No valid sequence elements available for SAC update.")

        mask = batch["mask"]
        loss_mask = batch["loss_mask"]
        actor_context_stats: dict[str, torch.Tensor] | None = None
        next_actor_context_stats: dict[str, torch.Tensor] | None = None
        critic_context_stats: dict[str, torch.Tensor] | None = None
        next_critic_context_stats: dict[str, torch.Tensor] | None = None
        actor_aux_loss = None
        critic_aux_loss = None
        actor_kl_loss_tensor: torch.Tensor | None = None
        critic_kl_loss_tensor: torch.Tensor | None = None

        if self.use_actor_context_encoder:
            actor_context_stats = self.actor_encoder.forward_sequence(
                batch["actor_context_inputs"],
                mask,
                deterministic=self.actor_context_deterministic,
            )
            with torch.no_grad():
                next_actor_context_stats = self.target_actor_encoder.forward_sequence(
                    batch["next_actor_context_inputs"],
                    mask,
                    deterministic=self.actor_context_deterministic,
                )
            if self.context_kl_weight_actor > 0.0:
                actor_kl_loss_tensor = temporal_kl_loss(
                    actor_context_stats["mean"], actor_context_stats["std"], mask, loss_mask,
                )

        if self.use_critic_context_encoder:
            critic_context_stats = self.critic_encoder.forward_sequence(
                batch["critic_context_inputs"],
                mask,
                deterministic=self.critic_context_deterministic,
            )
            with torch.no_grad():
                next_critic_context_stats = self.target_critic_encoder.forward_sequence(
                    batch["next_critic_context_inputs"],
                    mask,
                    deterministic=self.critic_context_deterministic,
                )
            if self.context_kl_weight_critic > 0.0:
                critic_kl_loss_tensor = temporal_kl_loss(
                    critic_context_stats["mean"], critic_context_stats["std"], mask, loss_mask,
                )

        obs = batch["obs"][valid]
        actions = batch["actions"][valid]
        rewards = batch["rewards"][valid].unsqueeze(-1)
        next_obs = batch["next_obs"][valid]
        dones = batch["dones"][valid].unsqueeze(-1)

        actor_context = next_actor_context = None
        actor_context_std = None
        actor_frac_alpha = None
        if actor_context_stats is not None and next_actor_context_stats is not None:
            actor_context = actor_context_stats["z"][valid]
            next_actor_context = next_actor_context_stats["z"][valid]
            actor_context_std = float(actor_context_stats["std"][valid].mean().item())
            actor_frac_alpha = float(actor_context_stats["frac_alpha"].item())
            if self.context_aux_loss_weight > 0.0 and self.actor_encoder is not None and self.actor_encoder.prediction_head is not None:
                actor_aux_target = torch.cat([batch["next_obs"][valid], batch["rewards_env"][valid].unsqueeze(-1)], dim=-1)
                actor_aux_pred = self.actor_encoder.predict_auxiliary(actor_context_stats["states"][valid])
                actor_aux_loss = F.mse_loss(actor_aux_pred, actor_aux_target)

        critic_context = target_critic_context = None
        critic_context_std = None
        critic_frac_alpha = None
        if critic_context_stats is not None and next_critic_context_stats is not None:
            critic_context = critic_context_stats["z"][valid]
            target_critic_context = next_critic_context_stats["z"][valid]
            critic_context_std = float(critic_context_stats["std"][valid].mean().item())
            critic_frac_alpha = float(critic_context_stats["frac_alpha"].item())
            if self.context_aux_loss_weight > 0.0 and self.critic_encoder is not None and self.critic_encoder.prediction_head is not None:
                critic_aux_target = torch.cat([batch["next_obs"][valid], batch["rewards_env"][valid].unsqueeze(-1)], dim=-1)
                critic_aux_pred = self.critic_encoder.predict_auxiliary(critic_context_stats["states"][valid])
                critic_aux_loss = F.mse_loss(critic_aux_pred, critic_aux_target)

        return self._update_flat_batch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            actor_context=actor_context,
            next_actor_context=next_actor_context,
            critic_context=critic_context,
            target_critic_context=target_critic_context,
            actor_context_std=actor_context_std,
            actor_frac_alpha=actor_frac_alpha,
            actor_aux_loss=actor_aux_loss,
            critic_context_std=critic_context_std,
            critic_frac_alpha=critic_frac_alpha,
            critic_aux_loss=critic_aux_loss,
            actor_kl_loss=actor_kl_loss_tensor,
            critic_kl_loss=critic_kl_loss_tensor,
        )

    def _update_flat_batch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        actor_context: torch.Tensor | None = None,
        next_actor_context: torch.Tensor | None = None,
        critic_context: torch.Tensor | None = None,
        target_critic_context: torch.Tensor | None = None,
        actor_context_std: float | None = None,
        actor_frac_alpha: float | None = None,
        actor_aux_loss: torch.Tensor | None = None,
        critic_context_std: float | None = None,
        critic_frac_alpha: float | None = None,
        critic_aux_loss: torch.Tensor | None = None,
        actor_kl_loss: torch.Tensor | None = None,
        critic_kl_loss: torch.Tensor | None = None,
    ) -> SACUpdateStats:
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs, next_actor_context)
            target_q = torch.min(
                self.target_q1(next_obs, next_actions, target_critic_context),
                self.target_q2(next_obs, next_actions, target_critic_context),
            ) - self.alpha.detach() * next_log_prob
            target_value = rewards + (1.0 - dones) * self.gamma * target_q

        q1_loss = F.mse_loss(self.q1(obs, actions, critic_context), target_value)
        q2_loss = F.mse_loss(self.q2(obs, actions, critic_context), target_value)
        critic_loss = q1_loss + q2_loss
        if critic_aux_loss is not None:
            critic_loss = critic_loss + self.context_aux_loss_weight * critic_aux_loss
        if critic_kl_loss is not None:
            critic_loss = critic_loss + self.context_kl_weight_critic * critic_kl_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [param for group in self.critic_optimizer.param_groups for param in group["params"]],
            max_norm=1.0,
        )
        self.critic_optimizer.step()

        sampled_actions, log_prob = self.actor.sample(obs, actor_context)
        q_pi = torch.min(
            self.q1(obs, sampled_actions, None if critic_context is None else critic_context.detach()),
            self.q2(obs, sampled_actions, None if critic_context is None else critic_context.detach()),
        )
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        if actor_aux_loss is not None:
            actor_loss = actor_loss + self.context_aux_loss_weight * actor_aux_loss
        if actor_kl_loss is not None:
            actor_loss = actor_loss + self.context_kl_weight_actor * actor_kl_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [param for group in self.actor_optimizer.param_groups for param in group["params"]],
            max_norm=1.0,
        )
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        if self.use_actor_context_encoder:
            self._soft_update(self.actor_encoder, self.target_actor_encoder)
        if self.use_critic_context_encoder:
            self._soft_update(self.critic_encoder, self.target_critic_encoder)

        return SACUpdateStats(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=float(self.alpha.item()),
            actor_aux_loss=None if actor_aux_loss is None else float(actor_aux_loss.item()),
            critic_aux_loss=None if critic_aux_loss is None else float(critic_aux_loss.item()),
            actor_context_std=actor_context_std,
            actor_frac_alpha=actor_frac_alpha,
            critic_context_std=critic_context_std,
            critic_frac_alpha=critic_frac_alpha,
            actor_kl_loss=None if actor_kl_loss is None else float(actor_kl_loss.item()),
            critic_kl_loss=None if critic_kl_loss is None else float(critic_kl_loss.item()),
        )

    def state_dict(self) -> dict[str, object]:
        state = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        if self.use_actor_context_encoder:
            state.update(
                {
                    "actor_encoder": self.actor_encoder.state_dict(),
                    "target_actor_encoder": self.target_actor_encoder.state_dict(),
                }
            )
        if self.use_critic_context_encoder:
            state.update(
                {
                    "critic_encoder": self.critic_encoder.state_dict(),
                    "target_critic_encoder": self.target_critic_encoder.state_dict(),
                }
            )
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.target_q1.load_state_dict(state["target_q1"])
        self.target_q2.load_state_dict(state["target_q2"])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        if self.use_actor_context_encoder and self.actor_encoder is not None and self.target_actor_encoder is not None:
            self.actor_encoder.load_state_dict(state["actor_encoder"])
            self.target_actor_encoder.load_state_dict(state["target_actor_encoder"])
        if self.use_critic_context_encoder and self.critic_encoder is not None and self.target_critic_encoder is not None:
            self.critic_encoder.load_state_dict(state["critic_encoder"])
            self.target_critic_encoder.load_state_dict(state["target_critic_encoder"])

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * source_param.data)
