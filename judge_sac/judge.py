"""History-based Judge reward model and shared fractional-memory modules."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class ResidualDynamics(nn.Module):
    def __init__(self, memory_dim: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.net = nn.Sequential(
            nn.Linear(memory_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class FractionalMemoryBackbone(nn.Module):
    """Shared fractional-memory backbone used by Judge and critic context encoders."""

    def __init__(
        self,
        *,
        input_dim: int,
        memory_dim: int,
        hidden_dim: int,
        frac_alpha: float,
        learnable_frac_alpha: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.device = torch.device(device)

        initial_alpha = float(np.clip(frac_alpha, 1e-4, 0.999))
        self.frac_alpha_logit = nn.Parameter(
            torch.logit(torch.tensor(initial_alpha, dtype=torch.float32)),
            requires_grad=learnable_frac_alpha,
        )
        self.dynamics = ResidualDynamics(memory_dim, input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(memory_dim)
        self.to(self.device)

    @property
    def frac_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.frac_alpha_logit).clamp(1e-4, 1.0)

    def forward_sequence(self, sequence_inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if sequence_inputs.dim() != 3:
            raise ValueError("sequence_inputs must be a [batch, time, dim] tensor.")
        if mask.dim() != 2:
            raise ValueError("mask must be a [batch, time] tensor.")

        batch_size, seq_len, _ = sequence_inputs.shape
        alpha = self.frac_alpha
        gamma_alpha = torch.exp(torch.lgamma(alpha + 1.0))

        ks = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        frac_weights = torch.pow(ks + 1.0, alpha) - torch.pow(ks, alpha)

        dynamics_list: list[torch.Tensor] = []
        current_memory = sequence_inputs.new_zeros(batch_size, self.memory_dim)
        state_history: list[torch.Tensor] = []

        for t in range(seq_len):
            step_mask = mask[:, t : t + 1]
            dynamics_input = torch.cat([current_memory, sequence_inputs[:, t]], dim=-1)
            dynamics_list.append(self.dynamics(dynamics_input))

            dynamics_stack = torch.stack(dynamics_list, dim=0)
            w = frac_weights[: t + 1].flip(0)
            weighted_sum = torch.einsum("t,tbm->bm", w, dynamics_stack)

            next_memory = self.layer_norm(weighted_sum / gamma_alpha)
            current_memory = step_mask * next_memory + (1.0 - step_mask) * current_memory
            state_history.append(current_memory)

        return torch.stack(state_history, dim=1)


class RewardHead(nn.Module):
    def __init__(self, memory_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return self.net(memory)


class GaussianContextEncoder(nn.Module):
    """Independent critic-side context encoder with Gaussian latent sampling."""

    def __init__(
        self,
        *,
        input_dim: int,
        memory_dim: int = 128,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        prediction_dim: int = 0,
        frac_alpha: float = 0.6,
        learnable_frac_alpha: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.prediction_dim = prediction_dim
        self.backbone = FractionalMemoryBackbone(
            input_dim=input_dim,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            frac_alpha=frac_alpha,
            learnable_frac_alpha=learnable_frac_alpha,
            device=device,
        )
        self.mean_head = nn.Linear(memory_dim, latent_dim)
        self.log_std_head = nn.Linear(memory_dim, latent_dim)
        self.prediction_head = (
            nn.Sequential(
                nn.Linear(memory_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, prediction_dim),
            )
            if prediction_dim > 0
            else None
        )
        self.to(self.device)

    def forward_sequence(
        self, sequence_inputs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False
    ) -> dict[str, torch.Tensor]:
        states = self.backbone.forward_sequence(sequence_inputs, mask)
        mean = self.mean_head(states)
        log_std = self.log_std_head(states).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        if deterministic:
            latent = mean
        else:
            latent = mean + torch.randn_like(std) * std
        expanded_mask = mask.unsqueeze(-1)
        return {
            "states": states,
            "mean": mean,
            "log_std": log_std,
            "std": std,
            "z": latent * expanded_mask,
            "frac_alpha": self.backbone.frac_alpha,
        }

    def predict_auxiliary(self, states: torch.Tensor) -> torch.Tensor:
        if self.prediction_head is None:
            raise ValueError("prediction_head is not enabled for this encoder.")
        return self.prediction_head(states)


class HistoryRewardJudge(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        memory_dim: int = 32,
        hidden_dim: int = 64,
        latent_dim: int | None = None,
        frac_alpha: float = 0.6,
        learnable_frac_alpha: bool = True,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        if latent_dim is None:
            latent_dim = memory_dim
        self.backbone = FractionalMemoryBackbone(
            input_dim=input_dim,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            frac_alpha=frac_alpha,
            learnable_frac_alpha=learnable_frac_alpha,
            device=device,
        )
        self.mean_head = nn.Linear(memory_dim, latent_dim)
        self.log_std_head = nn.Linear(memory_dim, latent_dim)
        self.reward_head = RewardHead(latent_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        self.eval()

    @property
    def frac_alpha(self) -> torch.Tensor:
        return self.backbone.frac_alpha

    def forward_sequence(self, judge_inputs: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        states = self.backbone.forward_sequence(judge_inputs, mask)
        mean = self.mean_head(states)
        log_std = self.log_std_head(states).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        z = mean + torch.randn_like(std) * std
        expanded_mask = mask.unsqueeze(-1)
        G_hat = self.reward_head(z).squeeze(-1) * mask
        G_hat_prev = torch.cat([torch.zeros_like(G_hat[:, :1]), G_hat[:, :-1]], dim=1)
        r_tilde = (G_hat - G_hat_prev) * mask
        return {
            "states": states,
            "mean": mean,
            "std": std,
            "z": z * expanded_mask,
            "r_tilde": r_tilde,
            "frac_alpha": self.backbone.frac_alpha,
        }

    def update_from_batch(
        self,
        judge_inputs: torch.Tensor,
        rewards_env: torch.Tensor,
        mask: torch.Tensor,
        loss_mask: torch.Tensor,
        kl_weight: float = 0.0,
    ) -> dict[str, float]:
        self.train()
        outputs = self.forward_sequence(judge_inputs, mask)
        effective_mask = mask * loss_mask

        r_judge_sum = (outputs["r_tilde"] * effective_mask).sum(dim=1)
        r_env_sum = (rewards_env * effective_mask).sum(dim=1)
        decomp_loss = ((r_judge_sum - r_env_sum) ** 2).mean()

        if kl_weight > 0.0:
            kl_loss = temporal_kl_loss(outputs["mean"], outputs["std"], mask, loss_mask)
            loss = decomp_loss + kl_weight * kl_loss
            kl_loss_value = float(kl_loss.item())
        else:
            loss = decomp_loss
            kl_loss_value = 0.0

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.eval()
        return {
            "judge_loss": float(loss.item()),
            "judge_decomp_loss": float(decomp_loss.item()),
            "judge_kl_loss": kl_loss_value,
            "frac_alpha": float(self.frac_alpha.item()),
        }


def temporal_kl_loss(
    mean: torch.Tensor,
    std: torch.Tensor,
    mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Temporal KL: KL(q(z_t) || prior_t) summed over valid steps.

    prior_t = N(0, I)             if t is an episode start (prev mask = 0)
    prior_t = q(z_{t-1})  (detached)  otherwise
    """
    effective_mask = mask * loss_mask
    prior_mean = torch.zeros_like(mean)
    prior_std = torch.ones_like(std)

    if mean.shape[1] > 1:
        prior_mean[:, 1:] = mean[:, :-1].detach()
        prior_std[:, 1:] = std[:, :-1].detach()
        prev_valid = mask[:, :-1].unsqueeze(-1)
        prior_mean[:, 1:] = prior_mean[:, 1:] * prev_valid
        prior_std[:, 1:] = prior_std[:, 1:] * prev_valid + (1.0 - prev_valid)

    safe_std = std.clamp(min=1e-8)
    safe_prior_std = prior_std.clamp(min=1e-8)
    log_ratio = safe_prior_std.log() - safe_std.log()
    kl_per_dim = log_ratio + (safe_std.pow(2) + (mean - prior_mean).pow(2)) / (2.0 * safe_prior_std.pow(2)) - 0.5
    kl_per_step = kl_per_dim.sum(dim=-1)

    denom = torch.clamp(effective_mask.sum(), min=1.0)
    return (kl_per_step * effective_mask).sum() / denom


def build_judge_inputs(
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    rewards_env: torch.Tensor,
    *,
    include_env_reward: bool = True,
) -> torch.Tensor:
    """Build docx-aligned Judge local features xi_t=[o_t, a_t, o_{t+1}, r_{t+1}^{env}]."""
    reward_feature = rewards_env.unsqueeze(-1)
    if not include_env_reward:
        reward_feature = torch.zeros_like(reward_feature)
    return torch.cat([obs, actions, next_obs, reward_feature], dim=-1)


def build_context_inputs(
    obs: torch.Tensor,
    prev_actions: torch.Tensor,
    prev_rewards: torch.Tensor,
) -> torch.Tensor:
    """Build pdf/html-style trajectory features u_t=[o_t, a_{t-1}, r_t] for context encoders."""
    return torch.cat([obs, prev_actions, prev_rewards.unsqueeze(-1)], dim=-1)


def fuse_reward(env_reward: float, judge_reward: float, reward_mode: str, mix_alpha: float) -> float:
    if reward_mode == "env":
        return float(env_reward)
    if reward_mode in {"judge", "replace"}:
        return float(judge_reward)
    if reward_mode == "mix":
        return float(mix_alpha * env_reward + (1.0 - mix_alpha) * judge_reward)
    raise ValueError(f"Unsupported reward mode: {reward_mode}")
