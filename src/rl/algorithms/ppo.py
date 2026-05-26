from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from src.models.gnn.rl_policy import GNNHierarchicalActorCritic
    from src.rl.buffer import RolloutBuffer
except ImportError:  # pragma: no cover - supports running from src as cwd
    from models.gnn.rl_policy import GNNHierarchicalActorCritic
    from rl.buffer import RolloutBuffer


def mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2) -> nn.Sequential:
    modules: list[nn.Module] = []
    last_dim = input_dim
    for _ in range(layers):
        modules.extend([nn.Linear(last_dim, hidden_dim), nn.Tanh()])
        last_dim = hidden_dim
    modules.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*modules)


class HierarchicalActorCritic(nn.Module):
    """Factorizes placement actions into macro choice and movement direction/grid placement."""

    def __init__(
        self,
        obs_dim: int,
        num_macros: int,
        num_directions: int = 4,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_macros = num_macros
        self.num_directions = num_directions
        self.encoder = mlp(obs_dim, hidden_dim, hidden_dim, hidden_layers)
        self.macro_head = nn.Linear(hidden_dim, num_macros)
        if num_directions == 64:
            self.subregion_head = nn.Linear(hidden_dim, 4)
            self.grid_cell_head = nn.Linear(hidden_dim, 16)
        else:
            self.direction_head = nn.Linear(hidden_dim, num_directions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        features = self.encoder(observations)
        macro_logits = self.macro_head(features)
        if self.num_directions == 64:
            subregion_logits = self.subregion_head(features)
            grid_cell_logits = self.grid_cell_head(features)
            # Combine logits to form joint worker action space logits (4 x 16 = 64)
            joint_logits = subregion_logits.unsqueeze(-1) + grid_cell_logits.unsqueeze(-2)
            direction_logits = joint_logits.view(subregion_logits.shape[0], -1)
        else:
            direction_logits = self.direction_head(features)
        values = self.value_head(features).squeeze(-1)
        return macro_logits, direction_logits, values

    def distribution(self, observations: torch.Tensor) -> tuple[Categorical, Categorical, torch.Tensor]:
        macro_logits, direction_logits, values = self.forward(observations)
        return Categorical(logits=macro_logits), Categorical(logits=direction_logits), values

    def act(self, observations: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        macro_dist, direction_dist, values = self.distribution(observations)
        if deterministic:
            macros = macro_dist.probs.argmax(dim=-1)
            directions = direction_dist.probs.argmax(dim=-1)
        else:
            macros = macro_dist.sample()
            directions = direction_dist.sample()
        actions = macros * self.num_directions + directions
        log_probs = macro_dist.log_prob(macros) + direction_dist.log_prob(directions)
        return actions, log_probs, values

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        macro_dist, direction_dist, values = self.distribution(observations)
        macros = actions // self.num_directions
        directions = actions % self.num_directions
        log_probs = macro_dist.log_prob(macros) + direction_dist.log_prob(directions)
        entropy = macro_dist.entropy() + direction_dist.entropy()
        return log_probs, entropy, values


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    batch_size: int = 64
    hidden_dim: int = 256
    hidden_layers: int = 2
    use_gnn_encoder: bool = False
    gnn_hidden_channels: int = 128
    gnn_embedding_dim: int = 256
    gnn_layers: int = 2
    gnn_dropout: float = 0.1


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_macros: int | None = None,
        num_directions: int = 4,
        config: PPOConfig | None = None,
        device: str | torch.device = "cpu",
        edge_index: torch.Tensor | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.num_directions = num_directions
        self.num_macros = num_macros or max(1, action_dim // num_directions)
        self.edge_index = edge_index.to(self.device) if edge_index is not None else None
        if self.config.use_gnn_encoder:
            if obs_dim % self.num_macros != 0:
                raise ValueError(f"obs_dim {obs_dim} must be divisible by num_macros {self.num_macros}.")
            self.features_per_macro = obs_dim // self.num_macros
            self.model = GNNHierarchicalActorCritic(
                features_per_macro=self.features_per_macro,
                num_directions=num_directions,
                hidden_channels=self.config.gnn_hidden_channels,
                embedding_dim=self.config.gnn_embedding_dim,
                num_layers=self.config.gnn_layers,
                dropout=self.config.gnn_dropout,
            ).to(self.device)
        else:
            self.features_per_macro = None
            self.model = HierarchicalActorCritic(
                obs_dim=obs_dim,
                num_macros=self.num_macros,
                num_directions=num_directions,
                hidden_dim=self.config.hidden_dim,
                hidden_layers=self.config.hidden_layers,
            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def set_graph(self, edge_index: torch.Tensor) -> None:
        """Switch the active graph topology (call before each rollout on a new design)."""
        self.edge_index = edge_index.to(self.device)

    def _edge_index(self) -> torch.Tensor:
        if self.edge_index is None:
            raise RuntimeError("No graph topology set. Call set_graph(edge_index) before using the GNN agent.")
        return self.edge_index

    @torch.no_grad()
    def act(self, observation: Any, deterministic: bool = False) -> tuple[int, float, float]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if self.config.use_gnn_encoder:
            action, log_prob, value = self.model.act(obs, self._edge_index(), deterministic=deterministic)
        else:
            action, log_prob, value = self.model.act(obs, deterministic=deterministic)
        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def value(self, observation: Any) -> float:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if self.config.use_gnn_encoder:
            _, _, values = self.model.forward(obs, self._edge_index())
        else:
            _, _, values = self.model.forward(obs)
        return float(values.item())

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        metrics: dict[str, float] = {}
        # Compute return statistics once over the full buffer for stable value normalization.
        full_batch = buffer.as_tensors()
        returns_mean = full_batch.returns.mean()
        returns_std = full_batch.returns.std() + 1e-8
        for _ in range(self.config.update_epochs):
            for batch in buffer.minibatches(self.config.batch_size):
                if self.config.use_gnn_encoder:
                    log_probs, entropy, values = self.model.evaluate_actions(
                        batch.observations, batch.actions, self._edge_index()
                    )
                else:
                    log_probs, entropy, values = self.model.evaluate_actions(batch.observations, batch.actions)
                ratio = torch.exp(log_probs - batch.old_log_probs)
                unclipped = ratio * batch.advantages
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * batch.advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                normalized_returns = (batch.returns - returns_mean) / returns_std
                value_loss = F.mse_loss(values, normalized_returns)
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss + self.config.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                approx_kl = (batch.old_log_probs - log_probs).mean().detach()
                metrics = {
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.mean().detach().cpu()),
                    "approx_kl": float(approx_kl.cpu()),
                }
        return metrics

    def save(self, path: str | Path) -> None:
        obs_dim = getattr(self.model, "obs_dim", None)
        if obs_dim is None and self.features_per_macro is not None:
            obs_dim = self.num_macros * self.features_per_macro
        payload = {
            "state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "num_macros": self.num_macros,
            "num_directions": self.num_directions,
            "obs_dim": obs_dim,
            "features_per_macro": self.features_per_macro,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "PPOAgent":
        payload = torch.load(path, map_location=device)
        config = PPOConfig(**payload["config"])
        num_macros = payload["num_macros"]
        features_per_macro = payload.get("features_per_macro")
        obs_dim = payload.get("obs_dim") or (num_macros * features_per_macro if features_per_macro else num_macros)
        agent = cls(
            obs_dim=obs_dim,
            action_dim=num_macros * payload["num_directions"],
            num_macros=num_macros,
            num_directions=payload["num_directions"],
            config=config,
            device=device,
            # edge_index intentionally omitted — caller must call set_graph() before use
        )
        agent.model.load_state_dict(payload["state_dict"])
        return agent
