from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.models.gnn.rl_policy import GNNHierarchicalActorCritic
    from src.rl.algorithms.ppo import HierarchicalActorCritic
    from src.rl.buffer import RolloutBuffer
except ImportError:  # pragma: no cover
    from models.gnn.rl_policy import GNNHierarchicalActorCritic
    from rl.algorithms.ppo import HierarchicalActorCritic
    from rl.buffer import RolloutBuffer


@dataclass
class A2CConfig:
    learning_rate: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    hidden_layers: int = 2
    use_gnn_encoder: bool = False
    gnn_hidden_channels: int = 128
    gnn_embedding_dim: int = 256
    gnn_layers: int = 2
    gnn_dropout: float = 0.1


class A2CAgent:
    """Synchronous advantage actor-critic with hierarchical discrete actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_macros: int | None = None,
        num_directions: int = 4,
        config: A2CConfig | None = None,
        device: str | torch.device = "cpu",
        edge_index: torch.Tensor | None = None,
    ) -> None:
        self.config = config or A2CConfig()
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
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate, eps=1e-5)

    def set_graph(self, edge_index: torch.Tensor) -> None:
        """Switch the active graph topology (call before each rollout on a new design)."""
        self.edge_index = edge_index.to(self.device)

    def _edge_index(self) -> torch.Tensor:
        if self.edge_index is None:
            raise RuntimeError("No graph topology set. Call set_graph(edge_index) before using the GNN agent.")
        return self.edge_index

    @torch.no_grad()
    def act(self, observation, deterministic: bool = False) -> tuple[int, float, float]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if self.config.use_gnn_encoder:
            action, log_prob, value = self.model.act(obs, self._edge_index(), deterministic=deterministic)
        else:
            action, log_prob, value = self.model.act(obs, deterministic=deterministic)
        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def value(self, observation) -> float:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if self.config.use_gnn_encoder:
            _, _, values = self.model.forward(obs, self._edge_index())
        else:
            _, _, values = self.model.forward(obs)
        return float(values.item())

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        batch = buffer.as_tensors()
        if self.config.use_gnn_encoder:
            log_probs, entropy, values = self.model.evaluate_actions(
                batch.observations, batch.actions, self._edge_index()
            )
        else:
            log_probs, entropy, values = self.model.evaluate_actions(batch.observations, batch.actions)
        policy_loss = -(log_probs * batch.advantages).mean()
        normalized_returns = (batch.returns - batch.returns.mean()) / (batch.returns.std() + 1e-8)
        value_loss = F.mse_loss(values, normalized_returns)
        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.mean().detach().cpu()),
        }

    def save(self, path: str | Path) -> None:
        obs_dim = getattr(self.model, "obs_dim", None)
        if obs_dim is None and self.features_per_macro is not None:
            obs_dim = self.num_macros * self.features_per_macro
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config.__dict__,
                "num_macros": self.num_macros,
                "num_directions": self.num_directions,
                "obs_dim": obs_dim,
                "features_per_macro": self.features_per_macro,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "A2CAgent":
        payload = torch.load(path, map_location=device)
        config = A2CConfig(**payload["config"])
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
