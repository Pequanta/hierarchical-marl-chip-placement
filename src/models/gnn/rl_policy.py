from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from .encoder import FlatPlacementObservationEncoder
except ImportError:  # pragma: no cover
    from encoder import FlatPlacementObservationEncoder


class GNNHierarchicalActorCritic(nn.Module):
    """Actor-critic policy that learns RL representations with the placement GNN.

    Macro logits are produced from node embeddings (one logit per node, so the action
    space scales with the design). Direction logits and values are produced from the
    pooled graph embedding.

    `edge_index` is **not** stored in the model. Every forward method requires it as an
    explicit argument, enabling the same weights to be applied to designs with different
    macro counts and connectivity patterns.

    Args:
        features_per_macro: Node feature dimension — fixed across all designs.
        num_macros: Accepted for backward compatibility but ignored.
        edge_index: Accepted for backward compatibility but ignored.
    """

    def __init__(
        self,
        features_per_macro: int,
        num_directions: int = 4,
        hidden_channels: int = 128,
        embedding_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_macros: int | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        del num_macros, edge_index  # accepted for backward compat; topology is passed to forward()
        self.features_per_macro = int(features_per_macro)
        self.num_directions = int(num_directions)
        self.representation = FlatPlacementObservationEncoder(
            features_per_macro=features_per_macro,
            hidden_channels=hidden_channels,
            out_channels=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.macro_head = nn.Linear(embedding_dim, 1)
        if num_directions == 64:
            self.subregion_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, 4)
            )
            self.grid_cell_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, 16)
            )
        else:
            self.direction_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, num_directions)
            )
        self.value_head = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, 1))

    def forward(self, observations: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rep = self.representation(observations, edge_index)
        macro_logits = self.macro_head(rep.node_embeddings).squeeze(-1)
        if self.num_directions == 64:
            subregion_logits = self.subregion_head(rep.graph_embedding)
            grid_cell_logits = self.grid_cell_head(rep.graph_embedding)
            # Combine logits to form joint worker action space logits (4 x 16 = 64)
            joint_logits = subregion_logits.unsqueeze(-1) + grid_cell_logits.unsqueeze(-2)
            direction_logits = joint_logits.view(subregion_logits.shape[0], -1)
        else:
            direction_logits = self.direction_head(rep.graph_embedding)
        values = self.value_head(rep.graph_embedding).squeeze(-1)
        return macro_logits, direction_logits, values

    def distribution(self, observations: torch.Tensor, edge_index: torch.Tensor) -> tuple[Categorical, Categorical, torch.Tensor]:
        macro_logits, direction_logits, values = self.forward(observations, edge_index)
        return Categorical(logits=macro_logits), Categorical(logits=direction_logits), values

    def act(self, observations: torch.Tensor, edge_index: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        macro_dist, direction_dist, values = self.distribution(observations, edge_index)
        if deterministic:
            macros = macro_dist.probs.argmax(dim=-1)
            directions = direction_dist.probs.argmax(dim=-1)
        else:
            macros = macro_dist.sample()
            directions = direction_dist.sample()
        actions = macros * self.num_directions + directions
        log_probs = macro_dist.log_prob(macros) + direction_dist.log_prob(directions)
        return actions, log_probs, values

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        macro_dist, direction_dist, values = self.distribution(observations, edge_index)
        macros = actions // self.num_directions
        directions = actions % self.num_directions
        log_probs = macro_dist.log_prob(macros) + direction_dist.log_prob(directions)
        entropy = macro_dist.entropy() + direction_dist.entropy()
        return log_probs, entropy, values

    def save_encoder(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.representation.encoder.state_dict(),
                "features_per_macro": self.features_per_macro,
            },
            path,
        )


def build_gnn_actor_critic_from_env(env: Any, **kwargs: Any) -> GNNHierarchicalActorCritic:
    graph = getattr(env, "graph", None)
    if graph is None or not hasattr(graph, "edge_index"):
        raise ValueError("Environment must expose graph.edge_index to build a GNN actor-critic.")

    num_macros = int(getattr(env, "num_macros", graph.num_nodes))
    obs_dim = int(env.observation_space.shape[0])
    if obs_dim % num_macros != 0:
        raise ValueError(f"Observation dim {obs_dim} is not divisible by num_macros {num_macros}.")
    return GNNHierarchicalActorCritic(
        features_per_macro=obs_dim // num_macros,
        **kwargs,
    )
