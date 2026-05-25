from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces

try:
    from .reward import canvas_size
except ImportError:  # pragma: no cover
    from reward import canvas_size


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class PlacementState:
    positions: np.ndarray
    node_features: np.ndarray
    step: int
    max_steps: int

    @property
    def flattened(self) -> np.ndarray:
        return np.concatenate([self.positions, self.node_features], axis=1).astype(np.float32).reshape(-1)


class PlacementStateEncoder:
    """Builds flattened observations from normalized macro positions and features."""

    def __init__(self, graph, include_step_fraction: bool = False) -> None:
        self.include_step_fraction = include_step_fraction
        self.num_macros = int(graph.num_nodes)
        # Static node features count is graph.x.shape[1] (which is now 6)
        self.node_feature_dim = int(graph.x.shape[1]) if hasattr(graph, "x") and graph.x is not None else 6
        # We append 4 features for subregion densities and 4 features for region connectivity
        self.features_per_macro = self.node_feature_dim + 2 + 4 + 4 + int(include_step_fraction)
        self.obs_dim = self.num_macros * self.features_per_macro

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

    def encode(self, graph, step: int = 0, max_steps: int = 1) -> np.ndarray:
        state = self.state(graph, step=step, max_steps=max_steps)
        positions = state.positions
        static_features = state.node_features

        # 1. Calculate dynamic local densities of the 4 subregions (2x2)
        sizes = graph.x[:, :2].detach().cpu().numpy() if isinstance(graph.x, torch.Tensor) else np.asarray(graph.x)[:, :2]
        subregion_areas = np.zeros(4, dtype=np.float32)
        for i in range(self.num_macros):
            rx = int(np.clip(positions[i, 0] * 2, 0, 1))
            ry = int(np.clip(positions[i, 1] * 2, 0, 1))
            subregion_idx = ry * 2 + rx
            subregion_areas[subregion_idx] += sizes[i, 0] * sizes[i, 1]

        canvas = canvas_size(graph)
        region_area = (canvas[0] * canvas[1]) / 4.0
        subregion_densities = np.clip(subregion_areas / region_area, 0.0, 1.0)
        tiled_densities = np.tile(subregion_densities, (self.num_macros, 1))

        # 2. Calculate dynamic net connectivity of each macro to subregions
        connectivity_to_subregions = np.zeros((self.num_macros, 4), dtype=np.float32)
        if hasattr(graph, "edge_index") and graph.edge_index is not None and graph.edge_index.numel() > 0:
            edge_index = graph.edge_index.detach().cpu().numpy() if isinstance(graph.edge_index, torch.Tensor) else np.asarray(graph.edge_index)
            src, dst = edge_index
            for s, d in zip(src, dst):
                rx = int(np.clip(positions[d, 0] * 2, 0, 1))
                ry = int(np.clip(positions[d, 1] * 2, 0, 1))
                subregion_idx = ry * 2 + rx
                connectivity_to_subregions[s, subregion_idx] += 1.0

        # Normalize connectivity
        sums = connectivity_to_subregions.sum(axis=1, keepdims=True)
        connectivity_to_subregions = np.divide(
            connectivity_to_subregions,
            sums,
            out=np.zeros_like(connectivity_to_subregions),
            where=sums > 0
        )

        # Concatenate: [positions (2), static_features (6), tiled_densities (4), connectivity_to_subregions (4)]
        combined = np.concatenate([
            positions,
            static_features,
            tiled_densities,
            connectivity_to_subregions
        ], axis=1)

        obs = combined.reshape(-1)
        if self.include_step_fraction:
            step_fraction = np.full((self.num_macros, 1), min(1.0, step / max(1, max_steps)), dtype=np.float32)
            obs = np.concatenate([combined, step_fraction], axis=1).reshape(-1)

        return np.clip(obs.astype(np.float32), 0.0, 1.0)

    def state(self, graph, step: int = 0, max_steps: int = 1) -> PlacementState:
        if not hasattr(graph, "pos") or graph.pos is None:
            raise ValueError("graph must define normalized macro positions in graph.pos.")

        positions = _as_numpy(graph.pos).astype(np.float32)
        node_features = (
            _as_numpy(graph.x).astype(np.float32)
            if hasattr(graph, "x") and graph.x is not None
            else np.empty((positions.shape[0], 0), dtype=np.float32)
        )
        return PlacementState(
            positions=np.clip(positions, 0.0, 1.0),
            node_features=np.clip(node_features, 0.0, 1.0),
            step=int(step),
            max_steps=int(max_steps),
        )
