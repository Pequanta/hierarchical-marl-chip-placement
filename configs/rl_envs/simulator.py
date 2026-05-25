from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.serialization import safe_globals
from torch_geometric.data import Data

try:
    from .action_space import HierarchicalAction, HierarchicalActionSpace
    from .constraints import PlacementConstraints
except ImportError:  # pragma: no cover
    from action_space import HierarchicalAction, HierarchicalActionSpace
    from constraints import PlacementConstraints


@dataclass
class StepResult:
    graph: Any
    action: HierarchicalAction


def load_graph(graph_path: str | Path) -> tuple[Data, dict[str, Any]]:
    with safe_globals([Data]):
        payload = torch.load(graph_path, map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and "graph" in payload:
        graph, metadata = payload["graph"], dict(payload.get("metadata", {}))
    elif isinstance(payload, Data):
        graph, metadata = payload, {}
    else:
        raise ValueError(f"Unsupported graph file format: {graph_path}")

    if hasattr(graph, "x") and graph.x is not None:
        # Extract width and height (indices 0 and 1)
        w = graph.x[:, 0]
        h = graph.x[:, 1]
        area = w * h
        aspect_ratio = w / (h + 1e-6)

        num_nodes = int(graph.num_nodes)
        degrees = torch.zeros(num_nodes, dtype=torch.float32)
        if hasattr(graph, "edge_index") and graph.edge_index is not None and graph.edge_index.numel() > 0:
            src = graph.edge_index[0]
            degrees.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))

        # Estimate pin count as a function of connectivity degree
        pin_count = degrees * 3.0 + 10.0

        # Build new features: [w, h, area, pin_count, aspect_ratio, connectivity_degree]
        # Width and height remain at indices 0 & 1 for compatibility with density/overlap logic
        graph.x = torch.stack([w, h, area, pin_count, aspect_ratio, degrees], dim=1)

    return graph, metadata


class PlacementSimulator:
    """Applies manager-worker placement actions to a PyG graph."""

    def __init__(
        self,
        graph: Data,
        action_space: HierarchicalActionSpace | None = None,
        constraints: PlacementConstraints | None = None,
    ) -> None:
        if not hasattr(graph, "pos") or graph.pos is None:
            raise ValueError("graph must define graph.pos.")

        self.graph = graph
        self.initial_pos = graph.pos.detach().clone()
        self.action_codec = action_space or HierarchicalActionSpace(int(graph.num_nodes))
        self.constraints = constraints or PlacementConstraints()

    @classmethod
    def from_file(
        cls,
        graph_path: str | Path,
        action_space: HierarchicalActionSpace | None = None,
        constraints: PlacementConstraints | None = None,
    ) -> tuple["PlacementSimulator", dict[str, Any]]:
        graph, metadata = load_graph(graph_path)
        return cls(graph, action_space=action_space, constraints=constraints), metadata

    def reset(self, seed: int | None = None, randomize: bool = True) -> Data:
        if randomize:
            generator = torch.Generator(device=self.graph.pos.device)
            if seed is not None:
                generator.manual_seed(int(seed))
            self.graph.pos = torch.rand(self.initial_pos.shape, generator=generator, dtype=self.initial_pos.dtype)
        else:
            self.graph.pos = self.initial_pos.detach().clone()
        self.constraints.apply(self.graph)
        return self.graph

    def step(self, action) -> StepResult:
        decoded = self.action_codec.decode(action)
        # Treat directions coordinate center as absolute placement target
        grid_pos = torch.as_tensor(
            self.action_codec.delta(decoded.direction_index),
            dtype=self.graph.pos.dtype,
            device=self.graph.pos.device,
        )
        self.graph.pos[decoded.macro_index] = grid_pos
        self.constraints.apply(self.graph)
        return StepResult(graph=self.graph, action=decoded)
