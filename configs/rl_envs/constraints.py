from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class BoundaryConstraint:
    low: float = 0.0
    high: float = 1.0
    clamp: bool = True

    def apply(self, positions: torch.Tensor) -> torch.Tensor:
        if self.clamp:
            return torch.clamp(positions, self.low, self.high)
        if torch.any((positions < self.low) | (positions > self.high)):
            raise ValueError("placement violates boundary constraints.")
        return positions

    def violation(self, positions: torch.Tensor) -> float:
        below = torch.clamp(self.low - positions, min=0.0)
        above = torch.clamp(positions - self.high, min=0.0)
        return float((below + above).sum().detach().cpu())


@dataclass
class PlacementConstraints:
    boundary: BoundaryConstraint = field(default_factory=BoundaryConstraint)
    overlap_weight: float = 0.0
    density_weight: float = 0.0

    def apply(self, graph) -> None:
        graph.pos = self.boundary.apply(graph.pos)

    def penalty(self, graph) -> float:
        penalty = self.boundary.violation(graph.pos)
        if self.overlap_weight > 0.0:
            penalty += self.overlap_weight * approximate_overlap(graph)
        if self.density_weight > 0.0:
            penalty += self.density_weight * density_penalty(graph)
        return float(penalty)


def _canvas_wh(graph) -> tuple[float, float]:
    if hasattr(graph, "canvas_size"):
        canvas = graph.canvas_size
        if isinstance(canvas, torch.Tensor):
            canvas = canvas.detach().cpu().numpy()
        canvas = np.asarray(canvas, dtype=np.float32)
        return float(canvas[0]), float(canvas[1])
    return 400.0, 400.0


def approximate_macro_sizes(graph) -> np.ndarray:
    if hasattr(graph, "macro_size"):
        sizes = graph.macro_size
    elif hasattr(graph, "size"):
        sizes = graph.size
    elif hasattr(graph, "x") and graph.x is not None and graph.x.shape[1] >= 2:
        sizes = graph.x[:, :2]
    else:
        # Default: 2% of canvas
        canvas_w, canvas_h = _canvas_wh(graph)
        return np.full((int(graph.num_nodes), 2), fill_value=[canvas_w * 0.02, canvas_h * 0.02], dtype=np.float32)

    if isinstance(sizes, torch.Tensor):
        sizes = sizes.detach().cpu().numpy()
    sizes = np.asarray(sizes, dtype=np.float32)
    if sizes.ndim == 1:
        sizes = np.repeat(sizes[:, None], 2, axis=1)
    # Clip only at the lower bound — sizes are in canvas units so the upper
    # bound of 1.0 from the old code was incorrect (clipping a 40-unit macro
    # in a 400-unit canvas to 1.0 broke the overlap calculation entirely).
    return np.clip(sizes[:, :2], 1e-6, None)


def approximate_overlap(graph) -> float:
    """Pairwise overlap area in canvas units².

    Positions (graph.pos) are normalised to [0, 1]; sizes (graph.x[:, :2])
    are in canvas units.  We convert positions to canvas units before the
    comparison so both quantities share the same coordinate space.

    The returned value is in canvas units², consistent with the canvas_area
    denominator used for normalisation in PlacementReward.
    """
    if not hasattr(graph, "pos") or graph.pos is None:
        return 0.0

    canvas_w, canvas_h = _canvas_wh(graph)
    pos = graph.pos.detach().cpu().numpy() if isinstance(graph.pos, torch.Tensor) else np.asarray(graph.pos)
    # Convert normalised [0, 1] positions to canvas units.
    pos_abs = pos * np.array([canvas_w, canvas_h], dtype=np.float32)
    sizes = approximate_macro_sizes(graph)  # canvas units

    total = 0.0
    for i in range(pos_abs.shape[0]):
        for j in range(i + 1, pos_abs.shape[0]):
            dx = max(0.0, (sizes[i, 0] + sizes[j, 0]) * 0.5 - abs(pos_abs[i, 0] - pos_abs[j, 0]))
            dy = max(0.0, (sizes[i, 1] + sizes[j, 1]) * 0.5 - abs(pos_abs[i, 1] - pos_abs[j, 1]))
            total += dx * dy
    return float(total)


def density_penalty(graph, bins: int = 8) -> float:
    if not hasattr(graph, "pos") or graph.pos is None:
        return 0.0

    pos = graph.pos.detach().cpu().numpy() if isinstance(graph.pos, torch.Tensor) else np.asarray(graph.pos)
    counts, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=bins, range=[[0.0, 1.0], [0.0, 1.0]])
    target = pos.shape[0] / float(bins * bins)
    return float(np.maximum(counts - target, 0.0).sum())
