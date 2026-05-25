from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:
    from .constraints import PlacementConstraints
except ImportError:  # pragma: no cover
    from constraints import PlacementConstraints


def canvas_size(graph) -> np.ndarray:
    if hasattr(graph, "canvas_size"):
        canvas = graph.canvas_size
        if isinstance(canvas, torch.Tensor):
            canvas = canvas.detach().cpu().numpy()
        return np.asarray(canvas, dtype=np.float32)
    return np.asarray([400.0, 400.0], dtype=np.float32)


def compute_hpwl(graph, divide_bidirectional_edges: bool = True) -> float:
    if not hasattr(graph, "edge_index") or graph.edge_index is None or graph.edge_index.numel() == 0:
        return 0.0

    pos = graph.pos.detach().cpu().numpy() if isinstance(graph.pos, torch.Tensor) else np.asarray(graph.pos)
    edge_index = graph.edge_index.detach().cpu().numpy() if isinstance(graph.edge_index, torch.Tensor) else np.asarray(graph.edge_index)
    pos_abs = pos * canvas_size(graph)
    src, dst = edge_index
    hpwl = np.abs(pos_abs[src, 0] - pos_abs[dst, 0]) + np.abs(pos_abs[src, 1] - pos_abs[dst, 1])
    total = float(np.sum(hpwl))
    return total / 2.0 if divide_bidirectional_edges else total


def compute_max_bin_density(graph, bins: int = 10) -> float:
    """Peak bin density ratio (dimensionless).

    Uses centroid-based bin assignment to exactly match the congestion metric
    reported by placement_benchmark.py (max_density field).  A value of 1.0
    means one bin is 100 % filled; values > 1 indicate stacking / overlap.
    """
    canvas = canvas_size(graph)
    pos = graph.pos.detach().cpu().numpy() if isinstance(graph.pos, torch.Tensor) else np.asarray(graph.pos)
    sizes = graph.x[:, :2].detach().cpu().numpy() if isinstance(graph.x, torch.Tensor) else np.asarray(graph.x)[:, :2]

    grid = np.zeros((bins, bins), dtype=np.float32)
    for idx in range(graph.num_nodes):
        bin_x = int(np.clip(np.floor(pos[idx, 0] * bins), 0, bins - 1))
        bin_y = int(np.clip(np.floor(pos[idx, 1] * bins), 0, bins - 1))
        grid[bin_y, bin_x] += sizes[idx, 0] * sizes[idx, 1]

    bin_area = (canvas[0] / bins) * (canvas[1] / bins)
    density = grid / max(bin_area, 1e-6)
    return float(np.max(density))


@dataclass
class RewardConfig:
    hpwl_scale: float = 1.0
    improvement_scale: float = 2.0
    overlap_scale: float = 10.0
    density_scale: float = 1.0
    congestion_scale: float = 5.0
    # Penalises max-bin-density degradation relative to episode-start baseline.
    # Calibrated for the dimensionless max_density metric (typical range 2–15).
    # Break-even with improvement_scale=2.0 at a ~2× max_density increase.
    congestion_improvement_scale: float = 0.8
    # Fraction of initial max_density allowed to increase before the terminal
    # bonus is withheld and a proportional terminal penalty fires instead.
    congestion_tolerance: float = 0.5
    terminal_bonus_scale: float = 2.0


def _area_spread_grid(graph, bins: int) -> np.ndarray:
    """Distribute each macro's area across all bins it overlaps (area-weighted)."""
    canvas = canvas_size(graph)
    pos = graph.pos.detach().cpu().numpy() if isinstance(graph.pos, torch.Tensor) else np.asarray(graph.pos)
    sizes = graph.x[:, :2].detach().cpu().numpy() if isinstance(graph.x, torch.Tensor) else np.asarray(graph.x)[:, :2]

    grid = np.zeros((bins, bins), dtype=np.float32)
    for i in range(graph.num_nodes):
        cx, cy = pos[i, 0] * canvas[0], pos[i, 1] * canvas[1]
        w, h = sizes[i, 0], sizes[i, 1]
        x_min, x_max = cx - w / 2.0, cx + w / 2.0
        y_min, y_max = cy - h / 2.0, cy + h / 2.0
        for bx in range(bins):
            bin_x_min = (bx / bins) * canvas[0]
            bin_x_max = ((bx + 1) / bins) * canvas[0]
            overlap_x = max(0.0, min(x_max, bin_x_max) - max(x_min, bin_x_min))
            if overlap_x <= 0.0:
                continue
            for by in range(bins):
                bin_y_min = (by / bins) * canvas[1]
                bin_y_max = ((by + 1) / bins) * canvas[1]
                overlap_y = max(0.0, min(y_max, bin_y_max) - max(y_min, bin_y_min))
                if overlap_y <= 0.0:
                    continue
                grid[by, bx] += overlap_x * overlap_y
    return grid


def compute_bin_overflow_congestion(graph, bins: int = 10, target_density: float = 0.7) -> float:
    """Total area overflow above routing-capacity threshold, in canvas area units²."""
    canvas = canvas_size(graph)
    canvas_area = canvas[0] * canvas[1]
    bin_capacity = (canvas_area / (bins * bins)) * target_density

    grid = _area_spread_grid(graph, bins)
    overflow = np.maximum(0.0, grid - bin_capacity)
    return float(np.sum(overflow))


def compute_density_penalty(graph, bins: int = 10) -> float:
    """Area overflow above the ideal uniform bin density, in canvas area units².

    Uses area-spread per bin so large macros spanning multiple bins are
    counted correctly.  Returns 0 when macros are perfectly uniformly spread.
    """
    grid = _area_spread_grid(graph, bins)
    total_macro_area = float(np.sum(grid))
    uniform_bin_area = total_macro_area / (bins * bins)
    overflow = np.maximum(0.0, grid - uniform_bin_area)
    return float(np.sum(overflow))


class PlacementReward:
    """Multi-objective reward: minimise HPWL without sacrificing routability.

    Episode-relative congestion tracking uses compute_max_bin_density, which
    is the same metric reported as 'congestion' in placement_benchmark.py.
    This ensures training directly optimises for what the benchmark measures.
    """

    def __init__(self, config: RewardConfig | None = None, constraints: PlacementConstraints | None = None) -> None:
        self.config = config or RewardConfig()
        self.constraints = constraints or PlacementConstraints()
        self.initial_hpwl: float | None = None
        self.initial_max_density: float | None = None

    def reset(self, graph) -> None:
        self.initial_hpwl = compute_hpwl(graph)
        self.initial_max_density = compute_max_bin_density(graph)

    def __call__(
        self,
        graph,
        terminated: bool = False,
    ) -> tuple[float, dict[str, float]]:
        hpwl = compute_hpwl(graph)
        overlap_penalty = self.constraints.penalty(graph)
        density_val = compute_density_penalty(graph)
        congestion_val = compute_bin_overflow_congestion(graph, bins=10, target_density=0.7)
        max_density = compute_max_bin_density(graph)

        canvas = canvas_size(graph)
        canvas_area = float(canvas[0] * canvas[1])

        # Design-aware HPWL normalisation: scale by max-possible wirelength.
        canvas_diagonal = float(np.sqrt(canvas[0] ** 2 + canvas[1] ** 2))
        num_edges = max(
            1,
            graph.edge_index.shape[1] // 2
            if hasattr(graph, "edge_index") and graph.edge_index is not None and graph.edge_index.numel() > 0
            else 1,
        )
        hpwl_norm = hpwl / max(canvas_diagonal * num_edges, 1e-6)

        overlap_norm = overlap_penalty / max(canvas_area, 1e-6)
        density_norm = density_val / max(canvas_area, 1e-6)
        congestion_norm = congestion_val / max(canvas_area, 1e-6)

        reward = (
            -self.config.hpwl_scale * hpwl_norm
            - self.config.overlap_scale * overlap_norm
            - self.config.density_scale * density_norm
            - self.config.congestion_scale * congestion_norm
        )

        # Episode-relative HPWL improvement (anchored at reset, prevents
        # step-level oscillation reward gaming).
        improvement = 0.0
        if self.initial_hpwl is not None and self.initial_hpwl > 1e-6:
            improvement = (self.initial_hpwl - hpwl) / self.initial_hpwl
            reward += self.config.improvement_scale * improvement

        # Episode-relative max_density degradation penalty.
        # Uses the same metric as placement_benchmark.py so training and
        # evaluation are aligned.  A minimum reference of 0.1 prevents
        # unbounded ratios when initial placement is perfectly spread.
        congestion_delta = 0.0
        if self.initial_max_density is not None:
            ref = max(self.initial_max_density, 0.1)
            congestion_delta = (max_density - self.initial_max_density) / ref
            reward -= self.config.congestion_improvement_scale * max(0.0, congestion_delta)

        # Terminal bonus: awarded only when max_density did not degrade beyond
        # the allowed tolerance; replaced by a proportional penalty otherwise.
        if terminated:
            excess = congestion_delta - self.config.congestion_tolerance
            if excess <= 0.0:
                reward += self.config.terminal_bonus_scale * max(improvement, 0.0)
            else:
                reward -= self.config.terminal_bonus_scale * excess

        return float(reward), {
            "hpwl": float(hpwl),
            "max_density": float(max_density),
            "overlap_penalty": float(overlap_penalty),
            "density_penalty": float(density_val),
            "congestion_penalty": float(congestion_val),
            "improvement": float(improvement),
            "congestion_delta": float(congestion_delta),
        }
