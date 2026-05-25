from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

try:
    from src.rl.algorithms.a2c import A2CAgent, A2CConfig
    from src.rl.algorithms.dqn import DQNAgent, DQNConfig
    from src.rl.algorithms.ppo import PPOAgent, PPOConfig
    from src.rl.buffer import ReplayBuffer, RolloutBuffer
    from src.rl.evaluator import EvaluationResult, evaluate_policy
except ImportError:  # pragma: no cover
    from rl.algorithms.a2c import A2CAgent, A2CConfig
    from rl.algorithms.dqn import DQNAgent, DQNConfig
    from rl.algorithms.ppo import PPOAgent, PPOConfig
    from rl.buffer import ReplayBuffer, RolloutBuffer
    from rl.evaluator import EvaluationResult, evaluate_policy

try:
    from configs.rl_envs import MacroPlacementEnv
except ImportError:  # pragma: no cover - keeps legacy package layouts usable
    try:
        from src.models.macro_placement_env import MacroPlacementEnv
    except ImportError:
        from models.macro_placement_env import MacroPlacementEnv


AlgorithmName = Literal["ppo", "a2c", "dqn"]


@dataclass
class TrainerConfig:
    graph_path: str
    algorithm: AlgorithmName = "ppo"
    total_timesteps: int = 100_000
    rollout_steps: int = 2_048
    eval_frequency: int = 10_000
    eval_episodes: int = 3
    seed: int | None = 42
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    save_best_graph: bool = True
    num_directions: int = 4
    env_config: dict[str, Any] = field(default_factory=dict)
    eval_env_config: dict[str, Any] = field(default_factory=dict)
    algorithm_config: dict[str, Any] = field(default_factory=dict)
    # Additional designs for multi-design GNN training. Each rollout samples one design
    # uniformly at random. Empty list means single-design training (uses graph_path only).
    graph_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(
        cls,
        rl_config_path: str | Path = "configs/rl.yaml",
        env_config_path: str | Path = "configs/env.yaml",
        training_config_path: str | Path = "configs/training.yaml",
    ) -> "TrainerConfig":
        """Build trainer settings from the project YAML configuration files."""

        rl_config_path = _default_config_path(rl_config_path)
        project_root = rl_config_path.resolve().parents[1]
        rl_config = _load_yaml(rl_config_path)
        env_config = _load_yaml(_config_path(env_config_path, project_root))
        training_config = _load_yaml(_config_path(training_config_path, project_root))
        return cls.from_dicts(rl_config, env_config, training_config, project_root=project_root)

    @classmethod
    def from_dicts(
        cls,
        rl_config: dict[str, Any] | None = None,
        env_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        project_root: str | Path | None = None,
    ) -> "TrainerConfig":
        rl_config = rl_config or {}
        env_config = env_config or {}
        training_config = training_config or {}
        project_root = Path(project_root).resolve() if project_root is not None else None

        trainer = dict(rl_config.get("trainer", {}))
        schedule = training_config.get("schedule", {})
        experiment = training_config.get("experiment", {})
        runtime = training_config.get("runtime", {})
        dataset = env_config.get("dataset", {})
        episode = env_config.get("episode", {})
        action_space = env_config.get("action_space", {})
        reward = env_config.get("reward", {})
        penalties = reward.get("penalties", {})
        observation = env_config.get("observation", {})

        algorithm = trainer.get("algorithm", rl_config.get("framework", {}).get("default_algorithm", "ppo"))
        algorithms = rl_config.get("algorithms", {})
        algorithm_config = dict(algorithms.get(algorithm, rl_config.get("active_algorithm_config", {})))

        graph_path = trainer.get("graph_path") or dataset.get("graph_path")
        if not graph_path:
            raise ValueError("TrainerConfig requires a graph path in rl.trainer.graph_path or env.dataset.graph_path.")

        env_kwargs = {
            "max_steps": episode.get("max_steps", 200),
            "movement_step": action_space.get("movement_step", 0.05),
            "hpwl_scale": reward.get("hpwl_scale", 0.001),
            "improvement_scale": float(reward.get("improvement_scale", 2.0)),
            "randomize_initial_positions": episode.get("randomize_initial_positions", True),
            "include_step_fraction": "step_fraction" in observation.get("fields", []),
            "overlap_weight": _penalty_weight(penalties, "overlap"),
            "density_weight": _penalty_weight(penalties, "density"),
            "congestion_weight": _penalty_weight(penalties, "congestion"),
            "congestion_improvement_scale": float(reward.get("congestion_improvement_scale", 0.8)),
            "congestion_tolerance": float(reward.get("congestion_tolerance", 0.5)),
            "num_directions": int(trainer.get("num_directions", action_space.get("num_directions", 64))),
        }

        extra_paths = [
            _resolve_path(p, project_root) for p in trainer.get("graph_paths", dataset.get("graph_paths", []))
        ]

        return cls(
            graph_path=_resolve_path(graph_path, project_root),
            algorithm=algorithm,
            total_timesteps=int(trainer.get("total_timesteps", schedule.get("total_timesteps", 100_000))),
            rollout_steps=int(trainer.get("rollout_steps", schedule.get("rollout_steps", 2_048))),
            eval_frequency=int(trainer.get("eval_frequency", schedule.get("eval_frequency", 10_000))),
            eval_episodes=int(trainer.get("eval_episodes", schedule.get("eval_episodes", 3))),
            seed=trainer.get("seed", experiment.get("seed", episode.get("seed", 42))),
            device=trainer.get("device", runtime.get("device", "cpu")),
            checkpoint_dir=_resolve_path(
                trainer.get("checkpoint_dir", experiment.get("checkpoint_dir", "checkpoints")),
                project_root,
            ),
            save_best_graph=bool(trainer.get("save_best_graph", True)),
            num_directions=int(trainer.get("num_directions", action_space.get("num_directions", 64))),
            env_config=env_kwargs,
            eval_env_config=env_kwargs.copy(),
            algorithm_config=algorithm_config,
            graph_paths=extra_paths,
        )


class HierarchicalRLTrainer:
    """End-to-end trainer for macro-placement agents.

    When `config.graph_paths` is non-empty, each on-policy rollout samples one design
    uniformly at random from the pool. This exposes the GNN encoder to diverse circuit
    topologies during training, producing transferable placement representations.
    """

    def __init__(self, config: TrainerConfig, env: Any | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.env = env or self._make_env(config.env_config, config.graph_path)
        self.eval_env = self._make_env(config.eval_env_config or config.env_config, config.graph_path)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            self.env.reset(seed=config.seed)
            self.eval_env.reset(seed=config.seed + 1)

        self.obs_shape = tuple(self.env.observation_space.shape)
        self.obs_dim = int(np.prod(self.obs_shape))
        self.action_dim = int(self.env.action_space.n)
        self.num_macros = getattr(self.env, "num_macros", max(1, self.action_dim // config.num_directions))
        self.agent = self._build_agent()
        self._sync_agent_graph(self.env)  # prime edge_index for the primary design
        self.best_eval_reward = -float("inf")
        self.history: list[dict[str, Any]] = []

        # Build the pool of training environments for multi-design training.
        # Always includes the primary env; additional paths create their own env instances.
        if config.graph_paths:
            self._design_envs: list[Any] = [self.env] + [
                self._make_env(config.env_config, path) for path in config.graph_paths
            ]
        else:
            self._design_envs = [self.env]

    def _make_env(self, env_config: dict[str, Any], graph_path: str | None = None) -> Any:
        path = graph_path or self.config.graph_path
        try:
            return MacroPlacementEnv(path, **env_config)
        except TypeError:
            legacy_keys = {"max_steps"}
            legacy_config = {key: value for key, value in env_config.items() if key in legacy_keys}
            return MacroPlacementEnv(path, **legacy_config)

    def _build_agent(self) -> Any:
        cfg = self.config.algorithm_config
        if self.config.algorithm == "ppo":
            return PPOAgent(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                num_macros=self.num_macros,
                num_directions=self.config.num_directions,
                config=PPOConfig(**cfg),
                device=self.device,
                edge_index=getattr(self.env.graph, "edge_index", None),
            )
        if self.config.algorithm == "a2c":
            return A2CAgent(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                num_macros=self.num_macros,
                num_directions=self.config.num_directions,
                config=A2CConfig(**cfg),
                device=self.device,
                edge_index=getattr(self.env.graph, "edge_index", None),
            )
        if self.config.algorithm == "dqn":
            return DQNAgent(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                config=DQNConfig(**cfg),
                device=self.device,
            )
        raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

    def train(self) -> list[dict[str, Any]]:
        print(f"Starting training with algorithm: {self.config.algorithm.upper()} on device: {self.device}")
        if self.config.algorithm == "dqn":
            return self._train_dqn()
        return self._train_on_policy()

    def _train_on_policy(self) -> list[dict[str, Any]]:
        episode_reward = 0.0
        episode_length = 0
        completed_episodes = 0
        global_step = 0

        # Pick first design and initialise.
        active_env = self._sample_design_env()
        observation, _ = active_env.reset(seed=self.config.seed)

        while global_step < self.config.total_timesteps:
            # Sample a design for this rollout. For single-design training this is a no-op.
            active_env = self._sample_design_env()
            self._sync_agent_graph(active_env)
            obs_shape = tuple(active_env.observation_space.shape)
            observation, _ = active_env.reset()
            episode_reward = 0.0
            episode_length = 0

            steps = min(self.config.rollout_steps, self.config.total_timesteps - global_step)
            buffer = RolloutBuffer(
                capacity=steps,
                observation_shape=obs_shape,
                gamma=self.agent.config.gamma,
                gae_lambda=self.agent.config.gae_lambda,
                device=self.device,
            )
            last_done = False

            for _ in range(steps):
                action, log_prob, value = self.agent.act(observation)
                next_observation, reward, terminated, truncated, _ = active_env.step(action)
                done = bool(terminated or truncated)
                buffer.add(observation, action, reward, done, value, log_prob)

                episode_reward += float(reward)
                episode_length += 1
                global_step += 1
                last_done = done
                observation = next_observation

                if done:
                    self.history.append(
                        {
                            "step": global_step,
                            "episode": completed_episodes,
                            "episode_reward": episode_reward,
                            "episode_length": episode_length,
                        }
                    )
                    completed_episodes += 1
                    observation, _ = active_env.reset()
                    episode_reward = 0.0
                    episode_length = 0

            last_value = 0.0 if last_done else self.agent.value(observation)
            buffer.compute_returns_and_advantages(last_value=last_value, last_done=last_done)
            train_metrics = self.agent.update(buffer)
            print(f"Step: {global_step}, Train Metrics: {train_metrics}")
            self._maybe_evaluate(global_step, train_metrics)

        return self.history

    def _sample_design_env(self) -> Any:
        """Return a random env from the design pool (uniform sampling)."""
        return random.choice(self._design_envs)

    def _sync_agent_graph(self, env: Any) -> None:
        """Push the current env's graph topology to the agent (GNN path only)."""
        set_graph = getattr(self.agent, "set_graph", None)
        if callable(set_graph) and hasattr(env, "graph") and hasattr(env.graph, "edge_index"):
            set_graph(env.graph.edge_index.to(self.device))

    def _train_dqn(self) -> list[dict[str, Any]]:
        cfg: DQNConfig = self.agent.config
        replay = ReplayBuffer(cfg.buffer_size, self.obs_shape, device=self.device)
        observation, _ = self.env.reset(seed=self.config.seed)
        episode_reward = 0.0
        episode_length = 0
        completed_episodes = 0

        for step in range(1, self.config.total_timesteps + 1):
            action, _, _ = self.agent.act(observation)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)
            replay.add(observation, action, reward, next_observation, done)

            train_metrics = {}
            if step % cfg.train_frequency == 0:
                train_metrics = self.agent.update(replay)

            episode_reward += float(reward)
            episode_length += 1
            observation = next_observation

            if done:
                self.history.append(
                    {
                        "step": step,
                        "episode": completed_episodes,
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                    }
                )
                completed_episodes += 1
                observation, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            print(f"Step: {step}, Train Metrics: {train_metrics}")
            self._maybe_evaluate(step, train_metrics)

        return self.history

    def _maybe_evaluate(self, step: int, train_metrics: dict[str, float]) -> None:
        if self.config.eval_frequency <= 0 or step % self.config.eval_frequency != 0:
            return

        result = evaluate_policy(self.agent, self.eval_env, episodes=self.config.eval_episodes)
        record = {"step": step, "eval": result.to_dict(), "train": train_metrics}
        self.history.append(record)
        if result.mean_reward > self.best_eval_reward:
            self.best_eval_reward = result.mean_reward
            self._save_checkpoint(step, result)

    @property
    def design_name(self) -> str:
        return Path(self.config.graph_path).stem.replace("_graph", "")

    def _save_checkpoint(self, step: int, result: EvaluationResult) -> None:
        model_path = self.checkpoint_dir / f"best_{self.config.algorithm}_{self.design_name}.pt"
        save = getattr(self.agent, "save", None)
        if callable(save):
            save(model_path)

        if self.config.save_best_graph and hasattr(self.eval_env, "graph"):
            graph_path = self.checkpoint_dir / f"best_placement_{self.design_name}_step_{step}.pt"
            torch.save(
                {
                    "graph": self.eval_env.graph,
                    "metrics": result.to_dict(),
                    "algorithm": self.config.algorithm,
                    "step": step,
                },
                graph_path,
            )


def train(config: TrainerConfig) -> tuple[Any, list[dict[str, Any]]]:
    trainer = HierarchicalRLTrainer(config)
    history = trainer.train()
    return trainer.agent, history


def train_from_yaml(
    rl_config_path: str | Path = "configs/rl.yaml",
    env_config_path: str | Path = "configs/env.yaml",
    training_config_path: str | Path = "configs/training.yaml",
) -> tuple[Any, list[dict[str, Any]]]:
    config = TrainerConfig.from_yaml(rl_config_path, env_config_path, training_config_path)
    return train(config)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(path: str | Path, root: Path | None) -> str:
    path = Path(path)
    if root is not None and not path.is_absolute():
        path = root / path
    return str(path)


def _config_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return root / path


def _default_config_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    repo_path = Path(__file__).resolve().parents[2] / path
    return repo_path if repo_path.exists() else path


def _penalty_weight(penalties: dict[str, Any], name: str) -> float:
    config = penalties.get(name, {})
    if not config.get("enabled", False):
        return 0.0
    return float(config.get("weight", 0.0))
