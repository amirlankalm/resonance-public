"""Active Experimentation Benchmark for Resonance (Circuit Spectroscopy & Polarization).

Generates reproducible development (100 worlds) and locked test splits (200 worlds),
and benchmarks agents and numerical baselines across budgets and conditions with 95% bootstrap CIs.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.envs.polarization import PolarizationEnv
from resonance.runners.runner import EpisodeRunner
from resonance.runners.manifest import RunArtifactLogger
from resonance.baselines.numerical import (
    run_prior_only_baseline,
    run_random_design_baseline,
    run_fixed_sweep_baseline,
    run_adaptive_design_baseline,
)


def compute_bootstrap_ci(data: List[float], n_boot: int = 1000, ci: float = 0.95) -> Dict[str, float]:
    """Compute mean and empirical 95% bootstrap confidence interval."""
    arr = np.array([x for x in data if x is not None and np.isfinite(x)])
    if len(arr) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return {"mean": mean_val, "ci_low": mean_val, "ci_high": mean_val}
    rng = np.random.default_rng(42)
    boot_indices = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = np.mean(arr[boot_indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot_means, alpha * 100))
    ci_high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return {"mean": mean_val, "ci_low": ci_low, "ci_high": ci_high}


class ActivePhysicsBenchmark:
    """Standard evaluation suite on 100 dev worlds or 200 test worlds."""

    def __init__(
        self,
        env_class=CircuitSpectroscopyEnv,
        num_worlds: int = 100,
        split_name: str = "dev",
        base_seed: int = 10000,
    ):
        self.env_class = env_class
        self.num_worlds = num_worlds
        self.split_name = split_name
        self.base_seed = base_seed

        # Deterministic world seeds
        self.world_seeds = [self.base_seed + i for i in range(self.num_worlds)]

    def evaluate_agent(
        self,
        agent,
        output_dir: Optional[str] = None,
        experiment_name: str = "benchmark_run",
    ) -> Dict[str, Any]:
        """Run agent across all benchmark worlds and aggregate metrics with bootstrap CIs."""
        env = self.env_class()

        logger = None
        if output_dir:
            logger = RunArtifactLogger(
                output_dir=output_dir,
                experiment_name=experiment_name,
                model_name=getattr(agent, "model_name_or_path", type(agent).__name__),
                decoding_params={"temperature": getattr(agent, "temperature", 0.0)},
                prompt_text="Active physics benchmark evaluation",
                task_name=env.get_public_observation()["task_name"],
            )

        runner = EpisodeRunner(env, agent, artifact_logger=logger)

        episodes = []
        for idx, seed in enumerate(self.world_seeds):
            world_id = f"{self.split_name}_world_{idx}"
            rec = runner.run_episode(world_seed=seed, world_id=world_id)
            episodes.append(rec)

        if logger:
            logger.write_artifacts()

        # Compute aggregate metrics with 95% bootstrap CIs
        rewards = [ep["reward"] for ep in episodes]
        nrmses = [ep["nrmse"] for ep in episodes if ep.get("nrmse") is not None]
        calls = [ep["calls_made"] for ep in episodes]
        briers = [ep["brier_score"] for ep in episodes if ep.get("brier_score") is not None]
        type_accs = [1.0 if ep.get("is_correct_type") else 0.0 for ep in episodes if ep.get("is_correct_type") is not None]

        return {
            "benchmark_split": self.split_name,
            "total_worlds": self.num_worlds,
            "reward": compute_bootstrap_ci(rewards),
            "nrmse": compute_bootstrap_ci(nrmses),
            "brier_score": compute_bootstrap_ci(briers),
            "type_classification_accuracy": compute_bootstrap_ci(type_accs),
            "mean_calls": float(np.mean(calls)),
            "episodes": episodes,
        }

    def evaluate_baselines(self) -> Dict[str, Any]:
        """Evaluate numerical ladder baselines across the exact same benchmark seeds."""
        baselines = {}

        # 1. Prior-only
        prior_rews, prior_nrmses, prior_briers, prior_accs = [], [], [], []
        for s in self.world_seeds:
            env = self.env_class()
            info = run_prior_only_baseline(env, seed=s)
            prior_rews.append(info["reward"])
            prior_nrmses.append(info["nrmse"])
            prior_briers.append(info["brier_score"])
            prior_accs.append(1.0 if info["is_correct_type"] else 0.0)

        baselines["prior_only"] = {
            "reward": compute_bootstrap_ci(prior_rews),
            "nrmse": compute_bootstrap_ci(prior_nrmses),
            "brier_score": compute_bootstrap_ci(prior_briers),
            "type_classification_accuracy": compute_bootstrap_ci(prior_accs),
            "calls": 0,
        }

        # 2. Random Design (3 calls)
        rnd_rews, rnd_nrmses, rnd_briers, rnd_accs = [], [], [], []
        for s in self.world_seeds:
            env = self.env_class()
            info = run_random_design_baseline(env, num_measurements=3, seed=s)
            rnd_rews.append(info["reward"])
            rnd_nrmses.append(info["nrmse"])
            rnd_briers.append(info["brier_score"])
            rnd_accs.append(1.0 if info["is_correct_type"] else 0.0)

        baselines["random_3calls"] = {
            "reward": compute_bootstrap_ci(rnd_rews),
            "nrmse": compute_bootstrap_ci(rnd_nrmses),
            "brier_score": compute_bootstrap_ci(rnd_briers),
            "type_classification_accuracy": compute_bootstrap_ci(rnd_accs),
            "calls": 3,
        }

        # 3. Fixed Log Sweep (3 calls)
        fix_rews, fix_nrmses, fix_briers, fix_accs = [], [], [], []
        for s in self.world_seeds:
            env = self.env_class()
            info = run_fixed_sweep_baseline(env, num_measurements=3, seed=s)
            fix_rews.append(info["reward"])
            fix_nrmses.append(info["nrmse"])
            fix_briers.append(info["brier_score"])
            fix_accs.append(1.0 if info["is_correct_type"] else 0.0)

        baselines["fixed_3calls"] = {
            "reward": compute_bootstrap_ci(fix_rews),
            "nrmse": compute_bootstrap_ci(fix_nrmses),
            "brier_score": compute_bootstrap_ci(fix_briers),
            "type_classification_accuracy": compute_bootstrap_ci(fix_accs),
            "calls": 3,
        }

        # 4. Adaptive EIG (3 calls)
        adp_rews, adp_nrmses, adp_briers, adp_accs = [], [], [], []
        for s in self.world_seeds:
            env = self.env_class()
            info = run_adaptive_design_baseline(env, max_measurements=3, seed=s)
            adp_rews.append(info["reward"])
            adp_nrmses.append(info["nrmse"])
            adp_briers.append(info["brier_score"])
            adp_accs.append(1.0 if info["is_correct_type"] else 0.0)

        baselines["adaptive_eig_3calls"] = {
            "reward": compute_bootstrap_ci(adp_rews),
            "nrmse": compute_bootstrap_ci(adp_nrmses),
            "brier_score": compute_bootstrap_ci(adp_briers),
            "type_classification_accuracy": compute_bootstrap_ci(adp_accs),
            "calls": 3,
        }

        return baselines
