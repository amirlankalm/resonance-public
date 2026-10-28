"""Prime Intellect Verifiers adapter for Resonance.

Provides a dataset-plus-rubric wrapper compatible with the Prime Intellect
verifiers specification.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np

from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.splits import get_dev_seeds, get_train_seeds, derive_test_seeds


class ResonanceVerifiersAdapter:
    """Prime Intellect Verifiers compatible environment adapter."""

    def __init__(
        self,
        split: str = "dev",
        num_problems: int = 50,
        salt: Optional[str] = None,
        **env_kwargs: Any,
    ):
        self.split = split
        self.env_kwargs = env_kwargs

        if split == "dev":
            self.seeds = get_dev_seeds()[:num_problems]
        elif split == "train":
            self.seeds = get_train_seeds(count=num_problems)
        elif split == "test":
            if not salt:
                raise ValueError("Evaluating on test split requires a verified secret salt.")
            self.seeds = derive_test_seeds(salt, count=num_problems)
        else:
            raise ValueError(f"Unknown split: {split}. Choose 'dev', 'train', or 'test'.")

    def get_dataset(self) -> List[Dict[str, Any]]:
        """Return dataset problems containing initial prompt and metadata."""
        dataset: List[Dict[str, Any]] = []
        for i, seed in enumerate(self.seeds):
            dataset.append({
                "problem_id": f"resonance_{self.split}_{i}",
                "seed": seed,
                "prompt": (
                    "You are a physicist conducting an active experiment in circuit spectroscopy. "
                    "Select probe frequencies and submit fitted circuit hypotheses in JSON format.\n"
                    "Remaining measurement budget: 12.\n"
                    "History:\n  (No measurements taken yet)\n"
                    "Provide your next action (measure or submit) as JSON."
                ),
                "rubric": {
                    "evaluation_type": "deterministic_physics_transfer",
                    "budget": 12,
                    "target_band_hz": [30.0, 30000.0],
                },
            })
        return dataset

    def create_env_for_problem(self, problem: Dict[str, Any]) -> CircuitSpectroscopyEnv:
        """Create a fresh environment initialized to the problem's seed."""
        env = CircuitSpectroscopyEnv(**self.env_kwargs)
        env.reset(seed=problem["seed"])
        return env

    def verify(
        self,
        problem: Dict[str, Any],
        trajectory_or_submission: Dict[str, Any] | List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Score a final submission against the problem's ground truth world."""
        env = self.create_env_for_problem(problem)
        if isinstance(trajectory_or_submission, list):
            # Trajectory: replay actions
            obs = env.get_public_observation()
            total_reward = 0.0
            done = False
            last_info: Dict[str, Any] = {}
            for action in trajectory_or_submission:
                obs, r, term, trunc, info = env.step(action)
                total_reward += r
                last_info = info
                if term or trunc:
                    done = True
                    break
            return {
                "reward": float(total_reward),
                "terminated": done,
                "info": last_info,
                "passed": total_reward > 0.40,
            }
        else:
            # Direct final submission
            obs, r, term, trunc, info = env.step(trajectory_or_submission)
            return {
                "reward": float(r),
                "terminated": bool(term),
                "info": info,
                "passed": r > 0.40,
            }
