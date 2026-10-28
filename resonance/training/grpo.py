"""Group-Relative Policy Gradient training recipe and multi-turn rollout collection.

Implements advantage-normalized REINFORCE across groups of G parallel rollouts:
A_i = (R_i - mean(R)) / (std(R) + eps).
Discloses explicitly that this is a lightweight training recipe without PPO importance-ratio
clipping or reference-model KL penalties, not an exact standard GRPO replication.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np

from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.runners.runner import EpisodeRunner


def compute_grpo_advantages(rewards: List[float], eps: float = 1e-6) -> Tuple[np.ndarray, bool]:
    """Compute group-relative advantages across G rollouts for the same initial world:

    A_i = (R_i - mean(R)) / (std(R) + eps)

    Returns:
        advantages: np.ndarray of shape (G,)
        is_zero_variance: whether reward variance in group was zero
    """
    arr = np.asarray(rewards, dtype=np.float64)
    mean_r = np.mean(arr)
    std_r = np.std(arr)
    is_zero_variance = bool(std_r < 1e-7)

    if is_zero_variance:
        # Zero variance group: advantages are zeros
        return np.zeros_like(arr), True

    advantages = (arr - mean_r) / (std_r + eps)
    return advantages, False


class GRPOMultiTurnCollector:
    """Collects groups of G multi-turn trajectories per world prompt for GRPO."""

    def __init__(
        self,
        env_class=CircuitSpectroscopyEnv,
        group_size: int = 4,
    ):
        self.env_class = env_class
        self.group_size = group_size

    def collect_group_rollouts(
        self,
        agent,
        world_seed: int,
    ) -> Dict[str, Any]:
        """Collect G rollouts for the same world seed with independent sensor noise."""
        trajectories = []
        rewards = []

        for g in range(self.group_size):
            env = self.env_class()
            # Set same world seed so the underlying circuit ground truth is identical,
            # but pass independent sensor_seed so runner reseeds sensor noise stream
            # AFTER env.reset(), giving each rollout independent sensor noise!
            sensor_seed = world_seed + 1000 * (g + 1)
            runner = EpisodeRunner(env, agent)
            rec = runner.run_episode(
                world_seed=world_seed,
                world_id=f"world_{world_seed}_g{g}",
                sensor_seed=sensor_seed,
            )
            trajectories.append(rec)
            rewards.append(rec["reward"])

        advantages, is_zero_var = compute_grpo_advantages(rewards)

        return {
            "world_seed": world_seed,
            "group_size": self.group_size,
            "rewards": rewards,
            "advantages": advantages.tolist(),
            "is_zero_variance": is_zero_var,
            "trajectories": trajectories,
        }


# Transparent alias for standardized algorithm naming
GroupRelativeCollector = GRPOMultiTurnCollector
