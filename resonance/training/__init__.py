"""Training utilities for SFT and RL."""

from resonance.training.trajectory_collector import MultiTurnTrajectoryCollector
from resonance.training.sft import generate_sft_dataset
from resonance.training.grpo import compute_grpo_advantages, GRPOMultiTurnCollector, GroupRelativeCollector

__all__ = [
    "MultiTurnTrajectoryCollector",
    "generate_sft_dataset",
    "compute_grpo_advantages",
    "GRPOMultiTurnCollector",
    "GroupRelativeCollector",
]
