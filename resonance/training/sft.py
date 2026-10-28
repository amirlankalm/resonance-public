"""Supervised Fine-Tuning (SFT) dataset preparation and training utilities."""

from __future__ import annotations
import json
import os
from typing import List, Dict, Any
from resonance.training.trajectory_collector import MultiTurnTrajectoryCollector


def generate_sft_dataset(
    num_trajectories: int = 100,
    output_jsonl: str = "artifacts/sft_trajectories.jsonl",
    base_seed: int = 50000,
) -> str:
    """Generate checked supervised expert trajectories for SFTTrainer."""
    collector = MultiTurnTrajectoryCollector()
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    print(f"[SFT] Generating {num_trajectories} expert trajectories...")
    records = []
    for i in range(num_trajectories):
        seed = base_seed + i
        traj = collector.collect_expert_trajectory(seed)
        records.append(traj)

    with open(output_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    mean_rew = sum(r["reward"] for r in records) / len(records)
    print(f"[SFT] Saved {len(records)} trajectories to {output_jsonl} (Mean reward: {mean_rew:.4f})")
    return output_jsonl
