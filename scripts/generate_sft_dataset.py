"""Generate checked supervised expert trajectories for SFT fine-tuning."""

from __future__ import annotations
import argparse
from resonance.training.sft import generate_sft_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate SFT Trajectory Dataset")
    parser.add_argument("--count", type=int, default=100, help="Number of trajectories to generate")
    parser.add_argument("--output", type=str, default="artifacts/sft_trajectories.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=50000, help="Base random seed")
    args = parser.parse_args()

    generate_sft_dataset(
        num_trajectories=args.count,
        output_jsonl=args.output,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
