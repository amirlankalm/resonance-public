"""One-command smoke run for Resonance (M7).

Runs 5 episodes with the scripted teacher policy and 5 episodes with a random
sampling policy in under 30 seconds on CPU. Prints a structured results table
and exits nonzero on any failure.
"""

from __future__ import annotations
import sys
import time
from typing import Any, Dict, List

import numpy as np

from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.baselines.numerical import (
    run_adaptive_design_baseline,
    run_random_design_baseline,
)


def run_smoke_suite(num_episodes: int = 5, verbose: bool = True) -> int:
    """Execute smoke suite and print formatted validation table."""
    start_time = time.time()
    env = CircuitSpectroscopyEnv(measurement_budget=12, max_turns=20)

    rows: List[Dict[str, Any]] = []
    base_seed = 10000

    if verbose:
        print("=" * 82)
        print(" Resonance Physics RL Environment: One-Command Smoke Verification Suite")
        print("=" * 82)
        print(f"Executing {num_episodes} teacher episodes + {num_episodes} random baseline episodes on CPU...")

    # 1. Scripted Teacher Baseline
    for ep in range(num_episodes):
        seed = base_seed + ep
        try:
            info = run_adaptive_design_baseline(env, max_measurements=6, seed=seed)
            rows.append({
                "policy": "Scripted Teacher (EIG/Active)",
                "episode": ep + 1,
                "seed": seed,
                "calls": info.get("calls_made", 0),
                "nrmse": f"{info.get('nrmse', 0.0):.4f}",
                "reward": f"{info.get('reward', 0.0):.4f}",
                "status": "PASS" if info.get("reward", 0.0) > 0.50 else "WARN",
            })
        except Exception as e:
            if verbose:
                print(f"[FAIL] Teacher episode {ep+1} raised: {e}")
            return 1

    # 2. Random Sampling Baseline
    for ep in range(num_episodes):
        seed = base_seed + ep
        try:
            info = run_random_design_baseline(env, num_measurements=6, seed=seed)
            rows.append({
                "policy": "Random Probes + Fit",
                "episode": ep + 1,
                "seed": seed,
                "calls": info.get("calls_made", 0),
                "nrmse": f"{info.get('nrmse', 0.0):.4f}",
                "reward": f"{info.get('reward', 0.0):.4f}",
                "status": "PASS" if info.get("reward", 0.0) is not None else "FAIL",
            })
        except Exception as e:
            if verbose:
                print(f"[FAIL] Random episode {ep+1} raised: {e}")
            return 1

    elapsed = time.time() - start_time

    if verbose:
        header = f"{'Policy':<28} | {'Ep':<3} | {'Seed':<6} | {'Calls':<5} | {'NRMSE':<7} | {'Reward':<7} | {'Status'}"
        sep = "-" * len(header)
        print("\n" + sep)
        print(header)
        print(sep)
        for r in rows:
            print(
                f"{r['policy']:<28} | {r['episode']:<3} | {r['seed']:<6} | "
                f"{r['calls']:<5} | {r['nrmse']:<7} | {r['reward']:<7} | {r['status']}"
            )
        print(sep)
        print(f"Total smoke verification time: {elapsed:.2f}s (Budget: < 30.0s)")
        print("=" * 82)

    if elapsed > 30.0:
        if verbose:
            print("[FAIL] Execution exceeded 30-second budget.")
        return 1

    if any(r["status"] == "FAIL" for r in rows):
        return 1

    return 0


def main() -> None:
    code = run_smoke_suite(num_episodes=5, verbose=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
