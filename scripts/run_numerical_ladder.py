"""Script to run the Gate B Numerical Ladder (0, 3, 6, 9, 12 calls).

Evaluates:
- Prior-only (0 calls)
- Random design
- Fixed log sweep
- Adaptive sequential experimental design
across development seeds.
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np

from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.baselines.numerical import (
    run_prior_only_baseline,
    run_random_design_baseline,
    run_fixed_sweep_baseline,
    run_adaptive_design_baseline,
)


def paired_bootstrap_ci(a, b, n_boot: int = 10000, seed: int = 0):
    """Compute 95% bootstrap confidence interval of paired difference (a - b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (a[idx] - b[idx]).mean()
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def run_ladder(num_seeds: int = 200, output_path: str = "artifacts/numerical_ladder_results.json"):
    budgets = [3, 6, 9, 12]
    results = {
        "prior_only": {},
        "random": {},
        "fixed": {},
        "adaptive": {},
    }

    print(f"=== Running Numerical Baseline Ladder across {num_seeds} seeds ===")

    # 1. Prior-only (0 calls)
    prior_nrmses, prior_rews, prior_briers, prior_accs = [], [], [], []
    for s in range(num_seeds):
        env = CircuitSpectroscopyEnv()
        info = run_prior_only_baseline(env, seed=s)
        prior_nrmses.append(info["nrmse"])
        prior_rews.append(info["reward"])
        prior_briers.append(info["brier_score"])
        prior_accs.append(1.0 if info["is_correct_type"] else 0.0)

    results["prior_only"] = {
        "calls": 0,
        "mean_nrmse": float(np.mean(prior_nrmses)),
        "mean_reward": float(np.mean(prior_rews)),
        "mean_brier": float(np.mean(prior_briers)),
        "type_acc": float(np.mean(prior_accs)),
    }
    print(f"Prior-only (0 calls): NRMSE={results['prior_only']['mean_nrmse']:.4f}, Reward={results['prior_only']['mean_reward']:.4f}, TypeAcc={results['prior_only']['type_acc']*100:.1f}%")

    for q in budgets:
        # 2. Random Design
        rnd_nrmses, rnd_rews, rnd_briers, rnd_accs = [], [], [], []
        for s in range(num_seeds):
            env = CircuitSpectroscopyEnv()
            info = run_random_design_baseline(env, num_measurements=q, seed=s)
            rnd_nrmses.append(info["nrmse"])
            rnd_rews.append(info["reward"])
            rnd_briers.append(info["brier_score"])
            rnd_accs.append(1.0 if info["is_correct_type"] else 0.0)

        results["random"][str(q)] = {
            "calls": q,
            "mean_nrmse": float(np.mean(rnd_nrmses)),
            "mean_reward": float(np.mean(rnd_rews)),
            "mean_brier": float(np.mean(rnd_briers)),
            "type_acc": float(np.mean(rnd_accs)),
        }

        # 3. Fixed Log Sweep
        fix_nrmses, fix_rews, fix_briers, fix_accs = [], [], [], []
        for s in range(num_seeds):
            env = CircuitSpectroscopyEnv()
            info = run_fixed_sweep_baseline(env, num_measurements=q, seed=s)
            fix_nrmses.append(info["nrmse"])
            fix_rews.append(info["reward"])
            fix_briers.append(info["brier_score"])
            fix_accs.append(1.0 if info["is_correct_type"] else 0.0)

        results["fixed"][str(q)] = {
            "calls": q,
            "mean_nrmse": float(np.mean(fix_nrmses)),
            "mean_reward": float(np.mean(fix_rews)),
            "mean_brier": float(np.mean(fix_briers)),
            "type_acc": float(np.mean(fix_accs)),
        }

        # 4. Adaptive Design
        adp_nrmses, adp_rews, adp_briers, adp_accs = [], [], [], []
        for s in range(num_seeds):
            env = CircuitSpectroscopyEnv()
            info = run_adaptive_design_baseline(env, max_measurements=q, seed=s)
            adp_nrmses.append(info["nrmse"])
            adp_rews.append(info["reward"])
            adp_briers.append(info["brier_score"])
            adp_accs.append(1.0 if info["is_correct_type"] else 0.0)

        diff_ci = paired_bootstrap_ci(adp_rews, fix_rews)
        results["adaptive"][str(q)] = {
            "calls": q,
            "mean_nrmse": float(np.mean(adp_nrmses)),
            "mean_reward": float(np.mean(adp_rews)),
            "mean_brier": float(np.mean(adp_briers)),
            "type_acc": float(np.mean(adp_accs)),
            "reward_minus_fixed_ci95": diff_ci,
        }

        print(f"Budget q={q:2d}:")
        print(f"  Random:   NRMSE={results['random'][str(q)]['mean_nrmse']:.4f}, Reward={results['random'][str(q)]['mean_reward']:.4f}, TypeAcc={results['random'][str(q)]['type_acc']*100:.1f}%")
        print(f"  Fixed:    NRMSE={results['fixed'][str(q)]['mean_nrmse']:.4f}, Reward={results['fixed'][str(q)]['mean_reward']:.4f}, TypeAcc={results['fixed'][str(q)]['type_acc']*100:.1f}%")
        print(f"  Adaptive: NRMSE={results['adaptive'][str(q)]['mean_nrmse']:.4f}, Reward={results['adaptive'][str(q)]['mean_reward']:.4f}, TypeAcc={results['adaptive'][str(q)]['type_acc']*100:.1f}%")
        print(f"  Paired diff (Adaptive - Fixed) 95% CI: [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gate B Numerical Ladder")
    parser.add_argument("--worlds", "--seeds", "--episodes", dest="num_seeds", type=int, default=200, help="Number of worlds/episodes (default: 200)")
    parser.add_argument("--output", type=str, default="artifacts/numerical_ladder_results.json", help="Output JSON path")
    args = parser.parse_args()
    run_ladder(num_seeds=args.num_seeds, output_path=args.output)
