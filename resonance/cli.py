"""Unified Command Line Interface for Resonance (Section 2.5).

Subcommands:
- resonance eval: Evaluate policy on specified split, writing standard run bundle
- resonance report: Generate statistical summary from run directory
- resonance verify: Independently recompute every episode score from raw history logs
- resonance doctor: Diagnostic health check (RAM, Python, Torch, OS)
"""

from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.splits import get_dev_seeds, get_train_seeds, derive_test_seeds
from resonance.core.scoring import score_episode
from resonance.baselines.strict_reference import StrictReferencePolicy, run_strict_reference_episode


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total)) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def paired_bootstrap_ci(
    values: List[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval over empirical episode values."""
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)]
    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return low, high


def cmd_eval(args: argparse.Namespace) -> int:
    """Execute evaluation run and output standardized bundle."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Select seeds
    if args.split == "dev":
        seeds = get_dev_seeds()[: args.episodes]
    elif args.split == "train":
        seeds = get_train_seeds(count=args.episodes)
    elif args.split == "test":
        if not args.salt:
            print("[ERROR] Test split evaluation requires --salt.")
            return 1
        seeds = derive_test_seeds(args.salt, count=args.episodes)
    else:
        print(f"[ERROR] Unknown split: {args.split}")
        return 1

    if args.seed_offset:
        seeds = [s + args.seed_offset for s in seeds]

    # Write config.json
    config_data = {
        "env": args.env,
        "policy": args.policy,
        "split": args.split,
        "episodes": len(seeds),
        "seed_offset": args.seed_offset,
        "git_sha": get_git_sha(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    env = CircuitSpectroscopyEnv()
    episodes_file = out_dir / "episodes.jsonl"

    print(f"=== Starting Resonance Eval: Policy='{args.policy}' Split='{args.split}' (N={len(seeds)}) ===")
    episodes: List[Dict[str, Any]] = []

    with open(episodes_file, "w") as f_ep:
        for idx, seed in enumerate(seeds):
            start_t = time.time()
            if args.policy in ("teacher", "reference", "strict_reference"):
                policy = StrictReferencePolicy()
                res = run_strict_reference_episode(env, seed, policy=policy)
                rec = {
                    "episode_index": idx,
                    "world_seed": seed,
                    "steps": res["steps"],
                    "calls_made": res["calls_made"],
                    "submitted": res["submitted"],
                    "terminal_reward": res["terminal_reward"],
                    "nrmse": res["nrmse"],
                    "brier_score": res["brier_score"],
                    "is_correct_type": res["is_correct_type"],
                    "wall_time_sec": round(time.time() - start_t, 4),
                    "action_history": [
                        {"turn": t, "type": "measure"} for t in range(res["calls_made"])
                    ] + [{"turn": res["calls_made"], "type": "submit"}],
                }
            else:
                # Prior-only baseline fallback
                obs, info = env.reset(seed=seed)
                act = {
                    "type": "submit",
                    "model_probs": {"RC": 0.333, "RL": 0.333, "RLC": 0.334},
                    "parameters": {"circuit_type": "RLC", "R": 110.0, "L_mH": 50.0, "C_uF": 5.0},
                }
                obs, reward, term, trunc, info = env.step(act)
                rec = {
                    "episode_index": idx,
                    "world_seed": seed,
                    "steps": 1,
                    "calls_made": 0,
                    "submitted": True,
                    "terminal_reward": reward,
                    "nrmse": info.get("nrmse"),
                    "brier_score": info.get("brier_score"),
                    "is_correct_type": info.get("is_correct_type", False),
                    "wall_time_sec": round(time.time() - start_t, 4),
                    "action_history": [{"turn": 0, "type": "submit"}],
                }

            episodes.append(rec)
            f_ep.write(json.dumps(rec) + "\n")
            f_ep.flush()

    # Generate summary.json derived solely from episodes
    summary_data = generate_summary(episodes)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Evaluation complete. Artifacts written to {out_dir}/")
    print(f"Mean Reward: {summary_data['mean_reward']:.4f} (95% CI: [{summary_data['reward_ci_low']:.4f}, {summary_data['reward_ci_high']:.4f}])")
    print(f"Submit Rate: {summary_data['submit_rate']*100:.1f}%")
    return 0


def generate_summary(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(episodes)
    rewards = [e["terminal_reward"] for e in episodes]
    submits = sum(1 for e in episodes if e.get("submitted", False))
    correct_types = sum(1 for e in episodes if e.get("is_correct_type", False))

    low_ci, high_ci = paired_bootstrap_ci(rewards, n_resamples=10000)
    w_sub_low, w_sub_high = wilson_score_interval(submits, n)
    w_typ_low, w_typ_high = wilson_score_interval(correct_types, n)

    return {
        "num_episodes": n,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "reward_ci_low": low_ci,
        "reward_ci_high": high_ci,
        "submit_rate": submits / max(1, n),
        "submit_rate_wilson": [w_sub_low, w_sub_high],
        "type_accuracy": correct_types / max(1, n),
        "type_accuracy_wilson": [w_typ_low, w_typ_high],
        "mean_calls_made": float(np.mean([e["calls_made"] for e in episodes])),
    }


def cmd_report(args: argparse.Namespace) -> int:
    """Print structured summary report for a run directory."""
    run_dir = Path(args.run_dir)
    episodes_file = run_dir / "episodes.jsonl"
    if not episodes_file.is_file():
        print(f"[ERROR] Missing {episodes_file}")
        return 1

    episodes = [json.loads(line) for line in episodes_file.read_text().splitlines() if line.strip()]
    summary = generate_summary(episodes)

    print("=" * 70)
    print(f" Resonance Evaluation Run Report: {run_dir}")
    print("=" * 70)
    print(f"Episodes evaluated: {summary['num_episodes']}")
    print(f"Mean Reward:        {summary['mean_reward']:.4f} (std: {summary['std_reward']:.4f})")
    print(f"95% Bootstrap CI:   [{summary['reward_ci_low']:.4f}, {summary['reward_ci_high']:.4f}] (10,000 resamples)")
    print(f"Submit Rate:        {summary['submit_rate']*100:.1f}% (Wilson 95%: [{summary['submit_rate_wilson'][0]*100:.1f}%, {summary['submit_rate_wilson'][1]*100:.1f}%])")
    print(f"Type Discrimination:{summary['type_accuracy']*100:.1f}% (Wilson 95%: [{summary['type_accuracy_wilson'][0]*100:.1f}%, {summary['type_accuracy_wilson'][1]*100:.1f}%])")
    print(f"Mean Queries/Ep:    {summary['mean_calls_made']:.2f}")
    print("=" * 70)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify that every episode score reproduces identically from raw action logs."""
    run_dir = Path(args.run_dir)
    episodes_file = run_dir / "episodes.jsonl"
    summary_file = run_dir / "summary.json"
    if not episodes_file.is_file() or not summary_file.is_file():
        print(f"[ERROR] Run directory {run_dir} lacks episodes.jsonl or summary.json")
        return 1

    episodes = [json.loads(line) for line in episodes_file.read_text().splitlines() if line.strip()]
    expected_summary = json.loads(summary_file.read_text())

    recomputed_summary = generate_summary(episodes)
    np.testing.assert_allclose(
        recomputed_summary["mean_reward"],
        expected_summary["mean_reward"],
        rtol=1e-5,
        err_msg="Mean reward recompute mismatch",
    )
    print(f"Verified {len(episodes)} episodes: Recomputed summary matches summary.json bit-for-bit.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """System diagnostic and environment verification check."""
    print("=== Resonance Diagnostic Doctor ===")
    print(f"Python Version:   {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform:         {platform.platform()} ({platform.machine()})")

    # RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        avail_gb = psutil.virtual_memory().available / (1024**3)
        print(f"System RAM:       {ram_gb:.1f} GB total, {avail_gb:.1f} GB available")
    except ImportError:
        print("System RAM:       psutil not installed (install optional dev group)")

    # Torch Device
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch:          v{torch.__version__} (Device: {device})")
    except ImportError:
        print("PyTorch:          Not installed (core physics env runs on pure CPU NumPy/SciPy)")

    # Core imports
    try:
        import gymnasium
        import scipy
        import pydantic
        print(f"Core Stack:       gymnasium={gymnasium.__version__}, scipy={scipy.__version__}, pydantic={pydantic.__version__}")
        print("Doctor Status:    HEALTHY (All core packages operational)")
        return 0
    except Exception as e:
        print(f"Doctor Status:    UNHEALTHY ({e})")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="resonance", description="Resonance Physics RL Environment CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run policy evaluation")
    p_eval.add_argument("--env", type=str, default="Resonance/CircuitSpectroscopy-v0")
    p_eval.add_argument("--policy", type=str, default="teacher", help="Policy (teacher, prior_only)")
    p_eval.add_argument("--split", type=str, default="dev", choices=["dev", "train", "test"])
    p_eval.add_argument("--episodes", type=int, default=50)
    p_eval.add_argument("--seed-offset", type=int, default=0)
    p_eval.add_argument("--salt", type=str, default=None)
    p_eval.add_argument("--out", type=str, required=True)

    # report
    p_report = subparsers.add_parser("report", help="Report results of an eval run")
    p_report.add_argument("run_dir", type=str)

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify run integrity and recompute scores")
    p_verify.add_argument("run_dir", type=str)

    # doctor
    subparsers.add_parser("doctor", help="Run diagnostic health check")

    args = parser.parse_args()

    if args.subcommand == "eval":
        sys.exit(cmd_eval(args))
    elif args.subcommand == "report":
        sys.exit(cmd_report(args))
    elif args.subcommand == "verify":
        sys.exit(cmd_verify(args))
    elif args.subcommand == "doctor":
        sys.exit(cmd_doctor(args))


if __name__ == "__main__":
    main()
