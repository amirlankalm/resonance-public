"""Run the unified headline evaluation comparing Base Qwen, SFT-only, Resonance RL, and EIG Teacher.

Generates the comprehensive evaluation table required by Report v2 and Oct 26 Update.
Evaluates across locked dev worlds (seeds 10000..10000+N-1) with 95% bootstrap confidence intervals.
Evaluates both STRICT and ASSISTED tracks for all models for strict like-for-like comparison.
"""

from __future__ import annotations
import argparse
import os
import json
import numpy as np

from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.models.prompted_llm import PromptedLLMAgent
from resonance.runners.runner import EpisodeRunner
from resonance.runners.manifest import RunArtifactLogger
from resonance.core.memory import MemoryGuard
from resonance.benchmarks.active_benchmark import compute_bootstrap_ci
from resonance.baselines.numerical import run_adaptive_design_baseline


def compute_paired_bootstrap_ci(a_vals: list, b_vals: list, num_bootstrap: int = 2000, ci_level: float = 0.95):
    """Compute 95% paired bootstrap confidence interval for Delta = A - B."""
    diffs = np.array(a_vals) - np.array(b_vals)
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "excludes_zero": False}
    boot_means = [float(np.mean(np.random.choice(diffs, size=n, replace=True))) for _ in range(num_bootstrap)]
    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    excludes_zero = bool((low > 0 and high > 0) or (low < 0 and high < 0))
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": excludes_zero,
    }


def run_model_eval(
    model_path: str,
    num_worlds: int,
    base_seed: int,
    output_dir: str,
    name: str,
    track: str = "assisted",
    allow_mock: bool = False,
    submission_timing_bonus: float = 0.0,
):
    print(f"\n>>> Running evaluation for: {name} ({model_path}, track={track}) on {num_worlds} worlds (seeds {base_seed}..{base_seed+num_worlds-1})...")
    env = CircuitSpectroscopyEnv(submission_timing_bonus=submission_timing_bonus)
    agent = PromptedLLMAgent(model_name_or_path=model_path, track=track, allow_mock=allow_mock, max_new_tokens=80)
    
    logger = RunArtifactLogger(
        output_dir=output_dir,
        experiment_name=f"unified_eval_{name}_{track}",
        model_name=model_path,
        decoding_params={"temperature": agent.temperature, "top_p": agent.top_p},
        prompt_text="Active circuit spectroscopy evaluation",
        task_name="circuit_spectroscopy",
        mode=track,
        is_mock=agent.is_mock,
        loader_confirmed=agent.loader_confirmed,
    )
    runner = EpisodeRunner(env, agent, artifact_logger=logger)
    
    episodes = []
    for i in range(num_worlds):
        seed = base_seed + i
        rec = runner.run_episode(world_seed=seed, world_id=f"dev_world_{i}")
        episodes.append(rec)
        status = "PASS" if rec.get("is_correct_type") else f"FAIL ({rec.get('failure_category', 'err')})"
        nrmse_disp = f"{rec['nrmse']:.4f}" if rec.get('nrmse') is not None else "N/A"
        if (i + 1) % 5 == 0 or (i + 1) == num_worlds:
            print(f"  [{name} - {track}] World {i+1}/{num_worlds} (seed={seed}) -> Calls: {rec['calls_made']}, NRMSE: {nrmse_disp}, Reward: {rec['terminal_reward']:.4f}, TypeAcc: {status}")
        MemoryGuard.cleanup_memory()

    logger.write_artifacts()
    del agent
    del runner
    MemoryGuard.cleanup_memory()

    terminal_rewards = [ep["terminal_reward"] for ep in episodes]
    episode_returns = [ep["episode_return"] for ep in episodes]
    submitted_eps = [ep for ep in episodes if ep.get("submitted") is True]
    nrmses = [ep["nrmse"] for ep in submitted_eps if ep.get("nrmse") is not None]
    briers = [ep["brier_score"] for ep in submitted_eps if ep.get("brier_score") is not None]
    unconditional_accs = [1.0 if ep.get("is_correct_type") is True else 0.0 for ep in episodes]
    calls = [ep["calls_made"] for ep in episodes]
    valid_rates = [ep.get("valid_action_rate", 1.0) for ep in episodes]

    return {
        "name": name,
        "model_path": model_path,
        "track": track,
        "worlds": num_worlds,
        "reward": compute_bootstrap_ci(terminal_rewards),
        "episode_return": compute_bootstrap_ci(episode_returns),
        "nrmse": compute_bootstrap_ci(nrmses) if nrmses else None,
        "n_submitted_nrmse": len(nrmses),
        "brier_score": compute_bootstrap_ci(briers) if briers else None,
        "type_acc": compute_bootstrap_ci(unconditional_accs),
        "mean_calls": float(np.mean(calls)),
        "mean_valid_action_rate": float(np.mean(valid_rates)),
        "episodes": episodes,
    }


def run_teacher_baseline_eval(num_worlds: int, base_seed: int, submission_timing_bonus: float = 0.0):
    print(f"\n>>> Running evaluation for: Adaptive Disagreement Design (Teacher) on {num_worlds} worlds...")
    episodes = []
    for i in range(num_worlds):
        seed = base_seed + i
        env = CircuitSpectroscopyEnv(submission_timing_bonus=submission_timing_bonus)
        info = run_adaptive_design_baseline(env, max_measurements=6, seed=seed)
        episodes.append(info)
        if (i + 1) % 10 == 0 or (i + 1) == num_worlds:
            print(f"  [Teacher] World {i+1}/{num_worlds} (seed={seed}) -> Calls: {info['calls_made']}, NRMSE: {info['nrmse']:.4f}, Reward: {info['reward']:.4f}")

    rewards = [ep["reward"] for ep in episodes]
    nrmses = [ep["nrmse"] for ep in episodes]
    briers = [ep["brier_score"] for ep in episodes]
    type_accs = [1.0 if ep["is_correct_type"] else 0.0 for ep in episodes]
    calls = [ep["calls_made"] for ep in episodes]

    return {
        "name": "Adaptive Disagreement Design (Teacher)",
        "model_path": "adaptive_numerical_teacher",
        "track": "assisted",
        "worlds": num_worlds,
        "reward": compute_bootstrap_ci(rewards),
        "nrmse": compute_bootstrap_ci(nrmses),
        "brier_score": compute_bootstrap_ci(briers),
        "type_acc": compute_bootstrap_ci(type_accs),
        "mean_calls": float(np.mean(calls)),
        "mean_valid_action_rate": 1.0,
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=50, help="Number of locked dev worlds (default: 50, seeds 10000..10049)")
    parser.add_argument("--base_seed", type=int, default=10000, help="Base seed for dev worlds")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft_model", type=str, default="checkpoints/resonance_qwen_sft")
    parser.add_argument("--rl_model", type=str, default="checkpoints/resonance_qwen_rl")
    parser.add_argument("--output_json", type=str, default="artifacts/unified_headline_eval.json")
    parser.add_argument("--allow_mock", action="store_true", default=False)
    parser.add_argument("--submission_timing_bonus", type=float, default=0.0, help="Explicit completion bonus for timely submission before turn exhaustion (default: 0.0)")
    args = parser.parse_args()

    print("=================================================================")
    print(f"   RESONANCE UNIFIED HEADLINE EVALUATION (N={args.worlds} Dev Worlds)   ")
    print("=================================================================")

    # 1. Evaluate Teacher Baseline (Adaptive Disagreement Design)
    teacher_res = run_teacher_baseline_eval(args.worlds, args.base_seed, submission_timing_bonus=args.submission_timing_bonus)

    # 2. Evaluate Base Model in Strict track
    base_strict_res = run_model_eval(
        model_path=args.base_model,
        num_worlds=args.worlds,
        base_seed=args.base_seed,
        output_dir="artifacts/model_eval_qwen_base_strict",
        name="Qwen2.5-0.5B (Frozen Base)",
        track="strict",
        allow_mock=args.allow_mock,
        submission_timing_bonus=args.submission_timing_bonus,
    )

    # 3. Evaluate Base Model in Assisted track (Like-for-like comparison!)
    base_assisted_res = run_model_eval(
        model_path=args.base_model,
        num_worlds=args.worlds,
        base_seed=args.base_seed,
        output_dir="artifacts/model_eval_qwen_base",
        name="Qwen2.5-0.5B (Frozen Base)",
        track="assisted",
        allow_mock=args.allow_mock,
        submission_timing_bonus=args.submission_timing_bonus,
    )

    # 4. Evaluate SFT-only model in Strict track
    sft_strict_res = None
    if os.path.exists(args.sft_model):
        sft_strict_res = run_model_eval(
            model_path=args.sft_model,
            num_worlds=args.worlds,
            base_seed=args.base_seed,
            output_dir="artifacts/model_eval_qwen_sft_strict",
            name="Qwen2.5-0.5B (SFT-Only)",
            track="strict",
            allow_mock=args.allow_mock,
            submission_timing_bonus=args.submission_timing_bonus,
        )

    # 5. Evaluate SFT-only model in Assisted track
    sft_assisted_res = None
    if os.path.exists(args.sft_model):
        sft_assisted_res = run_model_eval(
            model_path=args.sft_model,
            num_worlds=args.worlds,
            base_seed=args.base_seed,
            output_dir="artifacts/model_eval_qwen_sft",
            name="Qwen2.5-0.5B (SFT-Only)",
            track="assisted",
            allow_mock=args.allow_mock,
            submission_timing_bonus=args.submission_timing_bonus,
        )

    # 6. Evaluate RL Model in Strict track
    rl_strict_res = None
    if os.path.exists(args.rl_model):
        rl_strict_res = run_model_eval(
            model_path=args.rl_model,
            num_worlds=args.worlds,
            base_seed=args.base_seed,
            output_dir="artifacts/model_eval_qwen_rl_strict",
            name="Qwen2.5-0.5B (Resonance RL)",
            track="strict",
            allow_mock=args.allow_mock,
            submission_timing_bonus=args.submission_timing_bonus,
        )

    # 7. Evaluate RL Model in Assisted track
    rl_assisted_res = None
    if os.path.exists(args.rl_model):
        rl_assisted_res = run_model_eval(
            model_path=args.rl_model,
            num_worlds=args.worlds,
            base_seed=args.base_seed,
            output_dir="artifacts/model_eval_qwen_rl",
            name="Qwen2.5-0.5B (Resonance RL)",
            track="assisted",
            allow_mock=args.allow_mock,
            submission_timing_bonus=args.submission_timing_bonus,
        )

    # Paired differences across models on the exact same locked dev worlds
    paired_diffs = {}
    if rl_assisted_res and sft_assisted_res:
        rl_rews = [ep["terminal_reward"] for ep in rl_assisted_res["episodes"]]
        sft_rews = [ep["terminal_reward"] for ep in sft_assisted_res["episodes"]]
        base_rews = [ep["terminal_reward"] for ep in base_assisted_res["episodes"]]
        paired_diffs["delta_rl_minus_sft_assisted"] = compute_paired_bootstrap_ci(rl_rews, sft_rews)
        paired_diffs["delta_rl_minus_base_assisted"] = compute_paired_bootstrap_ci(rl_rews, base_rews)

    def _compact_res(r: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not r or "episodes" not in r:
            return r
        out = dict(r)
        fields = [
            "world_id", "world_seed", "sensor_seed", "steps", "calls_made",
            "terminated", "truncated", "submitted", "terminal_reward", "episode_return",
            "reward", "assisted", "fit_source", "invalid_action_count", "valid_action_rate",
            "nrmse", "rmse", "brier_score", "is_correct_type", "failure_category",
            "failure_detail", "elapsed_seconds"
        ]
        out["episodes"] = [{k: ep.get(k) for k in fields} for ep in r["episodes"]]
        return out

    results = {
        "num_worlds": args.worlds,
        "base_seed": args.base_seed,
        "teacher_baseline": _compact_res(teacher_res),
        "base_model_strict": _compact_res(base_strict_res),
        "base_model_assisted": _compact_res(base_assisted_res),
        "sft_model_strict": _compact_res(sft_strict_res),
        "sft_model_assisted": _compact_res(sft_assisted_res),
        "rl_model_strict": _compact_res(rl_strict_res),
        "rl_model_assisted": _compact_res(rl_assisted_res),
        # Backward compatibility aliases for existing scripts
        "base_model": _compact_res(base_strict_res),
        "sft_model": _compact_res(sft_assisted_res),
        "rl_model": _compact_res(rl_assisted_res),
        "paired_differences": paired_diffs,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as fp:
        json.dump(results, fp, indent=2)

    print("\n" + "="*100)
    print("UNIFIED HEADLINE BENCHMARK TABLE (95% BOOTSTRAP CONFIDENCE INTERVALS)")
    print("="*100)
    fmt = "| {:<32} | {:<10} | {:<24} | {:<18} | {:<16} | {:<6} |"
    print(fmt.format("Model / Agent", "Track", "Mean TermReward", "Conditional NRMSE", "Type Accuracy", "Calls"))
    print("|" + "-"*34 + "|" + "-"*12 + "|" + "-"*26 + "|" + "-"*20 + "|" + "-"*18 + "|" + "-"*8 + "|")

    all_runs = [
        teacher_res,
        base_strict_res,
        base_assisted_res,
    ]
    if sft_strict_res:
        all_runs.append(sft_strict_res)
    if sft_assisted_res:
        all_runs.append(sft_assisted_res)
    if rl_strict_res:
        all_runs.append(rl_strict_res)
    if rl_assisted_res:
        all_runs.append(rl_assisted_res)

    for r in all_runs:
        rew_str = f"{r['reward']['mean']:.4f} [{r['reward']['ci_low']:.4f}, {r['reward']['ci_high']:.4f}]"
        if r.get("nrmse") and r["nrmse"].get("mean") is not None:
            nrmse_str = f"{r['nrmse']['mean']:.4f} (n={r.get('n_submitted_nrmse', len(r['episodes']))})"
        else:
            nrmse_str = "None (no submit)"
        acc_str = f"{r['type_acc']['mean']*100:.1f}% [{r['type_acc']['ci_low']*100:.1f}%, {r['type_acc']['ci_high']*100:.1f}%]"
        calls_str = f"{r['mean_calls']:.1f}"
        print(fmt.format(r["name"], r.get("track", "assisted"), rew_str, nrmse_str, acc_str, calls_str))
    print("="*100)

    if paired_diffs:
        print("\n--- PAIRED WORLD DIFFERENCES (ASSISTED TRACK) ---")
        p_sft = paired_diffs.get("delta_rl_minus_sft_assisted", {})
        p_base = paired_diffs.get("delta_rl_minus_base_assisted", {})
        print(f"  Δ(RL - SFT):  {p_sft.get('mean_diff', 0):+.4f} [{p_sft.get('ci_low', 0):+.4f}, {p_sft.get('ci_high', 0):+.4f}] (Excludes 0: {p_sft.get('excludes_zero')})")
        print(f"  Δ(RL - Base): {p_base.get('mean_diff', 0):+.4f} [{p_base.get('ci_low', 0):+.4f}, {p_base.get('ci_high', 0):+.4f}] (Excludes 0: {p_base.get('excludes_zero')})")
        print("="*100 + "\n")


if __name__ == "__main__":
    main()
