"""Run active experiment evaluation on an open-source model.

Generates the 4 required run artifacts:
1. run_manifest.json
2. episodes.jsonl
3. summary.json
4. failures.md

Evaluates agents and baselines with 95% bootstrap confidence intervals on locked dev worlds.
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


def main():
    parser = argparse.ArgumentParser(description="Run Active Physics Experiment Evaluation")
    parser.add_argument("--model", type=str, default="checkpoints/resonance_qwen_rl", help="HF model name or checkpoint path")
    parser.add_argument("--mode", type=str, default="assisted", choices=["strict", "assisted"], help="Evaluation track: strict (pure model output) or assisted (disclosed numerical fitter)")
    parser.add_argument("--allow_mock", action="store_true", default=False, help="Allow fallback mock model (strictly for unit tests; fail-closed by default)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation worlds on dev split (locked headline protocol: 50, seeds 10000..10049; use --episodes 10 for quick smoke / 7B)")
    parser.add_argument("--base_seed", type=int, default=10000, help="Base seed for dev worlds (default: 10000)")
    parser.add_argument("--output_dir", type=str, default="artifacts/model_eval_qwen_rl", help="Artifacts directory")
    parser.add_argument("--submission_timing_bonus", type=float, default=0.0, help="Explicit completion bonus for timely submission before turn exhaustion (default: 0.0)")
    args = parser.parse_args()

    mem_init = MemoryGuard.get_system_memory_info()
    print(f"=== Resonance Active Experiment Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Track: {args.mode}")
    print(f"Allow Mock: {args.allow_mock}")
    print(f"Episodes: {args.episodes} (Dev Split, seeds {args.base_seed}..{args.base_seed + args.episodes - 1})")
    print(f"Output directory: {args.output_dir}")
    print(f"[Reproduction Note] To reproduce exact published artifacts, pass exact manifest args:")
    print(f"  Headline 50-world evaluation: --episodes 50 --base_seed 10000")
    print(f"  Exploratory 10-world 7B evaluation: --episodes 10 --base_seed 10000 --model ollama:qwen2.5:7b")
    print(f"[Memory Guard] System RAM: {mem_init['total_ram_mb']} MB, Available: {mem_init['available_ram_mb']} MB ({mem_init['percent_used']}% used)")

    env = CircuitSpectroscopyEnv(submission_timing_bonus=args.submission_timing_bonus)
    agent = PromptedLLMAgent(
        model_name_or_path=args.model,
        mode=args.mode,
        allow_mock=args.allow_mock,
    )

    logger = RunArtifactLogger(
        output_dir=args.output_dir,
        experiment_name=f"qwen_active_circuit_eval_{args.mode}",
        model_name=args.model,
        decoding_params={"temperature": agent.temperature, "top_p": agent.top_p},
        prompt_text="Active circuit spectroscopy experiment",
        task_name="circuit_spectroscopy",
        mode=args.mode,
        is_mock=agent.is_mock,
        loader_confirmed=agent.loader_confirmed,
    )

    runner = EpisodeRunner(env, agent, artifact_logger=logger)

    episodes = []
    for i in range(args.episodes):
        seed = args.base_seed + i
        print(f"Running episode {i+1}/{args.episodes} (world_seed={seed})...")
        rec = runner.run_episode(world_seed=seed, world_id=f"dev_world_{i}", execution_mode=args.mode)
        episodes.append(rec)
        status_str = "SUCCESS" if rec.get("is_correct_type") else f"FAIL ({rec.get('failure_category')})"
        MemoryGuard.cleanup_memory()
        mem_step = MemoryGuard.get_system_memory_info()
        print(f"  -> Calls: {rec['calls_made']}, NRMSE: {rec['nrmse']}, TermReward: {rec['terminal_reward']:.4f}, Return: {rec['episode_return']:.4f}, Status: {status_str} | RSS: {mem_step['process_rss_mb']} MB")

    artifact_paths = logger.write_artifacts()
    print(f"\nArtifacts successfully written to {args.output_dir}:")
    for k, v in artifact_paths.items():
        print(f"  - {k}: {v}")

    # Compute aggregate statistics with 95% bootstrap CIs (R5)
    term_rewards = [ep["terminal_reward"] for ep in episodes]
    ep_returns = [ep["episode_return"] for ep in episodes]

    # Unconditional accuracy: denominator is ALL assigned worlds (N)
    unconditional_accs = [1.0 if ep.get("is_correct_type") is True else 0.0 for ep in episodes]

    # Conditional metrics on submitted worlds
    submitted_eps = [ep for ep in episodes if ep.get("submitted") is True]
    nrmses = [ep["nrmse"] for ep in submitted_eps if ep.get("nrmse") is not None]
    briers = [ep["brier_score"] for ep in submitted_eps if ep.get("brier_score") is not None]

    term_rew_ci = compute_bootstrap_ci(term_rewards)
    ep_ret_ci = compute_bootstrap_ci(ep_returns)
    acc_ci = compute_bootstrap_ci(unconditional_accs)

    nrmse_ci = compute_bootstrap_ci(nrmses) if nrmses else None
    brier_ci = compute_bootstrap_ci(briers) if briers else None

    print("\n" + "="*70)
    print(f"AGGREGATE BENCHMARK RESULTS (TRACK: {args.mode.upper()}, N={len(episodes)} WORLDS):")
    print(f"  Mean Terminal Reward: {term_rew_ci['mean']:.4f} [95% CI: {term_rew_ci['ci_low']:.4f}, {term_rew_ci['ci_high']:.4f}]")
    print(f"  Mean Episode Return:  {ep_ret_ci['mean']:.4f} [95% CI: {ep_ret_ci['ci_low']:.4f}, {ep_ret_ci['ci_high']:.4f}]")
    print(f"  Unconditional Acc:    {acc_ci['mean']*100:.1f}% [95% CI: {acc_ci['ci_low']*100:.1f}%, {acc_ci['ci_high']*100:.1f}%] (n={len(episodes)})")
    if nrmse_ci is not None:
        print(f"  Conditional NRMSE:    {nrmse_ci['mean']:.4f} [95% CI: {nrmse_ci['ci_low']:.4f}, {nrmse_ci['ci_high']:.4f}] (n={len(nrmses)} submitted)")
    else:
        print(f"  Conditional NRMSE:    None (no valid submissions)")
    if brier_ci is not None:
        print(f"  Conditional Brier:    {brier_ci['mean']:.4f} [95% CI: {brier_ci['ci_low']:.4f}, {brier_ci['ci_high']:.4f}] (n={len(briers)} submitted)")
    else:
        print(f"  Conditional Brier:    None (no valid submissions)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
