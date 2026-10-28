"""Comprehensive Multi-Benchmark Evaluation Harness: Base vs Exploratory LoRA Adapter.

Runs the complete battery of scientific physics benchmarks:
1. UGPhysics (ICML 2025 - HKUST)
2. PhysReason (ACL 2025)
3. PHYSICS-Verified (Yale NLP)
4. BoxingGym (Stanford - Active Experimental Design)
5. LLM-SRBench (ICML - Symbolic Regression & Law Discovery)
6. Resonance Active Circuit Spectroscopy (Active RL Task)

Compares:
- Baseline Model (Frozen Qwen2.5-0.5B-Instruct zero-shot)
- Exploratory Checkpoint (Shared LoRA Adapter, Pass B)
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
from typing import Dict, Any

from resonance.core.memory import MemoryGuard
from resonance.models.prompted_llm import PromptedLLMAgent
from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.runners.runner import EpisodeRunner
from resonance.runners.manifest import RunArtifactLogger

from resonance.benchmarks.ugphysics import UGPHYSICS_PROBLEMS, evaluate_ugphysics
from resonance.benchmarks.phys_reason import PHYSREASON_PROBLEMS, evaluate_physreason
from resonance.benchmarks.physics_verified import (
    PHYSICS_VERIFIED_PROBLEMS,
    evaluate_physics_verified,
)
from resonance.benchmarks.boxing_gym import evaluate_boxing_gym
from resonance.benchmarks.llm_srbench import SR_PROBLEMS, evaluate_llm_srbench


def evaluate_model_on_all_benchmarks(
    model_path: str,
    label: str,
    circuit_episodes: int = 5,
    boxing_gym_episodes: int = 3,
) -> Dict[str, Any]:
    """Run all benchmarks on a single model and record detailed metrics."""
    print(f"\n==================================================================")
    print(f"=== Evaluating: {label} ({model_path}) ===")
    print(f"==================================================================")

    mem = MemoryGuard.get_system_memory_info()
    print(f"[MemoryGuard] System RAM: {mem['total_ram_mb']} MB | Free: {mem['available_ram_mb']} MB")

    agent = PromptedLLMAgent(model_name_or_path=model_path)
    report: Dict[str, Any] = {"model_label": label, "model_path": model_path, "benchmarks": {}}

    # 1. UGPhysics
    print("\n[1/6] Running UGPhysics (ICML 2025)...")
    t0 = time.time()
    ug_res = evaluate_ugphysics(agent, UGPHYSICS_PROBLEMS)
    MemoryGuard.cleanup_memory()
    report["benchmarks"]["ugphysics"] = {
        "accuracy": ug_res["accuracy"],
        "correct": ug_res["correct"],
        "total": ug_res["total_problems"],
        "domain_breakdown": ug_res["domain_breakdown"],
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(f"  -> UGPhysics Accuracy: {ug_res['accuracy']*100:.1f}% ({ug_res['correct']}/{ug_res['total_problems']})")

    # 2. PhysReason
    print("\n[2/6] Running PhysReason (ACL 2025)...")
    t0 = time.time()
    pr_res = evaluate_physreason(agent, PHYSREASON_PROBLEMS)
    MemoryGuard.cleanup_memory()
    report["benchmarks"]["phys_reason"] = {
        "accuracy": pr_res["accuracy"],
        "correct": pr_res["correct"],
        "total": pr_res["total_problems"],
        "difficulty_breakdown": pr_res["difficulty_breakdown"],
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(f"  -> PhysReason Accuracy: {pr_res['accuracy']*100:.1f}% ({pr_res['correct']}/{pr_res['total_problems']})")

    # 3. PHYSICS-Verified
    print("\n[3/6] Running PHYSICS-Verified (Yale NLP)...")
    t0 = time.time()
    pv_res = evaluate_physics_verified(agent, PHYSICS_VERIFIED_PROBLEMS)
    MemoryGuard.cleanup_memory()
    report["benchmarks"]["physics_verified"] = {
        "accuracy": pv_res["accuracy"],
        "correct": pv_res["correct"],
        "total": pv_res["total_problems"],
        "domain_breakdown": pv_res["domain_breakdown"],
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(f"  -> PHYSICS-Verified Accuracy: {pv_res['accuracy']*100:.1f}% ({pv_res['correct']}/{pv_res['total_problems']})")

    # 4. BoxingGym (Stanford Active Design)
    print("\n[4/6] Running BoxingGym (Stanford Active Discovery)...")
    t0 = time.time()
    bg_res = evaluate_boxing_gym(agent, num_episodes=boxing_gym_episodes, budget_queries=4)
    MemoryGuard.cleanup_memory()
    report["benchmarks"]["boxing_gym"] = {
        "mean_nrmse": bg_res["mean_nrmse"],
        "mean_reward": bg_res["mean_reward"],
        "episodes": bg_res["episodes_evaluated"],
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(f"  -> BoxingGym Mean NRMSE: {bg_res['mean_nrmse']:.4f} | Mean Reward: {bg_res['mean_reward']:.4f}")

    # 5. LLM-SRBench (ICML Symbolic Regression)
    print("\n[5/6] Running LLM-SRBench (ICML Equation Discovery)...")
    t0 = time.time()
    sr_res = evaluate_llm_srbench(agent, SR_PROBLEMS)
    MemoryGuard.cleanup_memory()
    report["benchmarks"]["llm_srbench"] = {
        "accuracy": sr_res["accuracy"],
        "correct": sr_res["correct"],
        "total": sr_res["total_problems"],
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(f"  -> LLM-SRBench Equation Discovery: {sr_res['accuracy']*100:.1f}% ({sr_res['correct']}/{sr_res['total_problems']})")

    # 6. Resonance Active Circuit Spectroscopy
    print("\n[6/6] Running Resonance Active Circuit Spectroscopy...")
    t0 = time.time()
    env = CircuitSpectroscopyEnv()
    runner = EpisodeRunner(env, agent)
    circuit_rewards = []
    circuit_nrmses = []
    circuit_correct_types = []
    budget_overruns = 0

    for ep in range(circuit_episodes):
        rec = runner.run_episode(world_seed=5000 + ep, world_id=f"eval_{ep}")
        circuit_rewards.append(float(rec["reward"]))
        if rec.get("nrmse") is not None:
            circuit_nrmses.append(float(rec["nrmse"]))
        if rec.get("is_correct_type") is True:
            circuit_correct_types.append(1.0)
        else:
            circuit_correct_types.append(0.0)

        if rec.get("failure_category") == "budget_overrun" or (rec.get("truncated", False) and not rec.get("submitted", False)):
            budget_overruns += 1
        MemoryGuard.cleanup_memory()

    mean_nrmse = round(float(sum(circuit_nrmses) / len(circuit_nrmses)), 4) if circuit_nrmses else None
    report["benchmarks"]["resonance_active_circuit"] = {
        "mean_reward": round(float(sum(circuit_rewards) / len(circuit_rewards)), 4),
        "mean_nrmse": mean_nrmse,
        "n_submitted_nrmse": len(circuit_nrmses),
        "type_accuracy": round(float(sum(circuit_correct_types) / len(circuit_correct_types)), 4),
        "budget_overrun_rate": round(float(budget_overruns / len(circuit_rewards)), 4),
        "episodes": circuit_episodes,
        "mode": "exploratory_smoke_5000_series",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    nrmse_display = f"{mean_nrmse:.4f} (n={len(circuit_nrmses)})" if mean_nrmse is not None else "None (no submissions)"
    print(f"  -> Resonance Circuit Reward: {report['benchmarks']['resonance_active_circuit']['mean_reward']:.4f} | NRMSE: {nrmse_display} | Type Acc: {report['benchmarks']['resonance_active_circuit']['type_accuracy']*100:.1f}%")

    # Free model memory before next run
    del agent
    MemoryGuard.cleanup_memory()

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate All Physics Benchmarks (Base vs Exploratory LoRA Adapter)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rl_model", type=str, default="checkpoints/resonance_qwen_rl")
    parser.add_argument("--circuit_episodes", type=int, default=5)
    parser.add_argument("--boxing_episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="artifacts/all_benchmarks_comparison.json")
    args = parser.parse_args()

    print("==================================================================")
    print("=== Multi-Benchmark Comprehensive Physics Evaluation Harness ===")
    print("==================================================================")

    # 1. Evaluate Base Model (Before RL)
    base_results = evaluate_model_on_all_benchmarks(
        model_path=args.base_model,
        label="Qwen2.5-0.5B (Before RL / Base)",
        circuit_episodes=args.circuit_episodes,
        boxing_gym_episodes=args.boxing_episodes,
    )

    # 2. Evaluate RL Model (After RL)
    rl_results = evaluate_model_on_all_benchmarks(
        model_path=args.rl_model,
        label="Qwen2.5-0.5B (After Resonance RL)",
        circuit_episodes=args.circuit_episodes,
        boxing_gym_episodes=args.boxing_episodes,
    )

    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model": base_results,
        "rl_model": rl_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nSaved full benchmark comparison to {args.output}!")

    # Print Clean Master Markdown Table with Honest Labeling (R6)
    print("\n" + "="*80)
    print("MASTER BENCHMARK COMPARISON TABLE: BEFORE RL vs AFTER RL")
    def format_delta_finding(delta: float, higher_is_better: bool = True) -> str:
        d = delta if higher_is_better else -delta
        if abs(delta) < 1e-4:
            return "Neutral / No change"
        elif d > 0.05:
            return "Empirical positive gain"
        elif d > 0:
            return "Slight positive gain"
        elif d < -0.05:
            return "Empirical negative transfer"
        else:
            return "Slight negative shift"

    print("="*80)
    print("| Benchmark Suite & Provenance | Evaluation Metric | Zero-Shot Base | Shared Adapter (Pass B) | Absolute Delta | Transfer Finding |")
    print("| :--- | :--- | :---: | :---: | :---: | :--- |")

    b_ug = base_results["benchmarks"]["ugphysics"]
    r_ug = rl_results["benchmarks"]["ugphysics"]
    d_ug = r_ug['accuracy'] - b_ug['accuracy']
    print(f"| **In-repo Suite (inspired by UGPhysics, n=15)** | Multiple-Choice Acc | {b_ug['accuracy']*100:.1f}% | {r_ug['accuracy']*100:.1f}% | {d_ug*100:+.1f}% | {format_delta_finding(d_ug)} |")

    b_pr = base_results["benchmarks"]["phys_reason"]
    r_pr = rl_results["benchmarks"]["phys_reason"]
    d_pr = r_pr['accuracy'] - b_pr['accuracy']
    print(f"| **In-repo Suite (inspired by PhysReason, n=10)** | Multi-Step Reasoning Acc | {b_pr['accuracy']*100:.1f}% | {r_pr['accuracy']*100:.1f}% | {d_pr*100:+.1f}% | {format_delta_finding(d_pr)} |")

    b_pv = base_results["benchmarks"]["physics_verified"]
    r_pv = rl_results["benchmarks"]["physics_verified"]
    d_pv = r_pv['accuracy'] - b_pv['accuracy']
    print(f"| **In-repo Suite (inspired by PHYSICS-Verified, n=8)** | Objective Numerical Acc | {b_pv['accuracy']*100:.1f}% | {r_pv['accuracy']*100:.1f}% | {d_pv*100:+.1f}% | {format_delta_finding(d_pv)} |")

    b_bg = base_results["benchmarks"]["boxing_gym"]
    r_bg = rl_results["benchmarks"]["boxing_gym"]
    d_bg = r_bg['mean_nrmse'] - b_bg['mean_nrmse']
    print(f"| **Active System ID (inspired by BoxingGym, n=3)** | Mean Parameter NRMSE | {b_bg['mean_nrmse']:.4f} | {r_bg['mean_nrmse']:.4f} | {d_bg:+.4f} | {format_delta_finding(d_bg, higher_is_better=False)} |")

    b_sr = base_results["benchmarks"]["llm_srbench"]
    r_sr = rl_results["benchmarks"]["llm_srbench"]
    d_sr = r_sr['accuracy'] - b_sr['accuracy']
    print(f"| **Law-Family Identification (inspired by LLM-SRBench, n=6)** | Law-Family Classification Acc | {b_sr['accuracy']*100:.1f}% | {r_sr['accuracy']*100:.1f}% | {d_sr*100:+.1f}% | {format_delta_finding(d_sr)} |")

    b_rc = base_results["benchmarks"]["resonance_active_circuit"]
    r_rc = rl_results["benchmarks"]["resonance_active_circuit"]
    d_rc_rew = r_rc['mean_reward'] - b_rc['mean_reward']
    d_rc_acc = r_rc['type_accuracy'] - b_rc['type_accuracy']
    d_rc_ovr = r_rc['budget_overrun_rate'] - b_rc['budget_overrun_rate']
    print(f"| **Resonance Active Circuit (Main Task, n={args.circuit_episodes})** | Mean Active RL Reward | {b_rc['mean_reward']:.4f} | {r_rc['mean_reward']:.4f} | {d_rc_rew:+.4f} | {format_delta_finding(d_rc_rew)} |")
    print(f"| **Resonance Active Circuit (Main Task, n={args.circuit_episodes})** | Type Discrimination Acc | {b_rc['type_accuracy']*100:.1f}% | {r_rc['type_accuracy']*100:.1f}% | {d_rc_acc*100:+.1f}% | {format_delta_finding(d_rc_acc)} |")
    print(f"| **Resonance Active Circuit (Main Task, n={args.circuit_episodes})** | Budget Overrun Rate | {b_rc['budget_overrun_rate']*100:.1f}% | {r_rc['budget_overrun_rate']*100:.1f}% | {d_rc_ovr*100:+.1f}% | {format_delta_finding(d_rc_ovr, higher_is_better=False)} |")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
