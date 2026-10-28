"""Run physics reasoning benchmark evaluation on an open-source model.

Evaluates problem solving across:
- AC Circuits & Resonance
- Optics & Malus' Law
- Classical Mechanics & Oscillations
- Thermodynamics & Carnot cycle
- Modern & Quantum Physics
"""

from __future__ import annotations
import json
import os
import argparse
from resonance.benchmarks.phys_reasoning import (
    PHYSICS_REASONING_PROBLEMS,
    evaluate_physics_reasoning,
)
from resonance.models.prompted_llm import PromptedLLMAgent
from resonance.core.memory import MemoryGuard


def main():
    parser = argparse.ArgumentParser(description="Run Physics Reasoning Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model name or path")
    parser.add_argument("--output", type=str, default="artifacts/phys_reasoning_results.json", help="Output path")
    args = parser.parse_args()

    mem_init = MemoryGuard.get_system_memory_info()
    print(f"=== Resonance Physics Reasoning Benchmark ===")
    print(f"Model: {args.model}")
    print(f"[Memory Guard] System RAM: {mem_init['total_ram_mb']} MB, Available: {mem_init['available_ram_mb']} MB ({mem_init['percent_used']}% used)")

    agent = PromptedLLMAgent(model_name_or_path=args.model)
    res = evaluate_physics_reasoning(agent, PHYSICS_REASONING_PROBLEMS)
    MemoryGuard.cleanup_memory()

    print(f"\nResults Summary:")
    print(f"Total Problems: {res['total_problems']}")
    print(f"Correct:        {res['correct']}")
    print(f"Accuracy:       {res['accuracy']*100:.1f}%\n")

    print(f"Domain Breakdown:")
    for domain, stats in res["domain_breakdown"].items():
        print(f"  - {domain:<20}: {stats['correct']}/{stats['total']} ({stats['accuracy']*100:.1f}%)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
