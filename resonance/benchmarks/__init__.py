"""Benchmark suites for physics reasoning and active scientific experimentation."""

from resonance.benchmarks.phys_reasoning import (
    PHYSICS_REASONING_PROBLEMS,
    evaluate_physics_reasoning,
    extract_numerical_answer,
)
from resonance.benchmarks.active_benchmark import ActivePhysicsBenchmark
from resonance.benchmarks.ugphysics import UGPHYSICS_PROBLEMS, evaluate_ugphysics
from resonance.benchmarks.phys_reason import PHYSREASON_PROBLEMS, evaluate_physreason
from resonance.benchmarks.physics_verified import (
    PHYSICS_VERIFIED_PROBLEMS,
    evaluate_physics_verified,
)
from resonance.benchmarks.boxing_gym import BOXING_GYM_SYSTEMS, evaluate_boxing_gym
from resonance.benchmarks.llm_srbench import SR_PROBLEMS, evaluate_llm_srbench

__all__ = [
    "PHYSICS_REASONING_PROBLEMS",
    "evaluate_physics_reasoning",
    "extract_numerical_answer",
    "ActivePhysicsBenchmark",
    "UGPHYSICS_PROBLEMS",
    "evaluate_ugphysics",
    "PHYSREASON_PROBLEMS",
    "evaluate_physreason",
    "PHYSICS_VERIFIED_PROBLEMS",
    "evaluate_physics_verified",
    "BOXING_GYM_SYSTEMS",
    "evaluate_boxing_gym",
    "SR_PROBLEMS",
    "evaluate_llm_srbench",
]
