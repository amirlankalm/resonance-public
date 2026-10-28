"""Unit tests for extended physics benchmarks: UGPhysics, PhysReason, PHYSICS-Verified, BoxingGym, LLM-SRBench."""

import unittest
from resonance.benchmarks.ugphysics import UGPHYSICS_PROBLEMS, evaluate_ugphysics, extract_multiple_choice_option
from resonance.benchmarks.phys_reason import PHYSREASON_PROBLEMS, evaluate_physreason, extract_numerical_answer
from resonance.benchmarks.physics_verified import PHYSICS_VERIFIED_PROBLEMS, evaluate_physics_verified
from resonance.benchmarks.boxing_gym import BOXING_GYM_SYSTEMS, evaluate_boxing_gym
from resonance.benchmarks.llm_srbench import SR_PROBLEMS, evaluate_llm_srbench, extract_sr_option


class MockBenchmarkAgent:
    """Mock agent returning predefined or dynamic responses for unit testing."""
    def __init__(self, mode: str = "oracle"):
        self.mode = mode

    def generate_response(self, prompt: str, system_prompt=None, max_new_tokens=None) -> str:
        if self.mode == "oracle":
            # Check for multiple choice
            for opt in ["A", "B", "C", "D"]:
                if f"Answer with the correct letter option." in prompt:
                    return f"Based on physical derivation, the answer is ({opt})."
            # Default numerical
            return "Calculation gives Answer: 10.0"
        return "Uncertain response without clear answer."


class TestExtendedBenchmarks(unittest.TestCase):
    def test_ugphysics_extraction(self):
        self.assertEqual(extract_multiple_choice_option("Therefore the correct choice is (B)."), "B")
        self.assertEqual(extract_multiple_choice_option("Answer: C"), "C")
        self.assertEqual(extract_multiple_choice_option("Option A is clearly the right result."), "A")

    def test_physreason_extraction(self):
        self.assertEqual(extract_numerical_answer("After step 4, the final Answer: 42.5"), 42.5)
        self.assertEqual(extract_numerical_answer(r"The result is \boxed{3.14}"), 3.14)

    def test_srbench_extraction(self):
        self.assertEqual(extract_sr_option("The governing equation is Option B"), "B")
        self.assertEqual(extract_sr_option("Answer: D (Quartic Law)"), "D")

    def test_boxing_gym_system(self):
        system = BOXING_GYM_SYSTEMS[0]
        params, fn = system.sample_world(123)
        self.assertIn("m", params)
        val = fn(5.0)
        self.assertGreater(val, 0.0)

    def test_mock_evaluations(self):
        agent = MockBenchmarkAgent(mode="oracle")
        # Run 1 problem test on each to verify execution pipeline
        ug_res = evaluate_ugphysics(agent, problems=UGPHYSICS_PROBLEMS[:2])
        self.assertEqual(ug_res["total_problems"], 2)

        pr_res = evaluate_physreason(agent, problems=PHYSREASON_PROBLEMS[:2])
        self.assertEqual(pr_res["total_problems"], 2)

        pv_res = evaluate_physics_verified(agent, problems=PHYSICS_VERIFIED_PROBLEMS[:2])
        self.assertEqual(pv_res["total_problems"], 2)

        sr_res = evaluate_llm_srbench(agent, problems=SR_PROBLEMS[:2])
        self.assertEqual(sr_res["total_problems"], 2)

        bg_res = evaluate_boxing_gym(agent, num_episodes=2, budget_queries=2)
        self.assertEqual(bg_res["episodes_evaluated"], 2)


if __name__ == "__main__":
    unittest.main()
