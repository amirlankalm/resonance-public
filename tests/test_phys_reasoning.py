"""Tests for physics reasoning benchmark questions and answer extractor."""

import unittest
from resonance.benchmarks.phys_reasoning import (
    PHYSICS_REASONING_PROBLEMS,
    extract_numerical_answer,
    evaluate_physics_reasoning,
)


class TestPhysReasoning(unittest.TestCase):

    def test_extract_numerical_answer(self):
        """Test parsing of \\boxed{...}, Final Answer, and trailing numbers."""
        case1 = "Therefore the cyclic resonant frequency is \\boxed{796} Hz."
        self.assertEqual(extract_numerical_answer(case1), 796.0)

        case2 = "Calculations yield 2.5. Final Answer: 2.5"
        self.assertEqual(extract_numerical_answer(case2), 2.5)

        case3 = "The result is 314.16"
        self.assertEqual(extract_numerical_answer(case3), 314.16)

    def test_evaluator_with_oracle(self):
        """An oracle solver should achieve 100% accuracy on the benchmark."""
        res = evaluate_physics_reasoning(agent_or_model=None)
        self.assertEqual(res["total_problems"], len(PHYSICS_REASONING_PROBLEMS))
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["correct"], len(PHYSICS_REASONING_PROBLEMS))


if __name__ == "__main__":
    unittest.main()
