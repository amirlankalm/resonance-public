"""Tests for numerical baselines and adaptive experimental design."""

import unittest
from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.baselines.numerical import (
    run_prior_only_baseline,
    run_random_design_baseline,
    run_fixed_sweep_baseline,
    run_adaptive_design_baseline,
)


class TestBaselines(unittest.TestCase):

    def test_prior_only(self):
        env = CircuitSpectroscopyEnv()
        info = run_prior_only_baseline(env)
        self.assertEqual(info["calls_made"], 0)
        self.assertIn("reward", info)

    def test_random_design(self):
        env = CircuitSpectroscopyEnv()
        info = run_random_design_baseline(env, num_measurements=4, seed=42)
        self.assertEqual(info["calls_made"], 4)
        self.assertIn("reward", info)
        self.assertIn("nrmse", info)

    def test_fixed_sweep(self):
        env = CircuitSpectroscopyEnv()
        info = run_fixed_sweep_baseline(env, num_measurements=4, seed=42)
        self.assertEqual(info["calls_made"], 4)
        self.assertIn("reward", info)

    def test_adaptive_design(self):
        env = CircuitSpectroscopyEnv()
        info = run_adaptive_design_baseline(env, max_measurements=4, seed=42)
        self.assertEqual(info["calls_made"], 4)
        self.assertIn("reward", info)


if __name__ == "__main__":
    unittest.main()
