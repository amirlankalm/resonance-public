"""Tests for Circuit Spectroscopy environment interaction, isolation, and budgets."""

import unittest
import numpy as np
from resonance.envs.circuit import CircuitSpectroscopyEnv


class TestCircuitEnv(unittest.TestCase):

    def test_reproducible_reset(self):
        """Seeded reset must generate identical circuits."""
        env1 = CircuitSpectroscopyEnv()
        env2 = CircuitSpectroscopyEnv()

        obs1, _ = env1.reset(seed=12345)
        obs2, _ = env2.reset(seed=12345)

        self.assertEqual(env1._hidden_params["circuit_type"], env2._hidden_params["circuit_type"])
        self.assertEqual(env1._hidden_params["R"], env2._hidden_params["R"])

    def test_state_isolation(self):
        """Public observation must not leak hidden parameters."""
        env = CircuitSpectroscopyEnv()
        obs, _ = env.reset(seed=42)

        for secret in ["_hidden_params", "R", "L_mH", "C_uF"]:
            self.assertNotIn(secret, obs)

    def test_turn_and_budget_accounting(self):
        """Measurement calls consume both turn and budget; invalid calls consume only turn."""
        env = CircuitSpectroscopyEnv(measurement_budget=12, max_turns=20)
        env.reset(seed=1)

        # 1. Invalid call consumes turn, but not measurement budget
        obs, rew, term, trunc, info = env.step({"type": "measure", "frequency_hz": -50.0})
        self.assertEqual(env.step_count, 1)
        self.assertEqual(env.instrument_calls, 0)
        self.assertEqual(obs["remaining_calls"], 12)
        self.assertEqual(obs["remaining_turns"], 19)

        # 2. Valid measurement call consumes both turn and budget
        obs, rew, term, trunc, info = env.step({"type": "measure", "frequency_hz": 1000.0})
        self.assertEqual(env.step_count, 2)
        self.assertEqual(env.instrument_calls, 1)
        self.assertEqual(obs["remaining_calls"], 11)
        self.assertEqual(obs["remaining_turns"], 18)

    def test_budget_exhaustion(self):
        """Cannot measure more than 12 times."""
        env = CircuitSpectroscopyEnv(measurement_budget=12)
        env.reset(seed=2)

        for _ in range(12):
            env.step({"type": "measure", "frequency_hz": 500.0})

        # 13th call should fail
        obs, rew, term, trunc, info = env.step({"type": "measure", "frequency_hz": 500.0})
        self.assertIn("error", info)
        self.assertEqual(env.instrument_calls, 12)

    def test_circuit_submission_and_scoring(self):
        """Submitting parameters evaluates complex RMSE and returns reward."""
        env = CircuitSpectroscopyEnv()
        env.reset(seed=3)

        true_type = env._hidden_params["circuit_type"]
        true_R = env._hidden_params["R"]
        true_L = env._hidden_params["L_mH"]
        true_C = env._hidden_params["C_uF"]

        # Take a valid measurement first
        env.step({"type": "measure", "frequency_hz": 1000.0})

        # Submit ground truth
        probs = {"RC": 0.0, "RL": 0.0, "RLC": 0.0}
        probs[true_type] = 1.0

        sub_action = {
            "type": "submit",
            "model_probs": probs,
            "parameters": {
                "circuit_type": true_type,
                "R": true_R,
                "L_mH": true_L,
                "C_uF": true_C,
            }
        }
        obs, reward, term, trunc, info = env.step(sub_action)

        self.assertTrue(term)
        self.assertTrue(info["is_correct_type"])
        self.assertAlmostEqual(info["rmse"], 0.0, places=6)
        self.assertAlmostEqual(info["brier_score"], 0.0, places=5)
        # 1 call made -> 1.0 - 0.005 = 0.995
        self.assertAlmostEqual(reward, 0.995, places=4)

    def test_rlc_worlds_have_sharp_resonance(self):
        """RLC worlds must strictly satisfy Q >= 1.5 and observable f0 across 1000 seeds."""
        for seed in range(1000):
            env = CircuitSpectroscopyEnv()
            env.reset(seed=seed)
            p = env._hidden_params
            if p["circuit_type"].lower() == "rlc":
                Q = np.sqrt(p["L_H"] / p["C_F"]) / p["R"]
                self.assertGreaterEqual(Q, 1.5 - 1e-9, f"seed {seed}: Q={Q:.3f}")
                f0 = 1.0 / (2.0 * np.pi * np.sqrt(p["L_H"] * p["C_F"]))
                self.assertTrue(50.0 <= f0 <= 25000.0, f"seed {seed}: f0={f0:.1f}")


if __name__ == "__main__":
    unittest.main()
