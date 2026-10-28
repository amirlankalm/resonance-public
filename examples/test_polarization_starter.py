"""Polarization starter reproduction test suite.

Covers the 8 local starter checks specified in Resonance Project Description:
1. Seeded reproducibility
2. No truth fields in public observations
3. Measurement-budget enforcement
4. Invalid angles
5. Terminal behavior
6. Noiseless exact-family fitting
7. Observation-copy isolation
8. Finite six-query baseline output
"""

import unittest
import numpy as np
from resonance.envs.polarization import PolarizationEnv
from resonance.physics.optics import malus_law


class TestPolarizationStarter(unittest.TestCase):

    def test_01_seeded_reproducibility(self):
        """Check that identical seeds yield identical hidden parameters and observations."""
        env1 = PolarizationEnv()
        obs1, _ = env1.reset(seed=42)
        env2 = PolarizationEnv()
        obs2, _ = env2.reset(seed=42)

        self.assertEqual(env1._hidden_params, env2._hidden_params)
        self.assertEqual(obs1["bounds"], obs2["bounds"])

    def test_02_no_truth_fields_in_public_observations(self):
        """Verify that public observation contains zero hidden truth variables."""
        env = PolarizationEnv()
        obs, _ = env.reset(seed=123)

        forbidden_keys = ["amplitude", "offset", "phi_deg", "_hidden_params"]
        for key in forbidden_keys:
            self.assertNotIn(key, obs)

    def test_03_measurement_budget_enforcement(self):
        """Verify that attempting more than 12 measurement calls is rejected."""
        env = PolarizationEnv(measurement_budget=12)
        obs, _ = env.reset(seed=10)

        for _ in range(12):
            obs, rew, term, trunc, info = env.step({"type": "measure", "angle_deg": 45.0})
            self.assertFalse(term)

        # 13th call must be blocked
        obs, rew, term, trunc, info = env.step({"type": "measure", "angle_deg": 45.0})
        self.assertIn("error", info)
        self.assertEqual(env.instrument_calls, 12)

    def test_04_invalid_angles(self):
        """Check that out-of-bounds angles are cleanly rejected without consuming budget."""
        env = PolarizationEnv()
        env.reset(seed=77)

        init_calls = env.instrument_calls
        obs, rew, term, trunc, info = env.step({"type": "measure", "angle_deg": -15.0})
        self.assertIn("error", info)
        self.assertEqual(env.instrument_calls, init_calls)

        obs, rew, term, trunc, info = env.step({"type": "measure", "angle_deg": 195.0})
        self.assertIn("error", info)
        self.assertEqual(env.instrument_calls, init_calls)

    def test_05_terminal_behavior(self):
        """Submit action terminates episode and computes terminal reward."""
        env = PolarizationEnv()
        env.reset(seed=99)

        obs, rew, term, trunc, info = env.step({
            "type": "submit",
            "amplitude": 0.5,
            "offset": 0.05,
            "phi_deg": 30.0,
        })
        self.assertTrue(term)
        self.assertIn("reward", info)
        self.assertIn("nrmse", info)

        # Further steps after termination must be rejected
        obs, rew2, term2, trunc2, info2 = env.step({"type": "measure", "angle_deg": 45.0})
        self.assertIn("error", info2)

    def test_06_noiseless_exact_family_fitting(self):
        """Submitting exact ground truth yields near 1.0 reward (minus query cost)."""
        env = PolarizationEnv(noise_std=0.0)
        env.reset(seed=123)

        true_A = env._hidden_params["amplitude"]
        true_b = env._hidden_params["offset"]
        true_phi = env._hidden_params["phi_deg"]

        obs, rew, term, trunc, info = env.step({
            "type": "submit",
            "amplitude": true_A,
            "offset": true_b,
            "phi_deg": true_phi,
        })
        self.assertTrue(term)
        self.assertAlmostEqual(info["nrmse"], 0.0, places=5)
        self.assertAlmostEqual(rew, 1.0, places=4)

    def test_07_observation_copy_isolation(self):
        """Modifying received observation history must not mutate environment internal state."""
        env = PolarizationEnv()
        obs, _ = env.reset(seed=55)

        env.step({"type": "measure", "angle_deg": 30.0})
        obs = env.get_public_observation()
        # Mutate local copy
        obs["history"].clear()

        # Re-fetch from env
        fresh_obs = env.get_public_observation()
        self.assertEqual(len(fresh_obs["history"]), 1)

    def test_08_finite_six_query_baseline_output(self):
        """Run six measurements and submit; verify finite outputs."""
        env = PolarizationEnv()
        env.reset(seed=88)

        angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
        for a in angles:
            env.step({"type": "measure", "angle_deg": a})

        obs, rew, term, trunc, info = env.step({
            "type": "submit",
            "amplitude": 0.7,
            "offset": 0.05,
            "phi_deg": 45.0,
        })
        self.assertTrue(term)
        self.assertTrue(np.isfinite(rew))
        self.assertTrue(np.isfinite(info["nrmse"]))
        self.assertEqual(info["calls_made"], 6)


if __name__ == "__main__":
    unittest.main()
