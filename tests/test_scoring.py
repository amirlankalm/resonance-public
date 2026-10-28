"""Tests for scoring, complex RMSE, Brier score, and reward formulas."""

import unittest
import numpy as np
from resonance.core.scoring import (
    compute_complex_response_rmse,
    compute_nrmse,
    compute_brier_score,
    compute_terminal_reward,
)


class TestScoring(unittest.TestCase):

    def test_complex_rmse_avoids_phase_wrap(self):
        """Phasors near -pi and +pi must not suffer from artificial 2pi jump error."""
        # Consider two phasors with same magnitude: one at +179 deg, one at -179 deg (diff is only 2 deg)
        theta1 = np.deg2rad(179.0)
        theta2 = np.deg2rad(-179.0)

        I1 = np.array([np.cos(theta1) + 1j * np.sin(theta1)])
        I2 = np.array([np.cos(theta2) + 1j * np.sin(theta2)])

        rmse = compute_complex_response_rmse(I1, I2)
        # 2 * sin(1 deg) ~ 2 * 0.01745 ~ 0.0349
        self.assertAlmostEqual(rmse, 2.0 * np.sin(np.deg2rad(1.0)), places=4)

    def test_brier_score(self):
        """Brier score is 0 for perfect prediction, 2 for worst possible prediction."""
        # Perfect
        probs_perfect = {"RC": 0.0, "RL": 1.0, "RLC": 0.0}
        self.assertAlmostEqual(compute_brier_score(probs_perfect, "RL"), 0.0)

        # Completely wrong
        probs_wrong = {"RC": 1.0, "RL": 0.0, "RLC": 0.0}
        # (1 - 0)^2 + (0 - 1)^2 = 2.0
        self.assertAlmostEqual(compute_brier_score(probs_wrong, "RL"), 2.0)

        # Uniform 3-class: (1/3 - 1)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9 ~ 0.6667
        probs_uniform = {"RC": 1/3, "RL": 1/3, "RLC": 1/3}
        self.assertAlmostEqual(compute_brier_score(probs_uniform, "RL"), 2/3, places=4)

    def test_terminal_reward_accounting(self):
        """Reward v2: R = 0.6*(1 - e) + 0.4*(1 - brier/2) - 0.005*q, capped at 0.20 if q == 0."""
        # e = 0, q = 0 with cap -> capped at zero_measurement_cap (0.20)
        self.assertAlmostEqual(compute_terminal_reward(0.0, 0, zero_measurement_cap=0.20), 0.20)

        # e = 0, q = 0 without cap -> 1.0
        self.assertAlmostEqual(compute_terminal_reward(0.0, 0), 1.0)

        # e = 0.05, q = 6, brier = None -> (1 - 0.05) - 0.005 * 6 = 0.95 - 0.03 = 0.92
        self.assertAlmostEqual(compute_terminal_reward(0.05, 6), 0.92)

        # e = 0.0, q = 1, brier = 0.0 -> 0.6*(1.0) + 0.4*(1.0) - 0.005 = 0.995
        self.assertAlmostEqual(compute_terminal_reward(0.0, 1, brier_score=0.0), 0.995)

        # e = 1.5, q = 2 -> clip(1 - 1.5, 0, 1) - 0.01 = 0 - 0.01 = -0.01
        self.assertAlmostEqual(compute_terminal_reward(1.5, 2), -0.01)


if __name__ == "__main__":
    unittest.main()
