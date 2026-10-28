"""Golden determinism and rejection sampling test suite (Section 2.3).

Verifies bit-identical integer logic, exact seed derivation, and tolerance-based
(1e-9 relative) floating-point agreement with committed golden worlds across platforms.
"""

from __future__ import annotations
import json
from pathlib import Path
import unittest
import numpy as np

from resonance.envs.circuit import CircuitSpectroscopyEnv, CircuitType
from resonance.physics.circuits import compute_resonant_frequency

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "circuit_worlds_20_golden.json"


class TestDeterminismGolden(unittest.TestCase):
    def test_20_golden_worlds_reproducibility(self):
        """Seeds 10000..10019 must reproduce identical circuit parameters to 1e-9 relative tolerance."""
        self.assertTrue(GOLDEN_PATH.is_file(), f"Golden file missing at {GOLDEN_PATH}")
        with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
            golden_data = json.load(f)

        self.assertEqual(len(golden_data), 20)
        env = CircuitSpectroscopyEnv()

        for g in golden_data:
            seed = g["seed"]
            obs, info = env.reset(seed=seed)
            hp = env._hidden_params

            # Exact equality on discrete types and seed logic
            self.assertEqual(hp["circuit_type"], g["circuit_type"], f"Type mismatch on seed {seed}")
            self.assertEqual(info["seed"], seed)

            # Tolerance-based (1e-9 relative) comparison on floating point values
            np.testing.assert_allclose(
                hp["R"],
                g["R"],
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"R mismatch on seed {seed}",
            )

            if g["L_mH"] is not None:
                self.assertIsNotNone(hp["L_mH"])
                np.testing.assert_allclose(
                    hp["L_mH"],
                    g["L_mH"],
                    rtol=1e-9,
                    atol=1e-9,
                    err_msg=f"L_mH mismatch on seed {seed}",
                )
            else:
                self.assertIsNone(hp["L_mH"])

            if g["C_uF"] is not None:
                self.assertIsNotNone(hp["C_uF"])
                np.testing.assert_allclose(
                    hp["C_uF"],
                    g["C_uF"],
                    rtol=1e-9,
                    atol=1e-9,
                    err_msg=f"C_uF mismatch on seed {seed}",
                )
            else:
                self.assertIsNone(hp["C_uF"])

            if g["f0"] is not None:
                _, f0 = compute_resonant_frequency(hp["L_H"], hp["C_F"])
                np.testing.assert_allclose(
                    f0,
                    g["f0"],
                    rtol=1e-9,
                    atol=1e-9,
                    err_msg=f"f0 mismatch on seed {seed}",
                )

    def test_rejection_sampling_acceptance_rates(self):
        """Measure acceptance rate per circuit family across 300 seed trials."""
        env = CircuitSpectroscopyEnv()
        counts = {"RC": 0, "RL": 0, "RLC": 0}
        total = 300
        for s in range(5000, 5000 + total):
            obs, info = env.reset(seed=s)
            ctype = env._hidden_params["circuit_type"]
            counts[ctype] += 1

        # All 3 circuit families must be sampled with realistic probabilities (approx 1/3 each)
        for ctype, count in counts.items():
            rate = count / total
            self.assertGreater(rate, 0.15, f"Acceptance rate for {ctype} ({rate:.2f}) was unexpectedly low")
            self.assertLess(rate, 0.50, f"Acceptance rate for {ctype} ({rate:.2f}) was unexpectedly high")

    def test_rejection_sampling_raises_on_impossible_bounds(self):
        """Env must raise RuntimeError if 100 rejection sampling attempts fail."""
        # Force impossible RLC parameters where max_R <= R_MIN
        class ImpossibleRLCEnv(CircuitSpectroscopyEnv):
            def __init__(self):
                super().__init__(allowed_types=[CircuitType.RLC])
                self.R_MIN = 190.0
                self.R_MAX = 200.0
                # Make L_H and C_F bounds force z0 / 1.5 < 190
                self.L_MH_MIN = 1.0
                self.L_MH_MAX = 2.0
                self.C_UF_MIN = 8.0
                self.C_UF_MAX = 10.0

        env = ImpossibleRLCEnv()
        with self.assertRaises(RuntimeError):
            env.reset(seed=12345)


if __name__ == "__main__":
    unittest.main()
