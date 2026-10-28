"""Security, robustness, and fuzzing tests for action schemas."""

import unittest
from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.core.scoring import audit_action_payload


class TestSchemasAndSecurity(unittest.TestCase):

    def setUp(self):
        self.env = CircuitSpectroscopyEnv()
        self.env.reset(seed=999)

    def test_nan_and_infinity_rejection(self):
        """NaN and Infinity values must be audited and rejected immediately."""
        bad_payloads = [
            {"type": "measure", "frequency_hz": float("nan")},
            {"type": "measure", "frequency_hz": float("inf")},
            {"type": "measure", "frequency_hz": float("-inf")},
            {
                "type": "submit",
                "model_probs": {"RC": float("nan"), "RL": 0.0, "RLC": 1.0},
                "parameters": {"circuit_type": "RLC", "R": 50.0, "L_mH": 10.0, "C_uF": 1.0}
            },
        ]
        for payload in bad_payloads:
            is_clean, err = audit_action_payload(payload)
            self.assertFalse(is_clean, f"Payload should have been rejected: {payload}")

            # Send to env and verify clean error without crash
            obs, rew, term, trunc, info = self.env.step(payload)
            self.assertFalse(term)
            self.assertIn("error", info)

    def test_extreme_exponent_rejection(self):
        """Values with astronomical magnitude must be rejected."""
        payload = {"type": "measure", "frequency_hz": 1e30}
        is_clean, err = audit_action_payload(payload)
        self.assertFalse(is_clean)

    def test_non_dict_payload(self):
        """Passing non-dict payloads like strings or numbers must not crash the env."""
        for invalid_act in ["measure 1000", 12345, None, [1, 2, 3]]:
            obs, rew, term, trunc, info = self.env.step(invalid_act)
            self.assertIn("error", info)

    def test_cost_free_loops_prevented(self):
        """Submitting invalid schema actions repeatedly must consume turns and eventually truncate."""
        env = CircuitSpectroscopyEnv(max_turns=5)
        env.reset(seed=12)

        for _ in range(4):
            obs, rew, term, trunc, info = env.step({"junk": "data"})
            self.assertFalse(trunc)

        # 5th invalid action must trigger truncation
        obs, rew, term, trunc, info = env.step({"junk": "data"})
        self.assertTrue(trunc)


if __name__ == "__main__":
    unittest.main()
