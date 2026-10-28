"""Tests for Gymnasium environment adapter."""

import unittest
import numpy as np
import gymnasium as gym
from resonance.envs.gym_adapter import ResonanceCircuitGymEnv, register_gym_envs


class TestGymAdapter(unittest.TestCase):

    def test_gym_interface(self):
        """Check reset, step, observation spaces, action spaces."""
        env = ResonanceCircuitGymEnv()
        obs, info = env.reset(seed=42)

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (5,))
        self.assertIn("status", info)
        self.assertEqual(env.action_space.n, 51)  # 50 frequencies + 1 submit action

        # Step discrete measurement action
        action = 10
        obs, rew, term, trunc, info = env.step(action)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (5,))
        self.assertIsInstance(rew, float)
        self.assertFalse(term)
        self.assertFalse(trunc)

        # Step discrete submit action (index 50)
        submit_action = 50
        obs, rew, term, trunc, info = env.step(submit_action)
        self.assertTrue(term)
        self.assertIn("reward", info)
        self.assertGreater(rew, 0.0)

    def test_gym_registration(self):
        """Check Gymnasium env registration."""
        register_gym_envs()
        env = gym.make("resonance/CircuitSpectroscopy-v0")
        obs, info = env.reset(seed=10)
        self.assertEqual(obs.shape, (5,))
        env.close()


if __name__ == "__main__":
    unittest.main()
