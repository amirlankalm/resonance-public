"""Gymnasium standard environment adapter for Resonance."""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.envs.polarization import PolarizationEnv
from resonance.baselines.numerical import evaluate_candidate_models
from resonance.physics.optics import malus_law
from scipy.optimize import minimize


class ResonanceCircuitGymEnv(gym.Env):
    """Standard Gymnasium wrapper for CircuitSpectroscopyEnv.

    Action space:
        Discrete(num_frequency_bins + 1):
        - 0 to (num_frequency_bins - 1): Measure at corresponding log-spaced probe frequency.
        - num_frequency_bins: Fit candidate models to history via least-squares and submit final hypothesis.

    Observation space:
        Box containing normalized history:
        [last_freq_norm, last_amplitude_norm, last_phase_norm, remaining_calls_norm, remaining_turns_norm]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        measurement_budget: int = 12,
        max_turns: int = 20,
        num_frequency_bins: int = 50,
    ):
        super().__init__()
        self.underlying_env = CircuitSpectroscopyEnv(
            measurement_budget=measurement_budget,
            max_turns=max_turns,
        )
        self.num_frequency_bins = num_frequency_bins
        self.submit_action_index = num_frequency_bins
        self.freq_grid = np.geomspace(
            self.underlying_env.BAND_MIN_HZ,
            self.underlying_env.BAND_MAX_HZ,
            num_frequency_bins,
        )

        # Action: pick an index from frequency grid to measure (0..49) or submit (50)
        self.action_space = spaces.Discrete(num_frequency_bins + 1)

        # Observation: [normalized_freq, normalized_amp, normalized_phase, rem_calls, rem_turns]
        # Finite bounds ensure compliance with gymnasium.utils.env_checker.check_env
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -2.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 3.0, 2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        public_obs, info = self.underlying_env.reset(seed=seed)
        vec_obs = self._vectorize_observation(public_obs)
        return vec_obs, info

    def step(
        self,
        action: int | Dict[str, Any],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if isinstance(action, (int, np.integer)):
            act_int = int(action)
            if act_int == self.submit_action_index:
                # Submit action: fit candidate models on accumulated history and submit best hypothesis
                fit_result = evaluate_candidate_models(
                    self.underlying_env.history,
                    V_rms=self.underlying_env.V_DRIVE_RMS,
                )
                act_dict = {
                    "type": "submit",
                    "model_probs": fit_result["model_probs"],
                    "parameters": fit_result["parameters"],
                }
            elif 0 <= act_int < self.num_frequency_bins:
                freq = float(self.freq_grid[act_int])
                act_dict = {"type": "measure", "frequency_hz": freq}
            else:
                act_dict = {"type": "invalid", "raw_action": act_int}
        else:
            act_dict = action

        public_obs, reward, terminated, truncated, info = self.underlying_env.step(act_dict)
        vec_obs = self._vectorize_observation(public_obs)
        return vec_obs, float(reward), bool(terminated), bool(truncated), info

    def _vectorize_observation(self, public_obs: Dict[str, Any]) -> np.ndarray:
        history = public_obs.get("history", [])
        if history:
            last = history[-1]
            freq = last["query_input"]["frequency_hz"]
            amp = last["reading"]["current_amplitude_A"] or 0.0
            phase = last["reading"]["current_phase_rad"] or 0.0
            freq_norm = (np.log10(freq) - np.log10(self.underlying_env.BAND_MIN_HZ)) / (
                np.log10(self.underlying_env.BAND_MAX_HZ) - np.log10(self.underlying_env.BAND_MIN_HZ)
            )
            amp_norm = amp / self.underlying_env.I_SCALE
            phase_norm = phase / np.pi
        else:
            freq_norm = 0.0
            amp_norm = 0.0
            phase_norm = 0.0

        rem_calls = public_obs.get("remaining_calls", 0) / float(self.underlying_env.measurement_budget)
        rem_turns = public_obs.get("remaining_turns", 0) / float(self.underlying_env.max_turns)

        vec = np.array(
            [
                float(np.clip(freq_norm, 0.0, 1.0)),
                float(np.clip(amp_norm, 0.0, 3.0)),
                float(np.clip(phase_norm, -2.0, 2.0)),
                float(np.clip(rem_calls, 0.0, 1.0)),
                float(np.clip(rem_turns, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return vec


class ResonancePolarizationGymEnv(gym.Env):
    """Standard Gymnasium wrapper for PolarizationEnv."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        measurement_budget: int = 12,
        max_turns: int = 20,
        num_angle_bins: int = 18,
    ):
        super().__init__()
        self.underlying_env = PolarizationEnv(
            measurement_budget=measurement_budget,
            max_turns=max_turns,
        )
        self.num_angle_bins = num_angle_bins
        self.submit_action_index = num_angle_bins
        self.angle_grid = np.linspace(0.0, 170.0, num_angle_bins)

        self.action_space = spaces.Discrete(num_angle_bins + 1)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        public_obs, info = self.underlying_env.reset(seed=seed)
        vec_obs = self._vectorize_observation(public_obs)
        return vec_obs, info

    def step(
        self,
        action: int | Dict[str, Any],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if isinstance(action, (int, np.integer)):
            act_int = int(action)
            if act_int == self.submit_action_index:
                # Least-squares fit of polarization model
                history = self.underlying_env.history
                if history:
                    angles = np.array([h["query_input"]["theta_deg"] for h in history], dtype=float)
                    intensities = np.array([h["reading"]["intensity"] for h in history], dtype=float)

                    def loss(p):
                        A, b, phi = p
                        if A <= 0 or b < 0:
                            return 1e6
                        pred = malus_law(angles, A, b, phi)
                        return float(np.sum((pred - intensities) ** 2))

                    res = minimize(
                        loss,
                        x0=[0.7, 0.05, 45.0],
                        bounds=[(0.4, 1.0), (0.0, 0.15), (0.0, 180.0)],
                        method="L-BFGS-B",
                    )
                    act_dict = {
                        "type": "submit",
                        "parameters": {
                            "amplitude": float(res.x[0]),
                            "offset": float(res.x[1]),
                            "phi_deg": float(res.x[2]),
                        },
                    }
                else:
                    act_dict = {
                        "type": "submit",
                        "parameters": {"amplitude": 0.7, "offset": 0.075, "phi_deg": 90.0},
                    }
            elif 0 <= act_int < self.num_angle_bins:
                angle = float(self.angle_grid[act_int])
                act_dict = {"type": "measure", "theta_deg": angle}
            else:
                act_dict = {"type": "invalid", "raw_action": act_int}
        else:
            act_dict = action

        public_obs, reward, terminated, truncated, info = self.underlying_env.step(act_dict)
        vec_obs = self._vectorize_observation(public_obs)
        return vec_obs, float(reward), bool(terminated), bool(truncated), info

    def _vectorize_observation(self, public_obs: Dict[str, Any]) -> np.ndarray:
        history = public_obs.get("history", [])
        if history:
            last = history[-1]
            angle = float(last["query_input"].get("theta_deg", 0.0))
            intensity = float(last["reading"].get("intensity", 0.0))
            angle_norm = angle / 180.0
            intensity_norm = intensity / 1.15
        else:
            angle_norm = 0.0
            intensity_norm = 0.0

        rem_calls = public_obs.get("remaining_calls", 0) / float(self.underlying_env.measurement_budget)
        rem_turns = public_obs.get("remaining_turns", 0) / float(self.underlying_env.max_turns)

        return np.array(
            [
                float(np.clip(angle_norm, 0.0, 1.0)),
                float(np.clip(intensity_norm, 0.0, 2.0)),
                float(np.clip(rem_calls, 0.0, 1.0)),
                float(np.clip(rem_turns, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )


def register_gym_envs() -> None:
    """Register Resonance environments with Gymnasium registry."""
    registered = gym.envs.registry.keys()
    for env_id in ("Resonance/CircuitSpectroscopy-v0", "resonance/CircuitSpectroscopy-v0"):
        if env_id not in registered:
            gym.register(
                id=env_id,
                entry_point="resonance.envs.gym_adapter:ResonanceCircuitGymEnv",
                max_episode_steps=20,
            )
    for env_id in ("Resonance/Polarization-v0", "resonance/Polarization-v0"):
        if env_id not in registered:
            gym.register(
                id=env_id,
                entry_point="resonance.envs.gym_adapter:ResonancePolarizationGymEnv",
                max_episode_steps=20,
            )
