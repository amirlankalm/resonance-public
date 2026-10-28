"""Polarization (Malus' Law) starter environment for Resonance."""

from __future__ import annotations
import copy
from typing import Dict, Any, Tuple, Optional
import numpy as np

from resonance.envs.base import BasePhysicsEnv
from resonance.physics.optics import malus_law, generate_polarization_eval_grid
from resonance.core.rng import IsolatedRNGManager
from resonance.core.schemas import (
    MeasureAction,
    SubmitPolarizationAction,
    MeasurementRecord,
)
from resonance.core.scoring import (
    compute_nrmse,
    compute_terminal_reward,
)


class PolarizationEnv(BasePhysicsEnv):
    """Malus' Law starter environment:

    Intensity y = b + A * cos^2(theta - phi) + N(0, sigma^2)
    where:
        A in [0.4, 1.0] (hidden amplitude)
        b in [0, 0.15] (hidden offset)
        phi in [0, 180) degrees (hidden polarization angle)
        sigma = 0.025 default detector noise standard deviation
        theta in [0, 180] degrees (agent chosen measurement angle)
    """

    SCALE_FACTOR = 1.15  # Normalization denominator for NRMSE

    def __init__(
        self,
        noise_std: float = 0.025,
        measurement_budget: int = 12,
        max_turns: int = 20,
        query_cost: float = 0.005,
    ):
        super().__init__(
            measurement_budget=measurement_budget,
            max_turns=max_turns,
            query_cost=query_cost,
        )
        self.noise_std = float(noise_std)
        self._eval_grid = generate_polarization_eval_grid(64)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.step_count = 0
        self.instrument_calls = 0
        self.is_terminated = False
        self.is_truncated = False
        self.history = []

        self.rng_mgr = IsolatedRNGManager(seed)

        # Sample hidden true world parameters from world RNG stream
        A_true = float(self.rng_mgr.rng_world.uniform(0.4, 1.0))
        b_true = float(self.rng_mgr.rng_world.uniform(0.0, 0.15))
        phi_true = float(self.rng_mgr.rng_world.uniform(0.0, 180.0))

        self._hidden_params = {
            "amplitude": A_true,
            "offset": b_true,
            "phi_deg": phi_true,
        }

        obs = self.get_public_observation()
        info = {
            "step": 0,
            "calls_made": 0,
            "status": "ready",
            "seed": self.rng_mgr.master_seed,
            "world_seed": self.rng_mgr.world_seed,
            "sensor_seed": self.rng_mgr.sensor_seed,
        }
        return obs, info

    def get_public_observation(self) -> Dict[str, Any]:
        """Return public observation. Strictly isolated from _hidden_params."""
        return {
            "task_name": "polarization_malus_law",
            "task_version": "1.0.0",
            "candidate_formulas": {
                "malus": "y = b + A * cos^2(theta - phi)"
            },
            "units": {
                "angle": "degrees [0, 180]",
                "intensity": "normalized detector units"
            },
            "bounds": {
                "amplitude": [0.4, 1.0],
                "offset": [0.0, 0.15],
                "phi_deg": [0.0, 180.0]
            },
            "noise_specification": {
                "noise_std": self.noise_std,
                "distribution": "Gaussian"
            },
            "history": copy.deepcopy(self.history),
            "remaining_calls": max(0, self.measurement_budget - self.instrument_calls),
            "remaining_turns": max(0, self.max_turns - self.step_count),
            "output_schema": {
                "type": "submit",
                "amplitude": "float",
                "offset": "float",
                "phi_deg": "float"
            }
        }

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.is_terminated or self.is_truncated:
            return (
                self.get_public_observation(),
                0.0,
                self.is_terminated,
                self.is_truncated,
                {"error": "Episode is already finished. Call reset()."}
            )

        self.step_count += 1
        truncated = self.step_count >= self.max_turns
        self.is_truncated = truncated

        # Audit action payload
        is_clean, err_msg = self.validate_action_security(action)
        if not is_clean:
            return (
                self.get_public_observation(),
                0.0,
                False,
                truncated,
                {"error": f"Invalid payload: {err_msg}", "is_valid": False}
            )

        action_type = action.get("type") if isinstance(action, dict) else None

        # MEASURE ACTION
        if action_type == "measure":
            if self.instrument_calls >= self.measurement_budget:
                return (
                    self.get_public_observation(),
                    0.0,
                    False,
                    truncated,
                    {"error": "Measurement budget exhausted (max 12 calls reached).", "is_valid": False}
                )

            try:
                m_action = MeasureAction(**action)
            except Exception as e:
                return (
                    self.get_public_observation(),
                    0.0,
                    False,
                    truncated,
                    {"error": f"Schema error: {e}", "is_valid": False}
                )

            theta = m_action.angle_deg
            if theta is None or theta < 0.0 or theta > 180.0:
                return (
                    self.get_public_observation(),
                    0.0,
                    False,
                    truncated,
                    {"error": f"Angle {theta} out of bounds [0, 180].", "is_valid": False}
                )

            # Consume budget
            self.instrument_calls += 1

            # Compute true intensity and add sensor noise from sensor RNG stream
            true_y = malus_law(
                theta_deg=theta,
                amplitude=self._hidden_params["amplitude"],
                offset=self._hidden_params["offset"],
                phi_deg=self._hidden_params["phi_deg"],
            )
            noise = float(self.rng_mgr.rng_sensor.normal(0.0, self.noise_std))
            reading_y = float(true_y + noise)

            rec = {
                "step": self.step_count,
                "query_input": {"angle_deg": float(theta)},
                "reading": {"intensity": reading_y},
                "noise_sigma": {"intensity": self.noise_std},
            }
            self.history.append(rec)

            return (
                self.get_public_observation(),
                0.0,
                False,
                truncated,
                {"status": "measured", "reading": reading_y, "calls_made": self.instrument_calls}
            )

        # SUBMIT ACTION
        elif action_type == "submit":
            try:
                sub_action = SubmitPolarizationAction(**action)
            except Exception as e:
                return (
                    self.get_public_observation(),
                    0.0,
                    False,
                    truncated,
                    {"error": f"Schema error: {e}", "is_valid": False}
                )

            self.is_terminated = True

            # Evaluate on 64 noise-free hidden points
            eval_angles = self._eval_grid
            true_vals = malus_law(
                theta_deg=eval_angles,
                amplitude=self._hidden_params["amplitude"],
                offset=self._hidden_params["offset"],
                phi_deg=self._hidden_params["phi_deg"],
            )
            pred_vals = malus_law(
                theta_deg=eval_angles,
                amplitude=sub_action.amplitude,
                offset=sub_action.offset,
                phi_deg=sub_action.phi_deg,
            )

            rmse = float(np.sqrt(np.mean((pred_vals - true_vals) ** 2)))
            nrmse = compute_nrmse(rmse, self.SCALE_FACTOR)
            reward = compute_terminal_reward(nrmse, self.instrument_calls, self.query_cost)

            info = {
                "status": "terminated",
                "rmse": rmse,
                "nrmse": nrmse,
                "reward": reward,
                "calls_made": self.instrument_calls,
                "steps": self.step_count,
                "predicted": {
                    "amplitude": sub_action.amplitude,
                    "offset": sub_action.offset,
                    "phi_deg": sub_action.phi_deg,
                },
            }

            return self.get_public_observation(), reward, True, False, info

        else:
            return (
                self.get_public_observation(),
                0.0,
                False,
                truncated,
                {"error": f"Unknown action type: {action_type}. Expected 'measure' or 'submit'.", "is_valid": False}
            )
