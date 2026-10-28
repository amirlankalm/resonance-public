"""Circuit Spectroscopy active experiment environment for Resonance."""

from __future__ import annotations
import copy
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from resonance.envs.base import BasePhysicsEnv
from resonance.physics.circuits import (
    CircuitType,
    compute_complex_current,
    compute_current_amplitude_and_phase,
    compute_resonant_frequency,
)
from resonance.core.rng import IsolatedRNGManager
from resonance.core.schemas import (
    MeasureAction,
    SubmitCircuitAction,
    MeasurementRecord,
)
from resonance.core.scoring import (
    compute_complex_response_rmse,
    compute_nrmse,
    compute_brier_score,
    compute_terminal_reward,
)


class CircuitSpectroscopyEnv(BasePhysicsEnv):
    """Substantive active physics environment: Circuit Spectroscopy.

    An unknown linear circuit (series RC, RL, or RLC) is driven in steady state.
    The agent chooses AC probe frequencies within the instrument band [30 Hz, 30 kHz],
    observes noisy current amplitude and phase, and subsequently classifies the circuit
    topology and infers its component parameters (R, L, C).
    """

    # Instrument constants
    V_DRIVE_RMS = 0.1          # 0.1 V RMS
    BAND_MIN_HZ = 30.0         # 30 Hz
    BAND_MAX_HZ = 30000.0      # 30 kHz
    R_MIN = 20.0               # 20 Ohms
    R_MAX = 200.0              # 200 Ohms
    L_MH_MIN = 1.0             # 1 mH
    L_MH_MAX = 100.0           # 100 mH
    C_UF_MIN = 0.1             # 0.1 uF
    C_UF_MAX = 10.0            # 10 uF

    # Fixed family normalization scale: V_drive / R_min = 0.1 / 20 = 0.005 A (5 mA)
    I_SCALE = V_DRIVE_RMS / R_MIN

    # Noise specifications
    AMPLITUDE_NOISE_STD = 5e-5  # 50 microamperes RMS
    PHASE_NOISE_STD = 0.03      # ~1.72 degrees (radians)
    SENSOR_DETECTION_FLOOR = 1e-5 # 10 microamperes RMS (phase masked below this)

    # Penalty for invalid action formatting / schema errors (R9)
    INVALID_ACTION_PENALTY = -0.05

    def __init__(
        self,
        measurement_budget: int = 12,
        max_turns: int = 20,
        query_cost: float = 0.005,
        allowed_types: Optional[List[CircuitType | str]] = None,
        submission_timing_bonus: float = 0.0,
    ):
        super().__init__(
            measurement_budget=measurement_budget,
            max_turns=max_turns,
            query_cost=query_cost,
        )
        self.submission_timing_bonus = float(submission_timing_bonus)
        if allowed_types is None:
            self.allowed_types = [CircuitType.RC, CircuitType.RL, CircuitType.RLC]
        else:
            self.allowed_types = [CircuitType(t) for t in allowed_types]

        # 64 log-spaced frequencies for held-out evaluation
        self._eval_grid = np.geomspace(self.BAND_MIN_HZ, self.BAND_MAX_HZ, 64)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.step_count = 0
        self.instrument_calls = 0
        self.is_terminated = False
        self.is_truncated = False
        self.history = []

        self.rng_mgr = IsolatedRNGManager(seed)

        # World generation loop with rejection sampling for unidentifiable edges
        max_attempts = 100
        for _ in range(max_attempts):
            ctype_idx = int(self.rng_mgr.rng_world.integers(0, len(self.allowed_types)))
            ctype = self.allowed_types[ctype_idx]

            R = float(self.rng_mgr.rng_world.uniform(self.R_MIN, self.R_MAX))
            L_mH = float(self.rng_mgr.rng_world.uniform(self.L_MH_MIN, self.L_MH_MAX)) if ctype in (CircuitType.RL, CircuitType.RLC) else None
            C_uF = float(self.rng_mgr.rng_world.uniform(self.C_UF_MIN, self.C_UF_MAX)) if ctype in (CircuitType.RC, CircuitType.RLC) else None

            # Check identifiability and ensure sharp resonance for active probing (N1)
            if ctype == CircuitType.RLC:
                assert L_mH is not None and C_uF is not None
                L_H = L_mH * 1e-3
                C_F = C_uF * 1e-6
                _, f0 = compute_resonant_frequency(L_H, C_F)
                # Resonant frequency must fall within observable band with margin
                if not (50.0 <= f0 <= 25000.0):
                    continue
                # Enforce substantive quality factor (Q >= 1.5) so resonance peak
                # is sharp and requires intelligent active probing rather than coarse sweeping.
                # If even R_MIN cannot reach Q >= 1.5 for this (L, C) corner, reject the
                # sample entirely instead of silently clamping.
                z0 = np.sqrt(L_H / C_F)
                max_R = min(self.R_MAX, float(z0 / 1.5))
                if max_R <= self.R_MIN:
                    continue
                R = float(self.rng_mgr.rng_world.uniform(self.R_MIN, max_R))

                # Invariant: final RLC world satisfies the sharp-resonance guarantee
                assert float(np.sqrt(L_H / C_F)) / R >= 1.5 - 1e-9

            break
        else:
            raise RuntimeError(
                "World generation failed after 100 attempts: could not sample an "
                "identifiable RLC world with Q >= 1.5 in the observable band."
            )

        self._hidden_params = {
            "circuit_type": ctype.value,
            "R": R,
            "L_mH": L_mH,
            "C_uF": C_uF,
            "L_H": L_mH * 1e-3 if L_mH is not None else None,
            "C_F": C_uF * 1e-6 if C_uF is not None else None,
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
        """Return public observation containing no hidden truth variables."""
        return {
            "task_name": "circuit_spectroscopy",
            "task_version": "1.0.0",
            "candidate_formulas": {
                "RC": "Z_RC = R - j / (omega * C)",
                "RL": "Z_RL = R + j * omega * L",
                "RLC": "Z_RLC = R + j * (omega * L - 1 / (omega * C))",
                "current": "I = V_rms / Z, omega = 2 * pi * f"
            },
            "units": {
                "frequency": "Hertz [30, 30000]",
                "current_amplitude": "Amperes RMS",
                "current_phase": "radians relative to drive [-pi, pi]",
                "resistance": "Ohms [20, 200]",
                "inductance": "millihenries [1, 100]",
                "capacitance": "microfarads [0.1, 10]",
                "drive_voltage": "0.1 V RMS"
            },
            "bounds": {
                "frequency_hz": [self.BAND_MIN_HZ, self.BAND_MAX_HZ],
                "R": [self.R_MIN, self.R_MAX],
                "L_mH": [self.L_MH_MIN, self.L_MH_MAX],
                "C_uF": [self.C_UF_MIN, self.C_UF_MAX],
            },
            "noise_specification": {
                "amplitude_noise_std_A": self.AMPLITUDE_NOISE_STD,
                "phase_noise_std_rad": self.PHASE_NOISE_STD,
                "detection_floor_A": self.SENSOR_DETECTION_FLOOR,
            },
            "history": copy.deepcopy(self.history),
            "remaining_calls": max(0, self.measurement_budget - self.instrument_calls),
            "remaining_turns": max(0, self.max_turns - self.step_count),
            "output_schema": {
                "type": "submit",
                "model_probs": {"RC": "float", "RL": "float", "RLC": "float"},
                "parameters": {
                    "circuit_type": "RC | RL | RLC",
                    "R": "float (Ohms)",
                    "L_mH": "float (mH, if RL/RLC)",
                    "C_uF": "float (uF, if RC/RLC)"
                }
            }
        }

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.is_terminated or self.is_truncated:
            return (
                self.get_public_observation(),
                0.0,
                self.is_terminated,
                self.is_truncated,
                {"error": "Episode is finished. Call reset()."}
            )

        self.step_count += 1
        truncated = self.step_count >= self.max_turns
        self.is_truncated = truncated

        # Security check
        is_clean, err_msg = self.validate_action_security(action)
        if not is_clean:
            return (
                self.get_public_observation(),
                self.INVALID_ACTION_PENALTY,
                False,
                truncated,
                {"error": f"Invalid payload: {err_msg}", "is_valid": False, "status": "invalid_format"}
            )

        if isinstance(action, str):
            from resonance.core.schemas import extract_and_parse_action
            parsed = extract_and_parse_action(action)
            if not parsed or not isinstance(parsed, dict):
                return (
                    self.get_public_observation(),
                    self.INVALID_ACTION_PENALTY,
                    False,
                    truncated,
                    {"error": "Failed to parse valid JSON action dictionary.", "is_valid": False, "status": "invalid_format"}
                )
            action = parsed
            # Re-check security on parsed dict
            is_clean, err_msg = self.validate_action_security(action)
            if not is_clean:
                return (
                    self.get_public_observation(),
                    self.INVALID_ACTION_PENALTY,
                    False,
                    truncated,
                    {"error": f"Invalid payload: {err_msg}", "is_valid": False, "status": "invalid_format"}
                )

        action_type = action.get("type") if isinstance(action, dict) else None

        # MEASURE ACTION
        if action_type == "measure":
            if self.instrument_calls >= self.measurement_budget:
                return (
                    self.get_public_observation(),
                    self.INVALID_ACTION_PENALTY,
                    False,
                    truncated,
                    {"error": "Measurement budget exhausted (max 12 calls).", "is_valid": False}
                )

            try:
                m_action = MeasureAction(**action)
            except Exception as e:
                return (
                    self.get_public_observation(),
                    self.INVALID_ACTION_PENALTY,
                    False,
                    truncated,
                    {"error": f"Schema error: {e}", "is_valid": False}
                )

            freq = m_action.frequency_hz
            if freq is None or freq < self.BAND_MIN_HZ or freq > self.BAND_MAX_HZ:
                return (
                    self.get_public_observation(),
                    self.INVALID_ACTION_PENALTY,
                    False,
                    truncated,
                    {"error": f"Frequency {freq} Hz out of instrument band [{self.BAND_MIN_HZ}, {self.BAND_MAX_HZ}].", "is_valid": False}
                )

            # Consume budget
            self.instrument_calls += 1

            # Compute true physics response
            true_amp, true_phase, above_floor = compute_current_amplitude_and_phase(
                circuit_type=self._hidden_params["circuit_type"],
                frequency_hz=freq,
                V_rms=self.V_DRIVE_RMS,
                R=self._hidden_params["R"],
                L_H=self._hidden_params["L_H"],
                C_F=self._hidden_params["C_F"],
                detection_floor=self.SENSOR_DETECTION_FLOOR,
            )

            # Add sensor noise from isolated sensor RNG
            noise_amp = float(self.rng_mgr.rng_sensor.normal(0.0, self.AMPLITUDE_NOISE_STD))
            meas_amp = max(0.0, float(true_amp + noise_amp))

            if meas_amp >= self.SENSOR_DETECTION_FLOOR and np.isfinite(true_phase):
                noise_phase = float(self.rng_mgr.rng_sensor.normal(0.0, self.PHASE_NOISE_STD))
                meas_phase = float(true_phase + noise_phase)
                # Wrap to [-pi, pi]
                meas_phase = float((meas_phase + np.pi) % (2.0 * np.pi) - np.pi)
            else:
                meas_phase = None  # Masked below floor

            rec = {
                "step": self.step_count,
                "query_input": {"frequency_hz": float(freq)},
                "reading": {
                    "current_amplitude_A": meas_amp,
                    "current_phase_rad": meas_phase,
                },
                "noise_sigma": {
                    "amplitude_A": self.AMPLITUDE_NOISE_STD,
                    "phase_rad": self.PHASE_NOISE_STD,
                }
            }
            self.history.append(rec)

            return (
                self.get_public_observation(),
                0.0,
                False,
                truncated,
                {
                    "status": "measured",
                    "reading": rec["reading"],
                    "calls_made": self.instrument_calls,
                }
            )

        # SUBMIT ACTION
        elif action_type == "submit":
            try:
                sub_action = SubmitCircuitAction(**action)
            except Exception as e:
                return (
                    self.get_public_observation(),
                    self.INVALID_ACTION_PENALTY,
                    False,
                    truncated,
                    {"error": f"Schema error: {e}", "is_valid": False, "status": "schema_error"}
                )

            self.is_terminated = True
            remaining_turns = max(0, self.max_turns - self.step_count)

            from resonance.core.scoring import score_episode
            breakdown = score_episode(
                world=self._hidden_params,
                history=self.history,
                submission=sub_action.model_dump(),
                eval_grid=self._eval_grid,
                V_drive_rms=self.V_DRIVE_RMS,
                query_cost=self.query_cost,
                zero_measurement_cap=0.20,
                submission_timing_bonus=self.submission_timing_bonus,
                remaining_turns=remaining_turns,
            )

            reward = breakdown.total_reward
            info = {
                "status": "submitted",
                "submitted": True,
                "rmse": float(breakdown.nrmse * max(float(np.max(np.abs(compute_complex_current(
                    circuit_type=self._hidden_params["circuit_type"],
                    frequency_hz=self._eval_grid,
                    V_rms=self.V_DRIVE_RMS,
                    R=self._hidden_params["R"],
                    L_H=self._hidden_params["L_H"],
                    C_F=self._hidden_params["C_F"],
                )))), 1e-6)),
                "nrmse": breakdown.nrmse,
                "reward": reward,
                "terminal_reward": reward,
                "brier_score": breakdown.brier_score,
                "is_correct_type": breakdown.is_correct_type,
                "calls_made": self.instrument_calls,
                "steps": self.step_count,
                "reward_breakdown": breakdown.to_dict(),
                "predicted": sub_action.parameters.model_dump(),
                "model_probs": sub_action.model_probs,
            }

            return self.get_public_observation(), reward, True, False, info

        else:
            return (
                self.get_public_observation(),
                self.INVALID_ACTION_PENALTY,
                False,
                truncated,
                {"error": f"Unknown action type: {action_type}. Expected 'measure' or 'submit'.", "is_valid": False}
            )
