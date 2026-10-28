"""Scripted solver-backed schema reference policy for Resonance.

Communicates purely through the strict JSON schema, using an internal numerical least-squares solver
(evaluate_candidate_models) for parameter fitting upon submission.
Proves that the JSON API and physics environment are fully solvable and schema-robust,
but operates as a solver-backed schema reference rather than an unassisted learned model baseline.
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from resonance.envs.circuit import CircuitSpectroscopyEnv, CircuitType
from resonance.baselines.numerical import evaluate_candidate_models


class StrictReferencePolicy:
    """Scripted solver-backed schema reference that communicates purely through strict JSON strings."""

    def __init__(
        self,
        probe_frequencies: Optional[List[float]] = None,
        V_rms: float = 0.1,
    ):
        self.solver_used = True
        self.fit_source = "numerical_fitter"
        if probe_frequencies is None:
            # 12 geometrically spaced diagnostic frequencies covering the band [30 Hz, 30 kHz]
            self.probe_frequencies = list(np.geomspace(30.0, 30000.0, 12))
        else:
            self.probe_frequencies = list(probe_frequencies)
        self.V_rms = float(V_rms)

    def select_action(self, observation_text: str, step_index: int) -> str:
        """Parse text history and return next JSON action string."""
        history = self._parse_history(observation_text)

        # If we have probed all frequencies, compute fit and submit
        if len(history) >= len(self.probe_frequencies):
            fit_result = evaluate_candidate_models(history, V_rms=self.V_rms)
            submit_payload = {
                "type": "submit",
                "model_probs": fit_result["model_probs"],
                "parameters": fit_result["parameters"],
            }
            return json.dumps(submit_payload, indent=2)

        # Otherwise, probe next diagnostic frequency
        next_freq = self.probe_frequencies[len(history)]
        measure_payload = {
            "type": "measure",
            "frequency_hz": float(next_freq),
        }
        return json.dumps(measure_payload, indent=2)

    @staticmethod
    def _parse_history(text: str) -> List[Dict[str, Any]]:
        """Parse structured history lines from environment text prompt."""
        history: List[Dict[str, Any]] = []
        # Match lines like: "f=30.0 Hz -> I_rms=0.154 mA, phase=1.522 rad"
        pattern = re.compile(
            r"f=([0-9.]+)\s*Hz\s*->\s*I_rms=([0-9.]+)\s*mA,\s*phase=([0-9.\-]+|Masked)\s*rad"
        )
        for line in text.splitlines():
            match = pattern.search(line)
            if match:
                f = float(match.group(1))
                i_mA = float(match.group(2))
                phase_str = match.group(3)
                phase = float(phase_str) if phase_str != "Masked" else None
                history.append({
                    "query_input": {"frequency_hz": f},
                    "reading": {
                        "current_amplitude_A": i_mA * 1e-3,
                        "current_phase_rad": phase,
                    },
                })
        return history


def run_strict_reference_episode(
    env: CircuitSpectroscopyEnv,
    seed: int,
    policy: Optional[StrictReferencePolicy] = None,
) -> Dict[str, Any]:
    """Execute a single episode in the strict unassisted JSON format."""
    if policy is None:
        policy = StrictReferencePolicy()

    obs_dict, info = env.reset(seed=seed)
    prompt = (
        "You are a physicist conducting an active experiment in circuit spectroscopy. "
        "Select probe frequencies and submit fitted circuit hypotheses in JSON format.\n"
        "Remaining measurement budget: 12.\n"
        "History:\n  (No measurements taken yet)\n"
        "Provide your next action (measure or submit) as JSON."
    )

    total_reward = 0.0
    steps = 0
    valid_actions = 0
    total_actions = 0
    submitted = False
    last_info = info

    for turn in range(env.max_turns):
        action_json = policy.select_action(prompt, turn)
        total_actions += 1

        obs_dict, reward, terminated, truncated, last_info = env.step(action_json)
        total_reward += reward
        steps += 1

        if last_info.get("status") != "invalid_format":
            valid_actions += 1

        if terminated:
            submitted = (last_info.get("status") == "submitted")
            break
        if truncated:
            break

        # Reconstruct next text prompt from observation history
        rem_calls = obs_dict.get("remaining_calls", 0)
        history = obs_dict.get("history", [])
        hist_lines = []
        for h in history:
            f = h["query_input"]["frequency_hz"]
            i_mA = (h["reading"]["current_amplitude_A"] or 0.0) * 1e3
            phase = h["reading"]["current_phase_rad"]
            p_str = f"{phase:.3f}" if phase is not None else "Masked"
            hist_lines.append(f"  f={f:.1f} Hz -> I_rms={i_mA:.3f} mA, phase={p_str} rad")
        hist_str = "\n".join(hist_lines) if hist_lines else "  (No measurements taken yet)"

        prompt = (
            f"Remaining measurement budget: {rem_calls}.\n"
            f"History:\n{hist_str}\n"
            "Provide your next action (measure or submit) as JSON."
        )

    return {
        "world_seed": seed,
        "steps": steps,
        "calls_made": last_info.get("calls_made", 0),
        "terminal_reward": float(last_info.get("terminal_reward", total_reward)),
        "nrmse": last_info.get("nrmse"),
        "brier_score": last_info.get("brier_score"),
        "is_correct_type": last_info.get("is_correct_type", False),
        "valid_action_rate": valid_actions / max(1, total_actions),
        "submitted": submitted,
    }
