"""Multi-turn trajectory collector producing scripted numerical-teacher demonstrations for SFT and RL.

Generates scripted demonstrations via sequential adaptive experimental design and numerical fitting.
Discloses explicitly that demonstrations are produced by a scripted algorithmic teacher,
not human expert verification.
"""

from __future__ import annotations
import json
from typing import Dict, Any, List, Optional
import numpy as np

from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.core.prompts import (
    SYSTEM_PROMPT_CIRCUIT,
    format_observation_user_prompt,
    build_initial_chat_history,
)
from resonance.baselines.numerical import evaluate_candidate_models


class MultiTurnTrajectoryCollector:
    """Collects multi-turn conversational trajectories from a scripted numerical teacher."""

    def __init__(self, env_class=CircuitSpectroscopyEnv):
        self.env_class = env_class

    def collect_expert_trajectory(
        self,
        world_seed: int,
        min_measurements: int = 3,
        max_measurements: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a scripted numerical-teacher demonstration trajectory.
        
        Uses adaptive disagreement exploration (EIG) and posterior-convergence early stopping
        to model active stopping decisions and generate a realistic spread of trajectory lengths.
        Note: Produced by a scripted algorithmic policy, not human expert verification.
        """
        env = self.env_class()
        obs, _ = env.reset(seed=world_seed)

        # Standard initial chat history (R5 byte-identical sharing)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_CIRCUIT, "mask_loss": False}
        ]

        if max_measurements is None:
            # Randomize target measurement budget and early-stop gate to ensure length diversity (R6)
            rng = np.random.default_rng(world_seed + 777)
            target_measurements = int(rng.choice([4, 5, 6, 8, 10]))
            min_stop_gate = int(rng.choice([3, 4, 5, 6]))
        else:
            target_measurements = max_measurements
            min_stop_gate = min_measurements

        stop_reason = "budget_reached"
        grid_freqs = np.geomspace(env.BAND_MIN_HZ, env.BAND_MAX_HZ, 40)
        probes = [env.BAND_MIN_HZ, env.BAND_MAX_HZ]

        while len(obs["history"]) < target_measurements and obs["remaining_calls"] > 0:
            # Check posterior convergence early stopping (R6)
            if len(obs["history"]) >= min_stop_gate:
                fit_res = evaluate_candidate_models(obs["history"], V_rms=env.V_DRIVE_RMS)
                probs = fit_res.get("model_probs", {})
                fits = fit_res.get("fits", {})
                max_p = max(probs.values()) if probs else 0.0
                sel = fit_res.get("selected_type")
                rss = fits.get(sel, {}).get("rss", float("inf"))
                if max_p > 0.98 and rss < 5e-7:
                    stop_reason = "posterior_converged"
                    break

            # User turn: provide current observations using unified prompt formatter (R5)
            user_msg = format_observation_user_prompt(obs)
            messages.append({"role": "user", "content": user_msg, "mask_loss": False})

            # Assistant turn: pick probe action
            if len(obs["history"]) < 2:
                freq = float(probes[len(obs["history"])])
                action = {"type": "measure", "frequency_hz": freq}
            else:
                fit_res = evaluate_candidate_models(obs["history"], V_rms=env.V_DRIVE_RMS)
                fits = fit_res.get("fits", {})
                from resonance.physics.circuits import compute_complex_current, CircuitType

                I_rc = compute_complex_current(
                    CircuitType.RC, grid_freqs, env.V_DRIVE_RMS,
                    fits["RC"]["R"], C_F=fits["RC"]["C_uF"] * 1e-6
                )
                I_rl = compute_complex_current(
                    CircuitType.RL, grid_freqs, env.V_DRIVE_RMS,
                    fits["RL"]["R"], L_H=fits["RL"]["L_mH"] * 1e-3
                )
                I_rlc = compute_complex_current(
                    CircuitType.RLC, grid_freqs, env.V_DRIVE_RMS,
                    fits["RLC"]["R"], L_H=fits["RLC"]["L_mH"] * 1e-3, C_F=fits["RLC"]["C_uF"] * 1e-6
                )
                amps = np.array([np.abs(I_rc), np.abs(I_rl), np.abs(I_rlc)])
                disagreement = np.var(amps, axis=0)

                visited = [h["query_input"]["frequency_hz"] for h in obs["history"]]
                for vf in visited:
                    dist = np.abs(np.log10(grid_freqs) - np.log10(vf))
                    penalty = np.exp(-0.5 * (dist / 0.15) ** 2)
                    disagreement *= (1.0 - 0.9 * penalty)

                best_idx = int(np.argmax(disagreement))
                action = {"type": "measure", "frequency_hz": float(grid_freqs[best_idx])}

            assistant_msg = f"```json\n{json.dumps(action, indent=2)}\n```"
            messages.append({"role": "assistant", "content": assistant_msg, "mask_loss": True})

            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

        # Final submission turn using unified prompt formatter
        user_msg = format_observation_user_prompt(obs)
        messages.append({"role": "user", "content": user_msg, "mask_loss": False})

        fit_result = evaluate_candidate_models(obs["history"], V_rms=env.V_DRIVE_RMS)
        submit_action = {
            "type": "submit",
            "model_probs": fit_result["model_probs"],
            "parameters": fit_result["parameters"],
        }
        assistant_submit = f"```json\n{json.dumps(submit_action, indent=2)}\n```"
        messages.append({"role": "assistant", "content": assistant_submit, "mask_loss": True})

        obs, reward, term, trunc, info = env.step(submit_action)

        return {
            "world_seed": world_seed,
            "messages": messages,
            "trajectory_length": len(obs["history"]),
            "stop_reason": stop_reason,
            "reward": float(reward),
            "nrmse": info.get("nrmse"),
            "brier_score": info.get("brier_score"),
            "is_correct_type": info.get("is_correct_type"),
        }
