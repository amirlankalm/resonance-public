"""Nous Research Atropos environment integration for Resonance.

Provides the multi-turn interaction loop, reward computation, and schema verification
compatible with the Atropos RL environment standard.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.splits import get_dev_seeds, get_train_seeds, derive_test_seeds


class AtroposResonanceEnv:
    """Nous Research Atropos compatible multi-turn environment."""

    def __init__(
        self,
        split: str = "dev",
        problem_index: int = 0,
        salt: Optional[str] = None,
        **env_kwargs: Any,
    ):
        self.split = split
        if split == "dev":
            seeds = get_dev_seeds()
        elif split == "train":
            seeds = get_train_seeds(count=1000)
        elif split == "test":
            if not salt:
                raise ValueError("Evaluating on test split requires a verified secret salt.")
            seeds = derive_test_seeds(salt, count=100)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.seed = seeds[problem_index % len(seeds)]
        self.env = CircuitSpectroscopyEnv(**env_kwargs)
        self.obs, self.info = self.env.reset(seed=self.seed)

    def initial_prompt(self) -> str:
        """Return the initial prompt presented to the agent."""
        return (
            "You are a physicist conducting an active experiment in circuit spectroscopy. "
            "Select probe frequencies and submit fitted circuit hypotheses in JSON format.\n"
            "Remaining measurement budget: 12.\n"
            "History:\n  (No measurements taken yet)\n"
            "Provide your next action (measure or submit) as JSON."
        )

    def step(self, action_json_or_dict: Any) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Process an agent turn and return next text observation, reward, done flag, and info."""
        obs, reward, terminated, truncated, info = self.env.step(action_json_or_dict)
        done = terminated or truncated

        # Render next state as textual message for language model policy
        rem_calls = obs.get("remaining_calls", 0)
        history = obs.get("history", [])
        if not history:
            hist_str = "  (No measurements taken yet)"
        else:
            lines = []
            for h in history:
                f = h["query_input"]["frequency_hz"]
                i_mA = (h["reading"]["current_amplitude_A"] or 0.0) * 1e3
                phase = h["reading"]["current_phase_rad"]
                p_str = f"{phase:.3f} rad" if phase is not None else "Masked"
                lines.append(f"  f={f:.1f} Hz -> I_rms={i_mA:.3f} mA, phase={p_str}")
            hist_str = "\n".join(lines)

        next_text = (
            f"Remaining measurement budget: {rem_calls}.\n"
            f"History:\n{hist_str}\n"
            "Provide your next action (measure or submit) as JSON."
        )
        return next_text, float(reward), bool(done), info
