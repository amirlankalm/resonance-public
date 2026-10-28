"""Shared prompt formatting and constants for Resonance circuit spectroscopy."""

from __future__ import annotations
from typing import Dict, Any, List

SYSTEM_PROMPT_CIRCUIT = (
    "You are a physicist conducting an active experiment in circuit spectroscopy. "
    "Select probe frequencies and submit fitted circuit hypotheses in JSON format."
)


def format_observation_user_prompt(obs: Dict[str, Any]) -> str:
    """Format the observation history and current budget into the standardized user turn prompt.
    
    This function is strictly shared across SFT data generation, GRPO rollouts,
    and model evaluation, ensuring byte-identical prompt generation.
    """
    history = obs.get("history", [])
    rem_calls = obs.get("remaining_calls", 0)

    lines = [
        f"Remaining measurement budget: {rem_calls}.",
        "History:",
    ]

    if not history:
        lines.append("  (No measurements taken yet)")
    else:
        for item in history:
            f_val = item["query_input"].get("frequency_hz")
            amp_val = item["reading"].get("current_amplitude_A")
            phase_val = item["reading"].get("current_phase_rad")
            phase_str = f"{phase_val:.3f} rad" if phase_val is not None else "Masked"
            amp_mA = (amp_val * 1000.0) if amp_val is not None else 0.0
            lines.append(f"  f={f_val:.1f} Hz -> I_rms={amp_mA:.3f} mA, phase={phase_str}")

    action_hint = "measure or submit" if rem_calls > 0 else "final submit"
    lines.append(f"Provide your next action ({action_hint}) as JSON.")
    return "\n".join(lines)


def build_initial_chat_history() -> List[Dict[str, str]]:
    """Return standard initial multi-turn chat history containing the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT_CIRCUIT}]
