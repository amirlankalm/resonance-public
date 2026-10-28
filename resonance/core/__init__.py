"""Core infrastructure, schemas, and scoring for Resonance."""

from resonance.core.rng import IsolatedRNGManager
from resonance.core.memory import MemoryGuard
from resonance.core.schemas import (
    MeasurementRecord,
    Observation,
    MeasureAction,
    CircuitParameters,
    SubmitCircuitAction,
    SubmitPolarizationAction,
)
from resonance.core.scoring import (
    compute_complex_response_rmse,
    compute_nrmse,
    compute_brier_score,
    compute_terminal_reward,
    audit_action_payload,
)
from resonance.core.prompts import (
    SYSTEM_PROMPT_CIRCUIT,
    format_observation_user_prompt,
    build_initial_chat_history,
)

__all__ = [
    "IsolatedRNGManager",
    "MemoryGuard",
    "MeasurementRecord",
    "Observation",
    "MeasureAction",
    "CircuitParameters",
    "SubmitCircuitAction",
    "SubmitPolarizationAction",
    "SYSTEM_PROMPT_CIRCUIT",
    "format_observation_user_prompt",
    "build_initial_chat_history",
    "compute_complex_response_rmse",
    "compute_nrmse",
    "compute_brier_score",
    "compute_terminal_reward",
    "audit_action_payload",
]
