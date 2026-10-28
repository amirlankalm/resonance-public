"""Base environment class for Resonance physics environments."""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np

from resonance.core.rng import IsolatedRNGManager
from resonance.core.scoring import audit_action_payload


class BasePhysicsEnv(ABC):
    """Abstract base class for active scientific experiment environments."""

    def __init__(
        self,
        measurement_budget: int = 12,
        max_turns: int = 20,
        query_cost: float = 0.005,
    ):
        self.measurement_budget = measurement_budget
        self.max_turns = max_turns
        self.query_cost = query_cost

        self.step_count = 0
        self.instrument_calls = 0
        self.is_terminated = False
        self.is_truncated = False
        self.history: list = []
        self.rng_mgr: IsolatedRNGManager = IsolatedRNGManager(0)

        # Hidden parameters are strictly encapsulated
        self._hidden_params: Dict[str, Any] = {}
        self._eval_grid: np.ndarray = np.array([])

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to a new world.

        Returns:
            observation: public observation dictionary
            info: public information dictionary
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one environment turn.

        Returns:
            observation: public observation dictionary
            reward: reward signal (0 until terminal submission or budget exhaustion)
            terminated: whether episode terminated normally (e.g. submit)
            truncated: whether episode truncated by turn limit
            info: diagnostic metrics and logs
        """
        pass

    @abstractmethod
    def get_public_observation(self) -> Dict[str, Any]:
        """Return public observation containing no hidden truth variables."""
        pass

    def check_turn_limits(self) -> bool:
        """Increment turn and check truncation."""
        self.step_count += 1
        if self.step_count >= self.max_turns:
            self.is_truncated = True
            return True
        return False

    def validate_action_security(self, action: Any) -> Tuple[bool, str]:
        """Run security checks against malicious action payloads."""
        return audit_action_payload(action)

    def reseed_sensor(self, run_seed: int) -> None:
        """Reseed only the sensor noise stream (for multiple noisy rollouts of same world)."""
        if hasattr(self, "rng_mgr") and self.rng_mgr is not None:
            self.rng_mgr.reseed_sensor(run_seed)
