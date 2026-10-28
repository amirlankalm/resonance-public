"""Base agent interfaces and test mock agents for Resonance."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseAgent(ABC):
    """Abstract agent interface for scientific reasoning and active experimentation."""

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the next action given public observation.

        Returns:
            Dictionary conforming to either MeasureAction or SubmitAction schema.
        """
        pass

    def reset(self) -> None:
        """Reset internal agent memory / conversational state at episode start."""
        pass


class MockHeuristicAgent(BaseAgent):
    """Fast deterministic test agent that takes 3 exploratory measurements then submits."""

    def __init__(self, target_task: str = "circuit"):
        self.target_task = target_task
        self.measurements_done = 0

    def reset(self) -> None:
        self.measurements_done = 0

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        history = observation.get("history", [])

        if self.target_task == "circuit":
            test_frequencies = [100.0, 1000.0, 10000.0]
            if len(history) < len(test_frequencies) and observation.get("remaining_calls", 0) > 0:
                freq = test_frequencies[len(history)]
                return {"type": "measure", "frequency_hz": freq}

            # Submit
            return {
                "type": "submit",
                "model_probs": {"RC": 0.1, "RL": 0.1, "RLC": 0.8},
                "parameters": {
                    "circuit_type": "RLC",
                    "R": 100.0,
                    "L_mH": 25.0,
                    "C_uF": 2.5,
                }
            }
        else:
            test_angles = [0.0, 45.0, 90.0]
            if len(history) < len(test_angles) and observation.get("remaining_calls", 0) > 0:
                angle = test_angles[len(history)]
                return {"type": "measure", "angle_deg": angle}

            return {
                "type": "submit",
                "amplitude": 0.7,
                "offset": 0.05,
                "phi_deg": 45.0,
            }
