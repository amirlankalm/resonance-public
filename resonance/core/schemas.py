"""Pydantic schemas and data validation models for Resonance."""

from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, model_validator, ConfigDict


def extract_and_parse_action(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first valid JSON object representing an action.
    
    Handles markdown code fences, leading/trailing thought tokens, and nested dictionaries.
    Uses json.JSONDecoder.raw_decode to avoid truncation on nested braces.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None
        
    text = raw_text.strip()
    
    # 1. Check for markdown code fence ```json ... ``` or ``` ... ```
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    fence_matches = re.findall(fence_pattern, text, re.IGNORECASE)
    for block in fence_matches:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
            
    # 2. Search for '{' and use raw_decode to extract top-level JSON object
    decoder = json.JSONDecoder()
    start_pos = 0
    while True:
        idx = text.find("{", start_pos)
        if idx == -1:
            break
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start_pos = idx + 1
        
    return None


class MeasurementRecord(BaseModel):
    """A single executed measurement observation."""
    step: int
    query_input: Dict[str, float]
    reading: Dict[str, Optional[float]]
    noise_sigma: Dict[str, float]


class Observation(BaseModel):
    """Public environment observation delivered to the agent."""
    task_name: str
    task_version: str
    candidate_formulas: Dict[str, str]
    units: Dict[str, str]
    bounds: Dict[str, Any]
    noise_specification: Dict[str, float]
    history: List[MeasurementRecord] = Field(default_factory=list)
    remaining_calls: int
    remaining_turns: int
    output_schema: Dict[str, Any]


class MeasureAction(BaseModel):
    """Action requesting an instrument measurement."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["measure"] = "measure"
    frequency_hz: Optional[float] = None
    angle_deg: Optional[float] = None

    @model_validator(mode="after")
    def validate_inputs(self) -> "MeasureAction":
        if self.frequency_hz is None and self.angle_deg is None:
            raise ValueError("Must provide either 'frequency_hz' or 'angle_deg'.")
        if self.frequency_hz is not None:
            if not np_isfinite(self.frequency_hz):
                raise ValueError("frequency_hz must be a finite number.")
            if self.frequency_hz <= 0:
                raise ValueError("frequency_hz must be strictly positive.")
        if self.angle_deg is not None:
            if not np_isfinite(self.angle_deg):
                raise ValueError("angle_deg must be a finite number.")
        return self


class CircuitParameters(BaseModel):
    """Predicted circuit parameter values."""
    model_config = ConfigDict(extra="forbid")
    circuit_type: Literal["RC", "RL", "RLC"]
    R: float = Field(..., gt=0, description="Resistance in Ohms")
    L_mH: Optional[float] = Field(None, description="Inductance in millihenries (required for RL, RLC)")
    C_uF: Optional[float] = Field(None, description="Capacitance in microfarads (required for RC, RLC)")

    @model_validator(mode="after")
    def validate_circuit_params(self) -> "CircuitParameters":
        for val in [self.R, self.L_mH, self.C_uF]:
            if val is not None and not np_isfinite(val):
                raise ValueError("Parameter values must be finite numbers.")

        if self.circuit_type in ("RL", "RLC"):
            if self.L_mH is None or self.L_mH <= 0:
                raise ValueError(f"L_mH must be positive for circuit_type={self.circuit_type}")
        if self.circuit_type in ("RC", "RLC"):
            if self.C_uF is None or self.C_uF <= 0:
                raise ValueError(f"C_uF must be positive for circuit_type={self.circuit_type}")
        return self


class SubmitCircuitAction(BaseModel):
    """Final submission action for Circuit Spectroscopy."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["submit"] = "submit"
    model_probs: Dict[str, float] = Field(
        ...,
        description="Probabilities assigned to candidate hypotheses: RC, RL, RLC"
    )
    parameters: CircuitParameters

    @model_validator(mode="after")
    def validate_probs(self) -> "SubmitCircuitAction":
        expected_keys = {"RC", "RL", "RLC"}
        if set(self.model_probs.keys()) != expected_keys:
            raise ValueError(f"model_probs keys must be strictly {expected_keys}, got {set(self.model_probs.keys())}")
        for k in expected_keys:
            p = self.model_probs[k]
            if not np_isfinite(p) or p < 0:
                raise ValueError(f"Probability for {k} must be finite and non-negative.")
        prob_sum = sum(self.model_probs.values())
        if prob_sum <= 0:
            raise ValueError("Sum of model_probs must be greater than zero.")
        return self


class SubmitPolarizationAction(BaseModel):
    """Final submission action for Polarization starter."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["submit"] = "submit"
    amplitude: float = Field(..., description="Intensity amplitude A")
    offset: float = Field(..., description="Baseline offset b")
    phi_deg: float = Field(..., description="Polarization axis phi in degrees")

    @model_validator(mode="after")
    def validate_polarization_params(self) -> "SubmitPolarizationAction":
        for val in [self.amplitude, self.offset, self.phi_deg]:
            if not np_isfinite(val):
                raise ValueError("Parameters must be finite numbers.")
        return self


def np_isfinite(val: Any) -> bool:
    try:
        import math
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False
