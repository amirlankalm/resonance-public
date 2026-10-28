"""Resonance physics engines and analytical solvers."""

from resonance.physics.optics import malus_law, generate_polarization_eval_grid
from resonance.physics.circuits import (
    CircuitType,
    compute_impedance,
    compute_complex_current,
    compute_current_amplitude_and_phase,
    compute_resonant_frequency,
    compute_quality_factor,
)

__all__ = [
    "malus_law",
    "generate_polarization_eval_grid",
    "CircuitType",
    "compute_impedance",
    "compute_complex_current",
    "compute_current_amplitude_and_phase",
    "compute_resonant_frequency",
    "compute_quality_factor",
]
