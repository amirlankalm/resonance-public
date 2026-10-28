"""AC Circuit spectroscopy physics calculations.

Implements sinusoidal steady-state impedance, current, amplitude, and phase
for linear series RC, RL, and RLC circuits.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, Any, Tuple
import numpy as np


class CircuitType(str, Enum):
    RC = "RC"
    RL = "RL"
    RLC = "RLC"


def compute_omega(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    """Convert cyclic frequency in Hz to angular frequency in rad/s: omega = 2 * pi * f."""
    return 2.0 * np.pi * frequency_hz


def compute_impedance(
    circuit_type: CircuitType | str,
    frequency_hz: np.ndarray | float,
    R: float,
    L_H: float | None = None,
    C_F: float | None = None,
) -> np.ndarray | complex:
    """Compute complex impedance Z(omega) for a linear circuit.

    Formulas:
        Z_RC  = R - j / (omega * C)
        Z_RL  = R + j * omega * L
        Z_RLC = R + j * (omega * L - 1 / (omega * C))

    Args:
        circuit_type: RC, RL, or RLC
        frequency_hz: frequency in Hz (> 0)
        R: resistance in Ohms (R > 0)
        L_H: inductance in Henries (for RL, RLC)
        C_F: capacitance in Farads (for RC, RLC)

    Returns:
        Complex impedance Z in Ohms.
    """
    ctype = CircuitType(circuit_type)
    omega = compute_omega(frequency_hz)

    # Avoid zero division
    omega = np.maximum(omega, 1e-12)

    if ctype == CircuitType.RC:
        if C_F is None or C_F <= 0:
            raise ValueError("Capacitance C_F must be positive for RC circuit.")
        # Z_RC = R - i / (omega * C)
        return R - 1j / (omega * C_F)

    elif ctype == CircuitType.RL:
        if L_H is None or L_H <= 0:
            raise ValueError("Inductance L_H must be positive for RL circuit.")
        # Z_RL = R + i * omega * L
        return R + 1j * (omega * L_H)

    elif ctype == CircuitType.RLC:
        if L_H is None or L_H <= 0 or C_F is None or C_F <= 0:
            raise ValueError("Both L_H and C_F must be positive for RLC circuit.")
        # Z_RLC = R + i * (omega * L - 1 / (omega * C))
        return R + 1j * (omega * L_H - 1.0 / (omega * C_F))

    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")


def compute_complex_current(
    circuit_type: CircuitType | str,
    frequency_hz: np.ndarray | float,
    V_rms: float,
    R: float,
    L_H: float | None = None,
    C_F: float | None = None,
) -> np.ndarray | complex:
    """Compute complex current I = V / Z (phasor in RMS units)."""
    Z = compute_impedance(circuit_type, frequency_hz, R, L_H, C_F)
    return V_rms / Z


def compute_current_amplitude_and_phase(
    circuit_type: CircuitType | str,
    frequency_hz: np.ndarray | float,
    V_rms: float,
    R: float,
    L_H: float | None = None,
    C_F: float | None = None,
    detection_floor: float = 1e-6,
) -> Tuple[np.ndarray | float, np.ndarray | float, np.ndarray | bool]:
    """Compute current amplitude |I| (Amperes RMS) and phase angle (radians relative to drive).

    Args:
        circuit_type: RC, RL, or RLC
        frequency_hz: frequency in Hz
        V_rms: drive voltage in Volts RMS
        R: resistance in Ohms
        L_H: inductance in Henries
        C_F: capacitance in Farads
        detection_floor: amplitude detection threshold below which phase is masked

    Returns:
        amplitude: |I| in Amperes RMS
        phase_rad: angle(I) in [-pi, pi] radians (or masked if below floor)
        is_above_floor: boolean mask of valid measurement
    """
    I_complex = compute_complex_current(circuit_type, frequency_hz, V_rms, R, L_H, C_F)
    amplitude = np.abs(I_complex)
    phase_rad = np.angle(I_complex)

    is_scalar = np.isscalar(amplitude)
    if is_scalar:
        above_floor = bool(amplitude >= detection_floor)
        if not above_floor:
            # Below sensor floor, phase is unresolvable
            phase_rad = float("nan")
        return float(amplitude), float(phase_rad), above_floor
    else:
        above_floor = amplitude >= detection_floor
        phase_rad = np.where(above_floor, phase_rad, np.nan)
        return amplitude, phase_rad, above_floor


def compute_resonant_frequency(L_H: float, C_F: float) -> Tuple[float, float]:
    """Compute angular and cyclic resonant frequency for series RLC:

    omega_0 = 1 / sqrt(L * C)
    f_0 = omega_0 / (2 * pi)
    """
    if L_H <= 0 or C_F <= 0:
        raise ValueError("L_H and C_F must be positive.")
    omega_0 = 1.0 / np.sqrt(L_H * C_F)
    f_0 = omega_0 / (2.0 * np.pi)
    return float(omega_0), float(f_0)


def compute_quality_factor(R: float, L_H: float, C_F: float) -> float:
    """Compute quality factor Q = (1 / R) * sqrt(L / C) for series RLC."""
    if R <= 0 or L_H <= 0 or C_F <= 0:
        raise ValueError("R, L_H, C_F must be positive.")
    return float((1.0 / R) * np.sqrt(L_H / C_F))
