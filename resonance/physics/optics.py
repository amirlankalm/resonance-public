"""Optics physics models for Resonance (e.g. Malus' Law starter environment)."""

from __future__ import annotations
import numpy as np


def malus_law(
    theta_deg: np.ndarray | float,
    amplitude: float,
    offset: float,
    phi_deg: float,
) -> np.ndarray | float:
    """Compute light intensity following Malus' Law:

    y = b + A * cos^2(theta - phi)

    Args:
        theta_deg: Angle of analyzer in degrees.
        amplitude: Peak intensity amplitude A.
        offset: Baseline intensity offset b.
        phi_deg: Polarization axis phi in degrees.

    Returns:
        Intensity y in detector units.
    """
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    cos_val = np.cos(theta_rad - phi_rad)
    return offset + amplitude * (cos_val**2)


def generate_polarization_eval_grid(num_points: int = 64) -> np.ndarray:
    """Generate deterministic evaluation grid for polarization environment.

    Args:
        num_points: Number of evaluation angles (default: 64).

    Returns:
        np.ndarray of angles in [0, 180] degrees.
    """
    return np.linspace(0.0, 180.0, num_points, endpoint=True)
