"""Tests for circuit physics equations, impedance, limits, and resonance."""

import unittest
import numpy as np
from resonance.physics.circuits import (
    CircuitType,
    compute_impedance,
    compute_complex_current,
    compute_current_amplitude_and_phase,
    compute_resonant_frequency,
    compute_quality_factor,
)


class TestCircuitPhysics(unittest.TestCase):

    def test_rc_low_and_high_frequency_limits(self):
        """RC circuit: Z = R - j / (omega * C).

        As f -> 0: |Z| -> inf, |I| -> 0, phase -> +pi/2 (+90 deg).
        As f -> inf: |Z| -> R, |I| -> V/R, phase -> 0.
        """
        R = 50.0
        C_F = 1e-6  # 1 uF
        V = 0.1

        # Very low frequency (1 Hz)
        amp_low, phase_low, _ = compute_current_amplitude_and_phase(
            CircuitType.RC, 1.0, V, R, C_F=C_F, detection_floor=1e-12
        )
        # Capacitor reactance X_C = 1 / (2*pi*1*1e-6) ~ 159,155 Ohms
        # I ~ 0.1 / 159155 ~ 6.28e-7 A
        self.assertAlmostEqual(amp_low, 0.1 / (1.0 / (2 * np.pi * 1.0 * 1e-6)), places=8)
        self.assertAlmostEqual(phase_low, np.pi / 2, places=2)

        # High frequency (1 MHz)
        amp_high, phase_high, _ = compute_current_amplitude_and_phase(
            CircuitType.RC, 1e6, V, R, C_F=C_F, detection_floor=1e-12
        )
        self.assertAlmostEqual(amp_high, V / R, places=4)
        self.assertAlmostEqual(phase_high, 0.0, places=2)

    def test_rl_low_and_high_frequency_limits(self):
        """RL circuit: Z = R + j * omega * L.

        As f -> 0: |Z| -> R, |I| -> V/R, phase -> 0.
        As f -> inf: |Z| -> inf, |I| -> 0, phase -> -pi/2 (-90 deg).
        """
        R = 50.0
        L_H = 0.02  # 20 mH
        V = 0.1

        # Low frequency (1 Hz)
        amp_low, phase_low, _ = compute_current_amplitude_and_phase(
            CircuitType.RL, 1.0, V, R, L_H=L_H, detection_floor=1e-12
        )
        self.assertAlmostEqual(amp_low, V / R, places=4)
        self.assertAlmostEqual(phase_low, 0.0, places=2)

        # High frequency (1 MHz)
        amp_high, phase_high, _ = compute_current_amplitude_and_phase(
            CircuitType.RL, 1e6, V, R, L_H=L_H, detection_floor=1e-12
        )
        self.assertAlmostEqual(amp_high, 0.0, places=5)
        self.assertAlmostEqual(phase_high, -np.pi / 2, places=2)

    def test_rlc_resonance_peak(self):
        """RLC series circuit at resonance: omega_0 * L = 1 / (omega_0 * C).

        At f_0: reactances cancel, Z = R, |I| is strictly maximal = V/R, phase = 0.
        """
        R = 40.0
        L_H = 0.025  # 25 mH
        C_F = 1e-6   # 1 uF
        V = 0.1

        omega_0, f_0 = compute_resonant_frequency(L_H, C_F)
        # f_0 = 1 / (2*pi*sqrt(0.025 * 1e-6)) = 1 / (2*pi*5e-4) ~ 1006.58 Hz
        self.assertAlmostEqual(f_0, 1006.58, places=1)

        # At resonance
        amp_res, phase_res, _ = compute_current_amplitude_and_phase(
            CircuitType.RLC, f_0, V, R, L_H=L_H, C_F=C_F
        )
        self.assertAlmostEqual(amp_res, V / R, places=5)
        self.assertAlmostEqual(phase_res, 0.0, places=5)

        # Above and below resonance, current must be strictly less than V/R
        amp_below, _, _ = compute_current_amplitude_and_phase(
            CircuitType.RLC, f_0 * 0.5, V, R, L_H=L_H, C_F=C_F
        )
        amp_above, _, _ = compute_current_amplitude_and_phase(
            CircuitType.RLC, f_0 * 2.0, V, R, L_H=L_H, C_F=C_F
        )
        self.assertLess(amp_below, amp_res)
        self.assertLess(amp_above, amp_res)

    def test_sensor_floor_phase_masking(self):
        """When amplitude is below sensor floor, phase must be masked (NaN)."""
        R = 50.0
        L_H = 1.0  # huge inductance -> tiny current at high frequency
        V = 0.1
        floor = 1e-5

        amp, phase, above_floor = compute_current_amplitude_and_phase(
            CircuitType.RL, 30000.0, V, R, L_H=L_H, detection_floor=floor
        )
        self.assertFalse(above_floor)
        self.assertTrue(np.isnan(phase))


if __name__ == "__main__":
    unittest.main()
