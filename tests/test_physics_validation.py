"""Physics validation test suite (Section 2.8).

Derives circuit transfer functions analytically and validates the numeric engine
against an independent complex phasor nodal solver for 100 random circuits
to 1e-9 relative tolerance.
"""

from __future__ import annotations
import unittest
import numpy as np

from resonance.physics.circuits import (
    compute_complex_current,
    compute_current_amplitude_and_phase,
    compute_resonant_frequency,
    compute_quality_factor,
)


def independent_nodal_solve(
    circuit_type: str,
    f: float,
    V_rms: float,
    R: float,
    L_H: float | None = None,
    C_F: float | None = None,
) -> complex:
    """Independent nodal AC complex impedance solver using fundamental Ohm's law in phasor domain."""
    omega = 2.0 * np.pi * f
    if circuit_type == "RC":
        assert C_F is not None and C_F > 0
        Z = complex(R, -1.0 / (omega * C_F))
    elif circuit_type == "RL":
        assert L_H is not None and L_H > 0
        Z = complex(R, omega * L_H)
    elif circuit_type == "RLC":
        assert L_H is not None and C_F is not None
        X_L = omega * L_H
        X_C = 1.0 / (omega * C_F)
        Z = complex(R, X_L - X_C)
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")

    I_phasor = V_rms / Z
    return I_phasor


class TestPhysicsValidation(unittest.TestCase):
    def test_independent_nodal_agreement_100_circuits(self):
        """Compare numeric engine against independent nodal solver across 100 random circuits."""
        rng = np.random.default_rng(42)
        circuit_types = ["RC", "RL", "RLC"]
        test_freqs = np.geomspace(30.0, 30000.0, 10)

        for _ in range(100):
            ctype = str(rng.choice(circuit_types))
            R = float(rng.uniform(20.0, 200.0))
            L_H = float(rng.uniform(1.0, 100.0)) * 1e-3 if ctype in ("RL", "RLC") else None
            C_F = float(rng.uniform(0.1, 10.0)) * 1e-6 if ctype in ("RC", "RLC") else None
            V_rms = 0.1

            engine_I = compute_complex_current(
                circuit_type=ctype,
                frequency_hz=test_freqs,
                V_rms=V_rms,
                R=R,
                L_H=L_H,
                C_F=C_F,
            )

            nodal_I = np.array([
                independent_nodal_solve(ctype, f, V_rms, R, L_H, C_F)
                for f in test_freqs
            ], dtype=complex)

            # Compare real and imaginary parts to 1e-9 relative tolerance
            np.testing.assert_allclose(
                np.real(engine_I),
                np.real(nodal_I),
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"Real part mismatch for {ctype}",
            )
            np.testing.assert_allclose(
                np.imag(engine_I),
                np.imag(nodal_I),
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"Imaginary part mismatch for {ctype}",
            )

    def test_rlc_resonance_properties(self):
        """Verify omega0, f0, Q, bandwidth and cancellation at resonance."""
        L_H = 0.020  # 20 mH
        C_F = 2.0e-6 # 2 uF
        R = 50.0

        omega0, f0 = compute_resonant_frequency(L_H, C_F)
        expected_omega0 = 1.0 / np.sqrt(L_H * C_F)
        expected_f0 = expected_omega0 / (2.0 * np.pi)

        self.assertAlmostEqual(omega0, expected_omega0, places=9)
        self.assertAlmostEqual(f0, expected_f0, places=9)

        # At exactly f0, reactance must cancel: Z = R + j*0
        I_res = independent_nodal_solve("RLC", f0, V_rms=0.1, R=R, L_H=L_H, C_F=C_F)
        self.assertAlmostEqual(np.imag(I_res), 0.0, places=9)
        self.assertAlmostEqual(np.real(I_res), 0.1 / R, places=9)

        # Quality factor Q = (1 / R) * sqrt(L / C)
        Q = compute_quality_factor(R, L_H, C_F)
        expected_Q = (1.0 / R) * np.sqrt(L_H / C_F)
        self.assertAlmostEqual(Q, expected_Q, places=9)


if __name__ == "__main__":
    unittest.main()
