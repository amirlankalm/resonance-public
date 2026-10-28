"""Law-Family Identification Smoke Test for Resonance.

Inspired by LLM-SRBench (ICML 2024): evaluates whether LLMs can identify governing physical
law families from tabular experimental (x, y) data via multiple-choice candidate formulas.
Note: This is a discrete law-family classification smoke test, not unconstrained symbolic regression.
"""

from __future__ import annotations
import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from resonance.core.memory import MemoryGuard


SR_PROBLEMS: List[Dict[str, Any]] = [
    {
        "id": "sr_kepler",
        "name": "Kepler's Third Law of Planetary Motion",
        "true_formula": "T = k * a^(3/2)",
        "law_family": "Power Law (exponent 1.5)",
        "x_label": "Semi-major axis a (AU)",
        "y_label": "Orbital period T (years)",
        "data_points": [(1.0, 1.01), (1.52, 1.88), (2.2, 3.26), (3.5, 6.55), (5.2, 11.86), (9.5, 29.3)],
        "target_family_option": "B",
        "question": (
            "Given experimental astronomical data of planetary semi-major axis 'a' and orbital period 'T':\n"
            "a: [1.0, 1.52, 2.2, 3.5, 5.2, 9.5]\n"
            "T: [1.01, 1.88, 3.26, 6.55, 11.86, 29.3]\n"
            "Which underlying mathematical law best describes the functional relationship T(a)?\n"
            "A) Linear: T = k * a\n"
            "B) Power law: T = k * a^(3/2)\n"
            "C) Exponential: T = T_0 * exp(a / a_0)\n"
            "D) Inverse square: T = k / a^2\n"
            "Answer with the correct letter option."
        ),
    },
    {
        "id": "sr_malus",
        "name": "Malus' Law of Light Polarization",
        "true_formula": "I = I_0 * cos^2(theta)",
        "law_family": "Cosine Squared",
        "x_label": "Polarizer angle theta (deg)",
        "y_label": "Transmitted intensity I",
        "data_points": [(0.0, 1.00), (30.0, 0.74), (45.0, 0.49), (60.0, 0.26), (90.0, 0.01)],
        "target_family_option": "C",
        "question": (
            "Given optical photometer measurements of transmitted light intensity 'I' versus relative polarizer angle 'theta' (degrees):\n"
            "theta: [0°, 30°, 45°, 60°, 90°]\n"
            "I: [1.00, 0.74, 0.49, 0.26, 0.01]\n"
            "Which functional form best characterizes this physical transmission law?\n"
            "A) Linear: I = I_0 * (1 - theta / 90)\n"
            "B) Sinusoidal: I = I_0 * sin(theta)\n"
            "C) Cosine squared: I = I_0 * cos^2(theta)\n"
            "D) Gaussian: I = I_0 * exp(-theta^2 / 2 sigma^2)\n"
            "Answer with the correct letter option."
        ),
    },
    {
        "id": "sr_stefan",
        "name": "Stefan-Boltzmann Thermal Radiation Law",
        "true_formula": "P = sigma * A * T^4",
        "law_family": "Quartic Power Law (T^4)",
        "x_label": "Temperature T (K)",
        "y_label": "Radiated power P (W)",
        "data_points": [(300.0, 459.0), (400.0, 1450.0), (500.0, 3540.0), (600.0, 7350.0)],
        "target_family_option": "D",
        "question": (
            "Given thermal calorimeter data of radiated power 'P' versus absolute temperature 'T' (Kelvin):\n"
            "T (K): [300, 400, 500, 600]\n"
            "P (W): [459, 1450, 3540, 7350]\n"
            "Doubling the temperature increases the radiated power by a factor of approximately 16. Which equation governs this phenomenon?\n"
            "A) P = k * T\n"
            "B) P = k * T^2\n"
            "C) P = k * T^3\n"
            "D) P = sigma * A * T^4 (Quartic Law)\n"
            "Answer with the correct letter option."
        ),
    },
    {
        "id": "sr_rc_discharge",
        "name": "Capacitor Transient RC Discharge",
        "true_formula": "V(t) = V_0 * exp(-t / (R*C))",
        "law_family": "Exponential Decay",
        "x_label": "Time t (ms)",
        "y_label": "Voltage V (Volts)",
        "data_points": [(0.0, 10.0), (1.0, 6.07), (2.0, 3.68), (3.0, 2.23), (5.0, 0.82)],
        "target_family_option": "A",
        "question": (
            "Given oscilloscope measurements of discharging capacitor voltage 'V' versus time 't' (ms):\n"
            "t: [0.0, 1.0, 2.0, 3.0, 5.0]\n"
            "V: [10.0, 6.07, 3.68, 2.23, 0.82]\n"
            "Notice that at t = 2.0 ms, V drops to approximately V_0 / e ≈ 3.68 V. What is the governing mathematical model?\n"
            "A) Exponential decay: V(t) = V_0 * exp(-t / tau)\n"
            "B) Logarithmic decay: V(t) = V_0 - k * ln(t + 1)\n"
            "C) Power law: V(t) = V_0 / t^2\n"
            "D) Linear ramp: V(t) = V_0 - alpha * t\n"
            "Answer with the correct letter option."
        ),
    },
    {
        "id": "sr_coulomb",
        "name": "Coulomb's Electrostatic Force Law",
        "true_formula": "F = (1 / 4 pi eps_0) * (q1 * q2 / r^2)",
        "law_family": "Inverse Square Law (1 / r^2)",
        "x_label": "Separation distance r (m)",
        "y_label": "Electrostatic force F (N)",
        "data_points": [(0.1, 9.0), (0.2, 2.25), (0.3, 1.00), (0.4, 0.56)],
        "target_family_option": "B",
        "question": (
            "Given torsion balance measurements of electrostatic force 'F' between two point charges versus distance 'r':\n"
            "r (m): [0.10, 0.20, 0.30, 0.40]\n"
            "F (N): [9.00, 2.25, 1.00, 0.56]\n"
            "When r doubles (from 0.1 to 0.2 m), force decreases by a factor of 4. Which equation governs this system?\n"
            "A) F = k / r\n"
            "B) F = k / r^2 (Inverse Square Law)\n"
            "C) F = k / r^3 (Dipole Law)\n"
            "D) F = k * exp(-r / r_0)\n"
            "Answer with the correct letter option."
        ),
    },
    {
        "id": "sr_resonant_freq",
        "name": "Circuit Resonance Frequency vs Inductance",
        "true_formula": "omega_0 = 1 / sqrt(L * C)",
        "law_family": "Inverse Square Root (1 / sqrt(L))",
        "x_label": "Inductance L (mH)",
        "y_label": "Resonant angular frequency omega_0 (krad/s)",
        "data_points": [(1.0, 100.0), (4.0, 50.0), (9.0, 33.3), (16.0, 25.0)],
        "target_family_option": "C",
        "question": (
            "Given laboratory spectroscopy data of circuit resonance frequency 'omega_0' versus series inductance 'L' for a fixed capacitor:\n"
            "L (mH): [1.0, 4.0, 9.0, 16.0]\n"
            "omega_0 (krad/s): [100.0, 50.0, 33.3, 25.0]\n"
            "Notice that multiplying L by 4 cuts omega_0 in half. Which functional form governs this circuit resonance?\n"
            "A) omega_0 = k / L\n"
            "B) omega_0 = k * sqrt(L)\n"
            "C) omega_0 = k / sqrt(L) (Inverse Square Root)\n"
            "D) omega_0 = k * L^2\n"
            "Answer with the correct letter option."
        ),
    },
]


def extract_sr_option(text: str) -> Optional[str]:
    """Extract chosen option A, B, C, or D from model text output."""
    m = re.findall(r"(?:answer|option|choice)(?:\s*(?:is|:|=)?\s*)([A-D])\b", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m2 = re.findall(r"[\(\[\*]([A-D])[\)\]\*]", text)
    if m2:
        return m2[-1].upper()
    m3 = re.findall(r"\b([A-D])(?:\)|\.|\s*$)", text)
    if m3:
        return m3[-1].upper()
    return None


def evaluate_llm_srbench(agent: Any, problems: List[Dict[str, Any]] = SR_PROBLEMS) -> Dict[str, Any]:
    """Evaluate an agent on LLM-SRBench scientific equation discovery benchmark."""
    system_prompt = (
        "You are an expert computational physicist specialized in symbolic regression and scientific law discovery. "
        "Analyze the provided noisy experimental data table, test candidate functional relationships, and identify "
        "the exact mathematical law. Conclude your answer clearly as 'Answer: <OPTION>' (A, B, C, or D)."
    )

    results = []
    correct_count = 0

    for prob in problems:
        prompt = prob["question"]
        resp = agent.generate_response(prompt, system_prompt=system_prompt, max_new_tokens=200)
        MemoryGuard.cleanup_memory()

        pred_opt = extract_sr_option(resp)
        target_opt = prob["target_family_option"]
        is_correct = (pred_opt == target_opt)

        if is_correct:
            correct_count += 1

        results.append({
            "id": prob["id"],
            "name": prob["name"],
            "target_option": target_opt,
            "predicted_option": pred_opt,
            "is_correct": is_correct,
            "true_formula": prob["true_formula"],
            "model_response": resp[:250] + ("..." if len(resp) > 250 else ""),
        })

    total = len(problems)
    accuracy = correct_count / total if total > 0 else 0.0

    return {
        "benchmark_name": "Law-Family Identification (inspired by LLM-SRBench)",
        "total_problems": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "problem_results": results,
    }
