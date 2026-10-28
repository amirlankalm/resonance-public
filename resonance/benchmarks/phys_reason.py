"""PhysReason Benchmark Suite for Resonance.

Based on PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning (ACL 2025).
Evaluates complex multi-step reasoning across:
- Physics Theorem Application
- Physics Process Understanding
- Physics Condition Analysis
- Numerical Calculation
Across Easy, Medium, and Hard difficulty levels.
"""

from __future__ import annotations
import re
from typing import Dict, Any, List, Optional
from resonance.core.memory import MemoryGuard


PHYSREASON_PROBLEMS: List[Dict[str, Any]] = [
    # --- Easy ---
    {
        "id": "physreason_easy_01",
        "difficulty": "Easy",
        "domain": "Thermodynamics",
        "theorem": "Ideal Gas Law & Isobaric Expansion",
        "question": (
            "An ideal gas occupies volume V_1 = 2.0 L at temperature T_1 = 300 K under constant pressure P = 1.0 atm. "
            "If the gas is heated isobarically until its temperature reaches T_2 = 450 K, what is its final volume V_2 in Liters?"
        ),
        "target_value": 3.0,
        "tolerance": 0.03,
        "unit": "L",
        "solution_steps": [
            "Identify isobaric condition (P = const).",
            "Apply Charles's law: V_1 / T_1 = V_2 / T_2.",
            "Solve for V_2: V_2 = V_1 * (T_2 / T_1) = 2.0 * (450 / 300) = 3.0 L.",
        ],
    },
    {
        "id": "physreason_easy_02",
        "difficulty": "Easy",
        "domain": "Mechanics",
        "theorem": "Conservation of Mechanical Energy",
        "question": (
            "A block of mass m = 2.0 kg is released from rest from the top of a frictionless curved incline of height h = 5.0 m. "
            "Taking acceleration due to gravity g = 9.8 m/s^2, what is the speed of the block in m/s when it reaches the bottom?"
        ),
        "target_value": 9.9,  # sqrt(2 * 9.8 * 5) = sqrt(98) ≈ 9.899 -> 9.9 m/s
        "tolerance": 0.03,
        "unit": "m/s",
        "solution_steps": [
            "Recognize no non-conservative work (frictionless).",
            "Apply conservation of energy: m*g*h = (1/2)*m*v^2.",
            "Solve for v = sqrt(2*g*h) = sqrt(2 * 9.8 * 5.0) = sqrt(98) = 9.90 m/s.",
        ],
    },
    {
        "id": "physreason_easy_03",
        "difficulty": "Easy",
        "domain": "Electromagnetism",
        "theorem": "Ohm's Law & Power Dissipation",
        "question": (
            "A constant voltage source of V = 12.0 V is connected across an ohmic resistor R = 8.0 Ohms. "
            "What is the electrical power dissipated as heat in the resistor in Watts (W)?"
        ),
        "target_value": 18.0,  # P = V^2 / R = 144 / 8 = 18.0 W
        "tolerance": 0.02,
        "unit": "W",
        "solution_steps": [
            "Apply Joule's law / Ohm's law: P = V^2 / R.",
            "Calculate P = (12.0)^2 / 8.0 = 144 / 8.0 = 18.0 W.",
        ],
    },

    # --- Medium ---
    {
        "id": "physreason_med_01",
        "difficulty": "Medium",
        "domain": "Electromagnetism",
        "theorem": "Faraday's Law of Induction & Lenz's Law",
        "question": (
            "A circular wire loop of radius r = 0.10 m and total resistance R = 2.0 Ohms lies flat in the x-y plane. "
            "A uniform magnetic field B(t) perpendicular to the loop changes with time according to B(t) = 0.5 * t^2 (Tesla), "
            "where t is in seconds. What is the induced electric current magnitude I in the loop in amperes (A) at time t = 4.0 s?"
        ),
        "target_value": 0.0628,  # Flux Phi = B * A = 0.5 t^2 * pi * r^2. EMF = dPhi/dt = t * pi * r^2. At t=4, EMF = 4 * pi * 0.01 = 0.04 * pi = 0.12566 V. I = EMF / R = 0.12566 / 2 = 0.06283 A.
        "tolerance": 0.05,
        "unit": "A",
        "solution_steps": [
            "Compute area A = pi * r^2 = 0.01 * pi m^2.",
            "Compute magnetic flux Phi(t) = B(t) * A = 0.5 * pi * r^2 * t^2.",
            "Apply Faraday's law: |EMF| = dPhi/dt = pi * r^2 * t.",
            "Evaluate EMF at t = 4.0 s: EMF = 0.04 * pi = 0.1257 V.",
            "Compute current I = EMF / R = 0.1257 / 2.0 = 0.0628 A (62.8 mA).",
        ],
    },
    {
        "id": "physreason_med_02",
        "difficulty": "Medium",
        "domain": "Optics",
        "theorem": "Double-Slit Interference & Path Difference",
        "question": (
            "In a Young's double-slit experiment, monochromatic light of wavelength lambda = 500 nm illuminates two slits "
            "separated by distance d = 0.20 mm. The interference pattern is viewed on a screen at distance L = 1.60 m. "
            "What is the distance on the screen between the central maximum and the third dark fringe (m = 2) in millimeters (mm)?"
        ),
        "target_value": 10.0,  # y_dark = (m + 1/2) * lambda * L / d. For 3rd dark fringe, m = 2 -> (2.5) * 500e-9 * 1.6 / 0.2e-3 = 2.5 * 8e-7 / 0.2e-3 * 1.6 = 2.5 * 4e-3 = 0.010 m = 10.0 mm.
        "tolerance": 0.03,
        "unit": "mm",
        "solution_steps": [
            "Path difference for destructive interference: Delta r = (m + 1/2) * lambda.",
            "For the third dark fringe, m = 2 (fringes at m=0, 1, 2). Path difference = 2.5 * lambda.",
            "Position on screen y = (2.5 * lambda * L) / d.",
            "Compute y = (2.5 * 500e-9 * 1.6) / 0.20e-3 = 0.010 m = 10.0 mm.",
        ],
    },
    {
        "id": "physreason_med_03",
        "difficulty": "Medium",
        "domain": "Classical Mechanics",
        "theorem": "Damped Harmonic Motion & Quality Factor",
        "question": (
            "A lightly damped oscillator consists of mass m = 0.5 kg and spring constant k = 200 N/m. "
            "The damping coefficient is b = 0.20 N*s/m. What is the Quality Factor Q of this mechanical oscillator rounded to an integer?"
        ),
        "target_value": 50.0,  # omega_0 = sqrt(k/m) = sqrt(200 / 0.5) = 20 rad/s. Q = m * omega_0 / b = 0.5 * 20 / 0.2 = 10 / 0.2 = 50.
        "tolerance": 0.04,
        "unit": "dimensionless",
        "solution_steps": [
            "Compute natural angular frequency omega_0 = sqrt(k/m) = sqrt(200 / 0.5) = 20 rad/s.",
            "Recall definition of mechanical Q: Q = (m * omega_0) / b = sqrt(m * k) / b.",
            "Calculate Q = (0.5 * 20) / 0.20 = 10 / 0.20 = 50.",
        ],
    },
    {
        "id": "physreason_med_04",
        "difficulty": "Medium",
        "domain": "Electromagnetism",
        "theorem": "Series RLC Resonance Impedance",
        "question": (
            "An AC series RLC circuit has resistance R = 40 Ohms, inductance L = 10 mH, and capacitance C = 1.0 microfarad. "
            "What is the magnitude of the circuit impedance |Z| in Ohms when driven at exactly twice its natural resonance frequency (f = 2 * f_0)?"
        ),
        "target_value": 155.2,  # omega_0 = 1/sqrt(LC) = 1/sqrt(1e-2 * 1e-6) = 1/sqrt(1e-8) = 10000 rad/s. At omega = 2*omega_0 = 20000: X_L = omega*L = 20000 * 0.01 = 200. X_C = 1/(omega*C) = 1/(20000 * 1e-6) = 50. Net reactance X = 200 - 50 = 150 Ohms. |Z| = sqrt(40^2 + 150^2) = sqrt(1600 + 22500) = sqrt(24100) = 155.24 Ohms.
        "tolerance": 0.05,
        "unit": "Ohms",
        "solution_steps": [
            "Compute resonant frequency omega_0 = 1 / sqrt(LC) = 1 / sqrt(10e-3 * 1e-6) = 10,000 rad/s.",
            "Set driven frequency omega = 2 * omega_0 = 20,000 rad/s.",
            "Compute X_L = omega * L = 20000 * 0.010 = 200 Ohms.",
            "Compute X_C = 1 / (omega * C) = 1 / (20000 * 1e-6) = 50 Ohms.",
            "Compute total impedance |Z| = sqrt(R^2 + (X_L - X_C)^2) = sqrt(40^2 + 150^2) = 155.2 Ohms.",
        ],
    },

    # --- Hard ---
    {
        "id": "physreason_hard_01",
        "difficulty": "Hard",
        "domain": "Modern Physics",
        "theorem": "Relativistic Doppler Effect & Redshift",
        "question": (
            "A distant galaxy recedes directly away from Earth along the line of sight with speed v = 0.60 c. "
            "A rest-frame spectral emission line of wavelength lambda_0 = 400 nm is observed on Earth. "
            "What is the observed wavelength lambda_obs in nanometers (nm) taking into account special relativistic time dilation and Doppler shift?"
        ),
        "target_value": 800.0,  # lambda_obs = lambda_0 * sqrt((1 + beta) / (1 - beta)) = 400 * sqrt(1.6 / 0.4) = 400 * sqrt(4) = 800 nm.
        "tolerance": 0.02,
        "unit": "nm",
        "solution_steps": [
            "Identify relativistic longitudinal Doppler formula: f_obs = f_0 * sqrt((1 - beta)/(1 + beta)).",
            "Invert to wavelength: lambda_obs = lambda_0 * sqrt((1 + beta)/(1 - beta)).",
            "With beta = 0.60: (1 + beta)/(1 - beta) = 1.60 / 0.40 = 4.0.",
            "Take square root: sqrt(4.0) = 2.0.",
            "Compute lambda_obs = 400 nm * 2.0 = 800 nm.",
        ],
    },
    {
        "id": "physreason_hard_02",
        "difficulty": "Hard",
        "domain": "Thermodynamics",
        "theorem": "Clapeyron-Clausius Vapor Pressure Equation",
        "question": (
            "The latent heat of vaporization of water is L_v = 40.7 kJ/mol and water boils at T_1 = 373.15 K under P_1 = 1.00 atm. "
            "Assuming ideal gas behavior for water vapor and constant L_v (gas constant R = 8.314 J/(mol*K)), "
            "at what temperature T_2 in Kelvin does water boil on a high mountain summit where atmospheric pressure is P_2 = 0.70 atm? "
            "Round to one decimal place."
        ),
        "target_value": 363.3,  # ln(P2/P1) = -L/R * (1/T2 - 1/T1). ln(0.70) = -0.35667. -0.35667 = -40700 / 8.314 * (1/T2 - 1/373.15) = -4895.36 * (1/T2 - 0.0026799). 1/T2 - 0.0026799 = 7.2859e-5 -> 1/T2 = 0.00275276 -> T2 = 363.27 K.
        "tolerance": 0.02,
        "unit": "K",
        "solution_steps": [
            "Apply Clausius-Clapeyron integrated equation: ln(P_2 / P_1) = -(L_v / R) * (1/T_2 - 1/T_1).",
            "Calculate ln(0.70) = -0.3567.",
            "Calculate L_v / R = 40700 / 8.314 = 4895.36 K.",
            "Solve for 1/T_2: 1/T_2 = 1/373.15 - (-0.3567 / 4895.36) = 0.002680 + 0.00007286 = 0.0027528 K^-1.",
            "Invert to find T_2 = 1 / 0.0027528 = 363.3 K.",
        ],
    },
    {
        "id": "physreason_hard_03",
        "difficulty": "Hard",
        "domain": "Quantum Mechanics",
        "theorem": "Finite Square Well Tunneling Transmission",
        "question": (
            "A monoenergetic beam of electrons (energy E = 2.0 eV) impinges upon a rectangular potential barrier of height V_0 = 10.0 eV "
            "and width L = 0.20 nm. The barrier attenuation coefficient kappa = sqrt(2 m (V_0 - E)) / hbar is approximately 14.5 nm^-1. "
            "In the wide-barrier approximation T ≈ 16 (E/V_0)(1 - E/V_0) e^(-2 kappa L), what is the transmission coefficient T in scientific notation? "
            "Give the exponent n where T = C * 10^n."
        ),
        "target_value": -3.0,  # 2*kappa*L = 2 * 14.5 * 0.2 = 5.8. e^(-5.8) = 0.003027. Prefactor: 16 * (2/10) * (8/10) = 16 * 0.16 = 2.56. T = 2.56 * 0.003027 = 0.00775 = 7.75 * 10^-3. Exponent n = -3.
        "tolerance": 0.3,
        "unit": "order of magnitude",
        "solution_steps": [
            "Compute barrier height difference: V_0 - E = 8.0 eV.",
            "Compute 2 * kappa * L = 2 * 14.5 nm^-1 * 0.20 nm = 5.80.",
            "Compute exponential decay: e^(-5.80) ≈ 3.03 * 10^-3.",
            "Compute prefactor: 16 * (E / V_0) * (1 - E / V_0) = 16 * 0.2 * 0.8 = 2.56.",
            "Multiply: T = 2.56 * 3.03 * 10^-3 = 7.75 * 10^-3. Order of magnitude is 10^-3 (exponent = -3).",
        ],
    },
]


def extract_numerical_answer(text: str) -> Optional[float]:
    """Extract final numerical answer from reasoning trace."""
    # Look for explicit Answer: patterns
    explicit = re.findall(r"(?:answer|result|value)(?:\s*(?:is|:|=)?\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text, re.IGNORECASE)
    if explicit:
        try:
            return float(explicit[-1])
        except ValueError:
            pass

    # Look for last boxed or bracketed number
    boxed = re.findall(r"\\boxed\{([-+]?\d*\.?\d+)\}", text)
    if boxed:
        try:
            return float(boxed[-1])
        except ValueError:
            pass

    # Fall back to trailing number
    trailing = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if trailing:
        try:
            return float(trailing[-1])
        except ValueError:
            pass

    return None


def evaluate_physreason(agent: Any, problems: List[Dict[str, Any]] = PHYSREASON_PROBLEMS) -> Dict[str, Any]:
    """Evaluate an agent on the PhysReason multi-step benchmark."""
    system_prompt = (
        "You are an expert theoretical physicist solving complex physics problems. "
        "Structure your response in clear numbered steps: "
        "1. Physical Theorem Application\n"
        "2. Physics Process Understanding\n"
        "3. Boundary Condition Analysis\n"
        "4. Calculation\n"
        "Conclude with your final answer as: 'Answer: <NUMBER>'."
    )

    diff_scores: Dict[str, Dict[str, int]] = {}
    problem_results = []
    correct_count = 0

    for prob in problems:
        diff = prob["difficulty"]
        if diff not in diff_scores:
            diff_scores[diff] = {"correct": 0, "total": 0}
        diff_scores[diff]["total"] += 1

        prompt = prob["question"]
        resp = agent.generate_response(prompt, system_prompt=system_prompt, max_new_tokens=300)
        MemoryGuard.cleanup_memory()

        pred_val = extract_numerical_answer(resp)
        target = prob["target_value"]
        tol = prob["tolerance"]

        is_correct = False
        rel_err = float("inf")
        if pred_val is not None:
            rel_err = abs(pred_val - target) / max(abs(target), 1e-9)
            if rel_err <= tol:
                is_correct = True
                correct_count += 1
                diff_scores[diff]["correct"] += 1

        problem_results.append({
            "id": prob["id"],
            "difficulty": diff,
            "domain": prob["domain"],
            "theorem": prob["theorem"],
            "target": target,
            "predicted": pred_val,
            "relative_error": round(rel_err, 4) if pred_val is not None else None,
            "is_correct": is_correct,
            "model_response": resp[:250] + ("..." if len(resp) > 250 else ""),
        })

    total = len(problems)
    accuracy = correct_count / total if total > 0 else 0.0

    diff_summary = {}
    for d, s in diff_scores.items():
        diff_summary[d] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0.0,
        }

    return {
        "benchmark_name": "PhysReason-ACL2025",
        "total_problems": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "difficulty_breakdown": diff_summary,
        "problem_results": problem_results,
    }
