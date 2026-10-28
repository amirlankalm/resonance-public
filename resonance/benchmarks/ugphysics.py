"""UGPhysics Benchmark Suite for Resonance.

Based on UGPhysics: A Comprehensive Undergraduate Physics Reasoning Benchmark (ICML 2025).
Evaluates university-level physics problem solving across 5 core undergraduate disciplines:
1. Classical Mechanics
2. Electromagnetism
3. Quantum Mechanics
4. Thermodynamics & Statistical Physics
5. Optics & Wave Phenomena
"""

from __future__ import annotations
import re
from typing import Dict, Any, List, Optional
from resonance.core.memory import MemoryGuard


UGPHYSICS_PROBLEMS: List[Dict[str, Any]] = [
    # --- 1. Classical Mechanics ---
    {
        "id": "ugphys_mech_01",
        "domain": "Classical Mechanics",
        "topic": "Rotational Dynamics & Angular Momentum",
        "question": (
            "A solid uniform disk of mass M = 4.0 kg and radius R = 0.5 m is rotating freely about its "
            "central axis at an initial angular speed omega_0 = 20.0 rad/s. A small lump of clay of mass "
            "m = 1.0 kg is dropped gently onto the rim of the disk and sticks. Neglecting friction on the "
            "axle, what is the final angular velocity omega_f of the system in rad/s?\n"
            "A) 10.0 rad/s\n"
            "B) 13.3 rad/s\n"
            "C) 15.0 rad/s\n"
            "D) 16.0 rad/s\n"
            "Answer with the correct letter option."
        ),
        "target_option": "B",  # I_disk = 1/2 M R^2 = 0.5 kg m^2. I_clay = m R^2 = 0.25 kg m^2. I_tot = 0.75. L = 0.5 * 20 = 10. omega_f = 10 / 0.75 = 13.33 rad/s.
        "target_value": 13.33,
        "explanation": "Conservation of angular momentum: I_disk * omega_0 = (I_disk + m*R^2) * omega_f. omega_f = (0.5 * 4 * 0.25 * 20) / (0.5 + 0.25) = 10 / 0.75 = 13.33 rad/s (Option B).",
    },
    {
        "id": "ugphys_mech_02",
        "domain": "Classical Mechanics",
        "topic": "Coupled Oscillations",
        "question": (
            "Two identical masses m are connected to each other by a spring of constant k, and each is connected "
            "to a fixed wall by an identical spring of constant k (three springs, two masses in line). "
            "What is the ratio of the frequency of the antisymmetric normal mode to the symmetric normal mode (omega_anti / omega_sym)?\n"
            "A) sqrt(2)\n"
            "B) sqrt(3)\n"
            "C) 2\n"
            "D) 3\n"
            "Answer with the correct letter option."
        ),
        "target_option": "B",  # omega_sym = sqrt(k/m), omega_anti = sqrt(3k/m) -> ratio = sqrt(3)
        "target_value": 1.732,
        "explanation": "Symmetric mode: middle spring is unstretched, omega_sym = sqrt(k/m). Antisymmetric mode: middle spring stretches by 2x, effective restoring force is 3k, omega_anti = sqrt(3k/m). Ratio = sqrt(3) (Option B).",
    },
    {
        "id": "ugphys_mech_03",
        "domain": "Classical Mechanics",
        "topic": "Central Force & Planetary Orbits",
        "question": (
            "A satellite is in a circular orbit of radius r around Earth. A brief thruster burn reduces its mechanical energy "
            "such that it enters an elliptical orbit with periapsis at r/2 and apoapsis at r. "
            "By what factor does the orbital period change compared to the original circular orbit?\n"
            "A) (3/4)^(3/2) ≈ 0.650\n"
            "B) (1/2)^(3/2) ≈ 0.354\n"
            "C) (3/4) = 0.750\n"
            "D) (2/3)^(3/2) ≈ 0.544\n"
            "Answer with the correct letter option."
        ),
        "target_option": "A",  # Semi-major axis a = (r + r/2) / 2 = 0.75 r. Kepler's 3rd law: T \propto a^(3/2) -> (3/4)^(3/2)
        "target_value": 0.650,
        "explanation": "Semi-major axis a = (r + r/2)/2 = 0.75 r. By Kepler's third law T proportional to a^(3/2), so T_ellipse / T_circle = (3/4)^(3/2) ≈ 0.650 (Option A).",
    },

    # --- 2. Electromagnetism ---
    {
        "id": "ugphys_em_01",
        "domain": "Electromagnetism",
        "topic": "Boundary Conditions on Dielectrics",
        "question": (
            "At a planar boundary between two linear, isotropic, homogeneous non-magnetic dielectric media with relative permittivities "
            "eps_1 = 2.0 and eps_2 = 6.0 without free surface charge, the electric field in medium 1 makes an angle theta_1 = 30 degrees "
            "with the normal to the interface. What is the angle theta_2 in degrees that the field in medium 2 makes with the normal?\n"
            "A) 10.9 degrees\n"
            "B) 30.0 degrees\n"
            "C) 60.0 degrees\n"
            "D) 73.9 degrees\n"
            "Answer with the correct letter option."
        ),
        "target_option": "C",  # tan(theta_2) / tan(theta_1) = eps_2 / eps_1 = 6/2 = 3. tan(theta_1) = tan(30) = 1/sqrt(3). tan(theta_2) = 3 / sqrt(3) = sqrt(3) -> theta_2 = 60 deg.
        "target_value": 60.0,
        "explanation": "Boundary conditions: E_parallel is continuous (E_1 sin theta_1 = E_2 sin theta_2) and D_normal is continuous (eps_1 E_1 cos theta_1 = eps_2 E_2 cos theta_2). Dividing gives tan(theta_2)/tan(theta_1) = eps_2/eps_1 = 3. tan(theta_2) = 3 * tan(30°) = sqrt(3) -> theta_2 = 60° (Option C).",
    },
    {
        "id": "ugphys_em_02",
        "domain": "Electromagnetism",
        "topic": "Poynting Vector & Radiation Pressure",
        "question": (
            "A monochromatic plane electromagnetic wave with peak electric field amplitude E_0 = 300 V/m travels in vacuum. "
            "What is the time-averaged Poynting vector magnitude <S> (intensity) in W/m^2?\n"
            "A) 59.7 W/m^2\n"
            "B) 119.4 W/m^2\n"
            "C) 238.8 W/m^2\n"
            "D) 477.5 W/m^2\n"
            "Answer with the correct letter option."
        ),
        "target_option": "B",  # <S> = 1/2 eps_0 c E_0^2 = 0.5 * 8.854e-12 * 3e8 * 90000 = 119.5 W/m^2
        "target_value": 119.4,
        "explanation": "<S> = 1/2 * eps_0 * c * E_0^2 = 0.5 * 8.854e-12 * 3e8 * 300^2 = 119.4 W/m^2 (Option B).",
    },
    {
        "id": "ugphys_em_03",
        "domain": "Electromagnetism",
        "topic": "Resonant Cavities & Quality Factor",
        "question": (
            "A series RLC circuit is driven at its natural resonance frequency f_0 = 10.0 kHz. If the bandwidth between "
            "the half-power (-3 dB) points is Delta f = 200 Hz, what is the Quality Factor Q of the resonator?\n"
            "A) 25\n"
            "B) 50\n"
            "C) 100\n"
            "D) 200\n"
            "Answer with the correct letter option."
        ),
        "target_option": "B",  # Q = f_0 / Delta f = 10000 / 200 = 50
        "target_value": 50.0,
        "explanation": "By definition, Q = f_0 / Delta f = 10,000 / 200 = 50 (Option B).",
    },

    # --- 3. Quantum Mechanics ---
    {
        "id": "ugphys_qm_01",
        "domain": "Quantum Mechanics",
        "topic": "Infinite Potential Well",
        "question": (
            "A particle of mass m is confined to an infinite square well of width L between x = 0 and x = L. "
            "If the particle transitions from the second excited state (n = 3) to the ground state (n = 1), "
            "what is the energy of the emitted photon in units of (pi^2 * hbar^2) / (2 * m * L^2)?\n"
            "A) 4\n"
            "B) 5\n"
            "C) 8\n"
            "D) 9\n"
            "Answer with the correct letter option."
        ),
        "target_option": "C",  # E_n = n^2 E_1. Delta E = (3^2 - 1^2) E_1 = (9 - 1) E_1 = 8 E_1.
        "target_value": 8.0,
        "explanation": "Energy levels: E_n = n^2 * E_1. For n=3 -> n=1 transition: Delta E = (3^2 - 1^2) * E_1 = 8 * E_1 (Option C).",
    },
    {
        "id": "ugphys_qm_02",
        "domain": "Quantum Mechanics",
        "topic": "Quantum Harmonic Oscillator",
        "question": (
            "For a 1D quantum harmonic oscillator of mass m and classical frequency omega in its first excited state |1>, "
            "what is the expectation value of the potential energy <V> = <1| (1/2) m omega^2 x^2 |1>?\n"
            "A) 1/4 hbar * omega\n"
            "B) 1/2 hbar * omega\n"
            "C) 3/4 hbar * omega\n"
            "D) 3/2 hbar * omega\n"
            "Answer with the correct letter option."
        ),
        "target_option": "C",  # By Virial theorem for harmonic oscillator, <T> = <V> = 1/2 E_n. For n=1, E_1 = (1 + 1/2) hbar omega = 3/2 hbar omega. <V> = 3/4 hbar omega.
        "target_value": 0.75,
        "explanation": "For any harmonic oscillator state, by the Virial Theorem <T> = <V> = E_n / 2. Here n=1, so E_1 = 3/2 hbar * omega, which implies <V> = 3/4 hbar * omega (Option C).",
    },
    {
        "id": "ugphys_qm_03",
        "domain": "Quantum Mechanics",
        "topic": "de Broglie Wavelength & Relativity",
        "question": (
            "An electron (rest mass m_e = 0.511 MeV/c^2) is accelerated from rest through a potential difference V = 150 V. "
            "What is its non-relativistic de Broglie wavelength lambda in Angstroms (1 A = 0.1 nm)?\n"
            "A) 0.50 A\n"
            "B) 1.00 A\n"
            "C) 1.23 A\n"
            "D) 2.46 A\n"
            "Answer with the correct letter option."
        ),
        "target_option": "B",  # lambda = sqrt(150 / V) A = sqrt(150/150) = 1.00 A
        "target_value": 1.00,
        "explanation": "lambda = h / p = h / sqrt(2 m_e e V) ≈ sqrt(150 / V) Angstroms = sqrt(150/150) = 1.00 Angstrom (Option B).",
    },

    # --- 4. Thermodynamics & Statistical Physics ---
    {
        "id": "ugphys_thermo_01",
        "domain": "Thermodynamics",
        "topic": "Carnot Cycle & Entropy",
        "question": (
            "A reversible Carnot heat engine operates between a hot reservoir at T_H = 600 K and a cold reservoir at T_C = 300 K. "
            "In each cycle, it absorbs Q_H = 1200 J of heat from the hot reservoir. What is the net mechanical work W produced "
            "per cycle and the change in total entropy of the universe Delta S_univ?\n"
            "A) W = 600 J, Delta S_univ = 0 J/K\n"
            "B) W = 600 J, Delta S_univ = 2 J/K\n"
            "C) W = 400 J, Delta S_univ = 0 J/K\n"
            "D) W = 900 J, Delta S_univ = -1 J/K\n"
            "Answer with the correct letter option."
        ),
        "target_option": "A",  # eta = 1 - 300/600 = 0.5. W = 0.5 * 1200 = 600 J. Reversible cycle -> Delta S_univ = 0.
        "target_value": 600.0,
        "explanation": "Efficiency eta = 1 - T_C/T_H = 1 - 300/600 = 0.50. Work W = eta * Q_H = 600 J. Because the process is reversible, Delta S_univ = 0 (Option A).",
    },
    {
        "id": "ugphys_thermo_02",
        "domain": "Thermodynamics",
        "topic": "Partition Function & Two-State System",
        "question": (
            "A system of N non-interacting particles has two non-degenerate energy levels: 0 and epsilon. "
            "In thermal equilibrium at temperature T (with beta = 1/(k_B T)), what is the average total internal energy U of the system?\n"
            "A) N * epsilon * e^(-beta*epsilon)\n"
            "B) N * epsilon / (1 + e^(beta*epsilon))\n"
            "C) N * epsilon / (1 + e^(-beta*epsilon))\n"
            "D) N * epsilon * (1 - e^(-beta*epsilon))\n"
            "Answer with the correct letter option."
        ),
        "target_option": "B",  # Single particle Z = 1 + e^(-beta*epsilon). <E> = epsilon * e^(-beta*epsilon) / (1 + e^(-beta*epsilon)) = epsilon / (e^(beta*epsilon) + 1). U = N <E>.
        "target_value": 1.0,
        "explanation": "Single-particle partition function Z = 1 + e^(-beta*epsilon). Average energy <epsilon> = (0 + epsilon * e^(-beta*epsilon)) / (1 + e^(-beta*epsilon)) = epsilon / (e^(beta*epsilon) + 1). Total U = N * epsilon / (1 + e^(beta*epsilon)) (Option B).",
    },
    {
        "id": "ugphys_thermo_03",
        "domain": "Thermodynamics",
        "topic": "Adiabatic Expansion",
        "question": (
            "A monatomic ideal gas (gamma = 5/3) expands adiabatically and reversibly from initial volume V_0 to 8 * V_0. "
            "If the initial temperature is T_0, what is the final temperature T_f?\n"
            "A) T_0 / 2\n"
            "B) T_0 / 4\n"
            "C) T_0 / 8\n"
            "D) T_0 / 16\n"
            "Answer with the correct letter option."
        ),
        "target_option": "B",  # T V^(gamma-1) = const. gamma - 1 = 2/3. (8)^(2/3) = (2^3)^(2/3) = 4. T_f = T_0 / 4.
        "target_value": 0.25,
        "explanation": "For reversible adiabatic expansion, T * V^(gamma - 1) = constant. gamma - 1 = 5/3 - 1 = 2/3. T_f / T_0 = (V_0 / 8 V_0)^(2/3) = (1/8)^(2/3) = 1/4 (Option B).",
    },

    # --- 5. Optics & Wave Phenomena ---
    {
        "id": "ugphys_optics_01",
        "domain": "Optics",
        "topic": "Brewster's Angle & Polarization",
        "question": (
            "Unpolarized light in air (n_1 = 1.0) reflects from a flat glass surface with index of refraction n_2 = 1.732 (sqrt(3)). "
            "At what incident angle theta_p (Brewster's angle) is the reflected light 100% linearly polarized?\n"
            "A) 30.0 degrees\n"
            "B) 45.0 degrees\n"
            "C) 60.0 degrees\n"
            "D) 75.0 degrees\n"
            "Answer with the correct letter option."
        ),
        "target_option": "C",  # tan(theta_p) = n_2 / n_1 = sqrt(3) -> theta_p = 60 deg.
        "target_value": 60.0,
        "explanation": "Brewster's law: tan(theta_p) = n_2 / n_1 = 1.732 = sqrt(3) -> theta_p = 60.0° (Option C).",
    },
    {
        "id": "ugphys_optics_02",
        "domain": "Optics",
        "topic": "Thin Film Interference",
        "question": (
            "A thin oil film (n = 1.40) floating on water (n = 1.33) is illuminated at normal incidence by white light. "
            "What is the minimum non-zero thickness t in nanometers that produces constructive interference in reflection "
            "for light of wavelength lambda = 560 nm in air?\n"
            "A) 100 nm\n"
            "B) 140 nm\n"
            "C) 200 nm\n"
            "D) 280 nm\n"
            "Answer with the correct letter option."
        ),
        "target_option": "A",  # Air (1.0) -> Oil (1.4): phase shift pi. Oil (1.4) -> Water (1.33): no phase shift. Net relative phase shift = pi. Constructive: 2 n t = (m + 1/2) lambda. For m=0: 2 * 1.40 * t = 0.5 * 560 -> 2.8 t = 280 -> t = 100 nm.
        "target_value": 100.0,
        "explanation": "Phase change at top boundary (n=1.0 to 1.40) is pi; at bottom boundary (n=1.40 to 1.33) is 0. Constructive condition: 2 * n * t = lambda / 2 -> t = lambda / (4 * n) = 560 / (4 * 1.40) = 100 nm (Option A).",
    },
    {
        "id": "ugphys_optics_03",
        "domain": "Optics",
        "topic": "Malus' Law & Cascade Polarizers",
        "question": (
            "Unpolarized light of intensity I_0 passes through three consecutive ideal linear polarizers. "
            "The transmission axis of the second polarizer is oriented at 45 degrees relative to the first, "
            "and the third polarizer is oriented at 90 degrees relative to the first (crossed with the first). "
            "What fraction of the initial intensity I_0 emerges from the third polarizer (I_out / I_0)?\n"
            "A) 0\n"
            "B) 1/16 ≈ 0.0625\n"
            "C) 1/8 = 0.125\n"
            "D) 1/4 = 0.25\n"
            "Answer with the correct letter option."
        ),
        "target_option": "C",  # After 1st: I_1 = 1/2 I_0. After 2nd: I_2 = I_1 cos^2(45) = 1/2 I_0 * (1/2) = 1/4 I_0. After 3rd: angle between 2nd and 3rd is 45 deg: I_3 = I_2 cos^2(45) = 1/4 I_0 * 1/2 = 1/8 I_0 = 0.125.
        "target_value": 0.125,
        "explanation": "After polarizer 1: I_1 = I_0 / 2. After polarizer 2 (at 45°): I_2 = I_1 * cos^2(45°) = I_0 / 4. After polarizer 3 (at 45° to polarizer 2): I_3 = I_2 * cos^2(45°) = I_0 / 8 = 0.125 (Option C).",
    },
]


def extract_multiple_choice_option(text: str) -> Optional[str]:
    """Extract chosen option A, B, C, or D from model text output."""
    # Look for explicit patterns like "Answer: A" or "Option B"
    explicit = re.findall(r"(?:answer|option|choice)(?:\s*(?:is|:|=)?\s*)([A-D])\b", text, re.IGNORECASE)
    if explicit:
        return explicit[-1].upper()

    # Look for stand-alone bracketed or parenthesized letters e.g. (B) or [B]
    parenthesized = re.findall(r"[\(\[\*]([A-D])[\)\]\*]", text)
    if parenthesized:
        return parenthesized[-1].upper()

    # Look for capital letter preceded by space/newline and followed by period or paren
    trailing = re.findall(r"\b([A-D])(?:\)|\.|\s*$)", text)
    if trailing:
        return trailing[-1].upper()

    return None


def evaluate_ugphysics(agent: Any, problems: List[Dict[str, Any]] = UGPHYSICS_PROBLEMS) -> Dict[str, Any]:
    """Evaluate an agent on the UGPhysics undergraduate physics benchmark."""
    system_prompt = (
        "You are an expert university physics professor. Solve the following undergraduate physics problem "
        "step-by-step with rigorous physical principles, formulas, and boundary conditions. "
        "Conclude with your final answer formatted clearly as 'Answer: <OPTION>' (A, B, C, or D)."
    )

    domain_scores: Dict[str, Dict[str, int]] = {}
    problem_results = []
    correct_count = 0

    for prob in problems:
        domain = prob["domain"]
        if domain not in domain_scores:
            domain_scores[domain] = {"correct": 0, "total": 0}
        domain_scores[domain]["total"] += 1

        prompt = prob["question"]
        resp = agent.generate_response(prompt, system_prompt=system_prompt, max_new_tokens=256)
        MemoryGuard.cleanup_memory()

        predicted_option = extract_multiple_choice_option(resp)
        target_option = prob["target_option"]
        is_correct = (predicted_option == target_option)

        if is_correct:
            correct_count += 1
            domain_scores[domain]["correct"] += 1

        problem_results.append({
            "id": prob["id"],
            "domain": domain,
            "topic": prob["topic"],
            "target_option": target_option,
            "predicted_option": predicted_option,
            "is_correct": is_correct,
            "model_response": resp[:250] + ("..." if len(resp) > 250 else ""),
        })

    total = len(problems)
    accuracy = correct_count / total if total > 0 else 0.0

    domain_summary = {}
    for d, s in domain_scores.items():
        domain_summary[d] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0.0,
        }

    return {
        "benchmark_name": "UGPhysics-ICML2025",
        "total_problems": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "domain_breakdown": domain_summary,
        "problem_results": problem_results,
    }
