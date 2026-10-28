"""BoxingGym Benchmark Suite for Resonance.

Based on BoxingGym: Automated Experimental Design and Model Discovery for AI Agents (Stanford).
"To understand a system, you must perturb it" — George E.P. Box.

Evaluates an AI agent's ability to:
1. Actively perturb an unknown black-box physical system under an experimental budget.
2. Observe noisy responses and form an internal predictive representation.
3. Generalize to predict responses at held-out test conditions (system identification NRMSE).
"""

from __future__ import annotations
import json
import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from resonance.core.memory import MemoryGuard


class BoxingGymSystem:
    """Simulated physical systems for BoxingGym active experimentation."""

    def __init__(self, name: str, equation_desc: str, param_ranges: Dict[str, Tuple[float, float]], input_range: Tuple[float, float]):
        self.name = name
        self.equation_desc = equation_desc
        self.param_ranges = param_ranges
        self.input_range = input_range

    def sample_world(self, seed: int) -> Tuple[Dict[str, float], Any]:
        rng = np.random.RandomState(seed)
        params = {}
        for k, (low, high) in self.param_ranges.items():
            params[k] = float(rng.uniform(low, high))

        if self.name == "DampedOscillator":
            # m * \ddot{x} + gamma * \dot{x} + k * x = F_0 cos(omega t)
            # Steady-state amplitude vs drive frequency omega
            m = params["m"]
            k_val = params["k"]
            gamma = params["gamma"]
            omega_0 = np.sqrt(k_val / m)

            def response_fn(omega_in: float) -> float:
                denom = np.sqrt((omega_0**2 - omega_in**2)**2 + (gamma * omega_in / m)**2)
                return float(1.0 / max(denom, 1e-4))

        elif self.name == "MichaelisMenten":
            # Enzyme kinetics: v = (V_max * [S]) / (K_m + [S])
            v_max = params["V_max"]
            k_m = params["K_m"]

            def response_fn(substrate: float) -> float:
                return float((v_max * substrate) / (k_m + substrate))

        elif self.name == "NonlinearSpring":
            # Duffing spring force: F(x) = k1 * x + k3 * x^3
            k1 = params["k1"]
            k3 = params["k3"]

            def response_fn(disp: float) -> float:
                return float(k1 * disp + k3 * (disp**3))
        else:
            raise ValueError(f"Unknown system {self.name}")

        return params, response_fn


BOXING_GYM_SYSTEMS = [
    BoxingGymSystem(
        name="DampedOscillator",
        equation_desc="Driven damped mechanical oscillator: A(omega) = F_0 / sqrt((omega_0^2 - omega^2)^2 + (gamma*omega/m)^2)",
        param_ranges={"m": (0.5, 2.0), "k": (20.0, 100.0), "gamma": (0.2, 1.5)},
        input_range=(1.0, 20.0),
    ),
    BoxingGymSystem(
        name="MichaelisMenten",
        equation_desc="Biochemical catalytic rate: v(S) = (V_max * S) / (K_m + S)",
        param_ranges={"V_max": (10.0, 50.0), "K_m": (1.0, 8.0)},
        input_range=(0.5, 25.0),
    ),
    BoxingGymSystem(
        name="NonlinearSpring",
        equation_desc="Duffing restoring force: F(x) = k1 * x + k3 * x^3",
        param_ranges={"k1": (2.0, 10.0), "k3": (0.5, 3.0)},
        input_range=(0.2, 3.0),
    ),
]


def evaluate_boxing_gym(agent: Any, num_episodes: int = 6, budget_queries: int = 4) -> Dict[str, Any]:
    """Evaluate an agent on BoxingGym active experimental perturbation & system identification."""
    episode_results = []
    total_nrmse = 0.0

    for ep_idx in range(num_episodes):
        system = BOXING_GYM_SYSTEMS[ep_idx % len(BOXING_GYM_SYSTEMS)]
        seed = 42000 + ep_idx
        params, true_fn = system.sample_world(seed)
        sensor_rng = np.random.RandomState(seed + 999)

        # Held-out evaluation grid (5 points)
        eval_x = np.linspace(system.input_range[0] * 1.1, system.input_range[1] * 0.9, 5)
        true_eval_y = np.array([true_fn(x) for x in eval_x])
        y_scale = np.max(np.abs(true_eval_y)) if np.max(np.abs(true_eval_y)) > 1e-6 else 1.0

        # Multi-turn interaction loop
        history: List[Tuple[float, float]] = []
        for step in range(budget_queries):
            # Prompt the agent to choose next perturbation x
            hist_str = "\n".join([f"  Perturbation x={h[0]:.3f} -> Response y={h[1]:.3f}" for h in history]) or "  (No perturbations yet)"
            prompt = (
                f"You are actively identifying an unknown physical system: '{system.name}'.\n"
                f"Governing form: {system.equation_desc}\n"
                f"Valid input range: [{system.input_range[0]}, {system.input_range[1]}].\n"
                f"Current history of observations:\n{hist_str}\n"
                f"Step {step+1}/{budget_queries}. What single numerical perturbation x should you test next?\n"
                f"Output strictly: 'Perturbation: <NUMBER>'."
            )
            resp = agent.generate_response(prompt, max_new_tokens=100)
            MemoryGuard.cleanup_memory()

            # Parse query x
            m = re.findall(r"(?:perturbation|test|input|x)(?:\s*(?:is|:|=)?\s*)([-+]?\d*\.?\d+)", resp, re.IGNORECASE)
            if m:
                try:
                    q_x = float(m[-1])
                except ValueError:
                    q_x = float(np.mean(system.input_range))
            else:
                q_x = float(np.mean(system.input_range))

            # Clamp to range
            q_x = float(np.clip(q_x, system.input_range[0], system.input_range[1]))
            # Observe noisy response
            noise = float(sensor_rng.normal(0, 0.05 * y_scale))
            obs_y = true_fn(q_x) + noise
            history.append((q_x, obs_y))

        # Final prediction on held-out points
        final_prompt = (
            f"Based on your {budget_queries} experimental observations for system '{system.name}':\n"
            + "\n".join([f"  x={h[0]:.3f} -> y={h[1]:.3f}" for h in history]) + "\n"
            f"Predict the system response y at these test points: {list(np.round(eval_x, 2))}.\n"
            f"Format your predictions as a JSON list of numbers: ```json [y1, y2, y3, y4, y5] ```."
        )
        final_resp = agent.generate_response(final_prompt, max_new_tokens=150)
        MemoryGuard.cleanup_memory()

        # Parse predictions
        pred_y = []
        json_m = re.search(r"\[([\d\s\.,eE\-\+]+)\]", final_resp)
        if json_m:
            try:
                pred_y = [float(val.strip()) for val in json_m.group(1).split(",") if val.strip()]
            except Exception:
                pred_y = []

        if len(pred_y) != 5:
            # Fallback to mean observation
            mean_obs = np.mean([h[1] for h in history])
            pred_y = [float(mean_obs)] * 5

        pred_y_arr = np.array(pred_y)
        rmse = float(np.sqrt(np.mean((pred_y_arr - true_eval_y)**2)))
        nrmse = float(rmse / y_scale)
        total_nrmse += nrmse

        # Calculate probe diversity (standard deviation of selected x values)
        xs = [h[0] for h in history]
        x_span_ratio = float((max(xs) - min(xs)) / (system.input_range[1] - system.input_range[0]))

        episode_results.append({
            "episode": ep_idx + 1,
            "system": system.name,
            "queries": [round(x, 2) for x in xs],
            "x_span_ratio": round(x_span_ratio, 3),
            "nrmse": round(nrmse, 4),
            "true_y": [round(y, 3) for y in true_eval_y],
            "pred_y": [round(y, 3) for y in pred_y_arr],
        })

    avg_nrmse = total_nrmse / max(1, num_episodes)
    return {
        "benchmark_name": "BoxingGym-Stanford",
        "episodes_evaluated": num_episodes,
        "mean_nrmse": round(avg_nrmse, 4),
        "mean_reward": round(max(0.0, 1.0 - avg_nrmse), 4),
        "episode_results": episode_results,
    }
