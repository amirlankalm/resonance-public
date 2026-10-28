"""Numerical baseline methods for Resonance environments.

Implements the complete numerical ladder specified in the project roadmap:
1. Prior-only baseline (0 queries)
2. Random frequency design + Least-Squares / MLE fitting
3. Fixed log-spaced sweep + Least-Squares / MLE fitting
4. Sequential adaptive experimental design (Maximum Disagreement & Information Gain)
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize, curve_fit

from resonance.physics.circuits import (
    CircuitType,
    compute_complex_current,
    compute_resonant_frequency,
)


def fit_rc_circuit(
    freqs: np.ndarray,
    I_meas: np.ndarray,
    V_rms: float = 0.1,
) -> Tuple[float, float, float]:
    """Fit series RC parameters (R, C_uF) using complex current response.

    Returns:
        (best_R, best_C_uF, residual_sum_sq)
    """
    # Scale to mA so gradients are O(1) avoiding premature SciPy termination
    I_meas_mA = I_meas * 1e3

    def loss(p):
        R_val, C_uF_val = p
        if R_val <= 0 or C_uF_val <= 0:
            return 1e9
        I_pred = compute_complex_current(CircuitType.RC, freqs, V_rms, R_val, C_F=C_uF_val * 1e-6) * 1e3
        diff = I_pred - I_meas_mA
        return float(np.sum(np.real(diff * np.conj(diff))))

    best_loss = float("inf")
    best_params = (100.0, 2.0)
    for r_g in [40.0, 100.0, 180.0]:
        for c_g in [0.3, 1.5, 6.0]:
            res = minimize(
                loss,
                x0=[r_g, c_g],
                bounds=[(15.0, 250.0), (0.05, 15.0)],
                method="L-BFGS-B"
            )
            if res.fun < best_loss:
                best_loss = float(res.fun)
                best_params = (float(res.x[0]), float(res.x[1]))

    return best_params[0], best_params[1], best_loss * 1e-6


def fit_rl_circuit(
    freqs: np.ndarray,
    I_meas: np.ndarray,
    V_rms: float = 0.1,
) -> Tuple[float, float, float]:
    """Fit series RL parameters (R, L_mH) using complex current response.

    Returns:
        (best_R, best_L_mH, residual_sum_sq)
    """
    I_meas_mA = I_meas * 1e3

    def loss(p):
        R_val, L_mH_val = p
        if R_val <= 0 or L_mH_val <= 0:
            return 1e9
        I_pred = compute_complex_current(CircuitType.RL, freqs, V_rms, R_val, L_H=L_mH_val * 1e-3) * 1e3
        diff = I_pred - I_meas_mA
        return float(np.sum(np.real(diff * np.conj(diff))))

    best_loss = float("inf")
    best_params = (100.0, 20.0)
    for r_g in [40.0, 100.0, 180.0]:
        for l_g in [3.0, 25.0, 80.0]:
            res = minimize(
                loss,
                x0=[r_g, l_g],
                bounds=[(15.0, 250.0), (0.5, 150.0)],
                method="L-BFGS-B"
            )
            if res.fun < best_loss:
                best_loss = float(res.fun)
                best_params = (float(res.x[0]), float(res.x[1]))

    return best_params[0], best_params[1], best_loss * 1e-6


def fit_rlc_circuit(
    freqs: np.ndarray,
    I_meas: np.ndarray,
    V_rms: float = 0.1,
) -> Tuple[float, float, float, float]:
    """Fit series RLC parameters (R, L_mH, C_uF) using complex current response.

    Returns:
        (best_R, best_L_mH, best_C_uF, residual_sum_sq)
    """
    I_meas_mA = I_meas * 1e3

    def loss(p):
        R_val, L_mH_val, C_uF_val = p
        if R_val <= 0 or L_mH_val <= 0 or C_uF_val <= 0:
            return 1e9
        I_pred = compute_complex_current(
            CircuitType.RLC,
            freqs,
            V_rms,
            R_val,
            L_H=L_mH_val * 1e-3,
            C_F=C_uF_val * 1e-6,
        ) * 1e3
        diff = I_pred - I_meas_mA
        return float(np.sum(np.real(diff * np.conj(diff))))

    best_loss = float("inf")
    best_params = (100.0, 20.0, 2.0)

    for r_g in [30.0, 90.0, 160.0]:
        for l_g in [5.0, 30.0, 80.0]:
            for c_g in [0.4, 2.0, 6.0]:
                res = minimize(
                    loss,
                    x0=[r_g, l_g, c_g],
                    bounds=[(15.0, 250.0), (0.5, 150.0), (0.05, 15.0)],
                    method="L-BFGS-B"
                )
                if res.fun < best_loss:
                    best_loss = float(res.fun)
                    best_params = (float(res.x[0]), float(res.x[1]), float(res.x[2]))

    return best_params[0], best_params[1], best_params[2], best_loss * 1e-6


def reconstruct_measured_phasors(history: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct measured complex phasors from environment history."""
    freq_list = []
    phasor_list = []

    for item in history:
        freq = float(item["query_input"]["frequency_hz"])
        amp = item["reading"]["current_amplitude_A"]
        phase = item["reading"]["current_phase_rad"]

        if amp is None:
            continue

        if phase is None or np.isnan(phase):
            phase = 0.0

        # Phasor: amp * exp(j * phase)
        phasor = amp * (np.cos(phase) + 1j * np.sin(phase))
        freq_list.append(freq)
        phasor_list.append(phasor)

    return np.array(freq_list, dtype=np.float64), np.array(phasor_list, dtype=np.complex128)


def evaluate_candidate_models(
    history: List[Dict[str, Any]],
    V_rms: float = 0.1,
) -> Dict[str, Any]:
    """Fit all 3 candidate models on history and compute model selection metrics."""
    freqs, I_meas = reconstruct_measured_phasors(history)
    K = len(freqs)

    if K == 0:
        # No data: return prior defaults
        return {
            "selected_type": "RLC",
            "model_probs": {"RC": 0.333, "RL": 0.333, "RLC": 0.334},
            "parameters": {
                "circuit_type": "RLC",
                "R": 110.0,
                "L_mH": 50.0,
                "C_uF": 5.0,
            }
        }

    # Fit RC
    r_rc, c_rc, rss_rc = fit_rc_circuit(freqs, I_meas, V_rms)
    # Fit RL
    r_rl, l_rl, rss_rl = fit_rl_circuit(freqs, I_meas, V_rms)
    # Fit RLC
    r_rlc, l_rlc, c_rlc, rss_rlc = fit_rlc_circuit(freqs, I_meas, V_rms)

    # Bayesian Information Criterion: BIC = K * ln(RSS / K + eps) + p * ln(K)
    eps = 1e-14
    bic_rc = K * np.log(max(rss_rc / K, eps)) + 2.0 * np.log(max(K, 2))
    bic_rl = K * np.log(max(rss_rl / K, eps)) + 2.0 * np.log(max(K, 2))
    bic_rlc = K * np.log(max(rss_rlc / K, eps)) + 3.0 * np.log(max(K, 2))

    bics = np.array([bic_rc, bic_rl, bic_rlc])
    # Softmax over negative half-BIC: p ~ exp(-0.5 * (BIC - min_BIC))
    shifted = -0.5 * (bics - np.min(bics))
    probs = np.exp(np.clip(shifted, -20.0, 20.0))
    probs = probs / np.sum(probs)

    best_idx = int(np.argmin(bics))
    types = ["RC", "RL", "RLC"]
    sel_type = types[best_idx]

    params = {"circuit_type": sel_type}
    if sel_type == "RC":
        params["R"] = r_rc
        params["C_uF"] = c_rc
    elif sel_type == "RL":
        params["R"] = r_rl
        params["L_mH"] = l_rl
    else:
        params["R"] = r_rlc
        params["L_mH"] = l_rlc
        params["C_uF"] = c_rlc

    return {
        "selected_type": sel_type,
        "model_probs": {
            "RC": float(probs[0]),
            "RL": float(probs[1]),
            "RLC": float(probs[2]),
        },
        "parameters": params,
        "fits": {
            "RC": {"R": r_rc, "C_uF": c_rc, "rss": rss_rc, "bic": float(bic_rc)},
            "RL": {"R": r_rl, "L_mH": l_rl, "rss": rss_rl, "bic": float(bic_rl)},
            "RLC": {"R": r_rlc, "L_mH": l_rlc, "C_uF": c_rlc, "rss": rss_rlc, "bic": float(bic_rlc)},
        }
    }


# ============================================================================
# BASELINE RUNNERS
# ============================================================================

def run_prior_only_baseline(env, seed: Optional[int] = None) -> Dict[str, Any]:
    """Prior-only baseline: 0 queries, immediate submit with prior midpoint."""
    obs, _ = env.reset(seed=seed)
    action = {
        "type": "submit",
        "model_probs": {"RC": 0.333, "RL": 0.333, "RLC": 0.334},
        "parameters": {
            "circuit_type": "RLC",
            "R": 110.0,
            "L_mH": 50.0,
            "C_uF": 5.0,
        }
    }
    _, reward, term, trunc, info = env.step(action)
    return info


def run_random_design_baseline(env, num_measurements: int = 6, seed: Optional[int] = None) -> Dict[str, Any]:
    """Random frequency design: chooses N frequencies randomly log-spaced in instrument band."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    log_min = np.log10(env.BAND_MIN_HZ)
    log_max = np.log10(env.BAND_MAX_HZ)
    random_freqs = 10 ** rng.uniform(log_min, log_max, size=num_measurements)

    for freq in random_freqs:
        act = {"type": "measure", "frequency_hz": float(freq)}
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break

    # Fit models on accumulated history
    fit_result = evaluate_candidate_models(obs["history"], V_rms=env.V_DRIVE_RMS)
    submit_act = {
        "type": "submit",
        "model_probs": fit_result["model_probs"],
        "parameters": fit_result["parameters"],
    }
    _, reward, term, trunc, info = env.step(submit_act)
    return info


def run_fixed_sweep_baseline(env, num_measurements: int = 6, seed: Optional[int] = None) -> Dict[str, Any]:
    """Fixed log-spaced frequency sweep."""
    obs, _ = env.reset(seed=seed)
    fixed_freqs = np.geomspace(env.BAND_MIN_HZ, env.BAND_MAX_HZ, num_measurements)

    for freq in fixed_freqs:
        act = {"type": "measure", "frequency_hz": float(freq)}
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break

    fit_result = evaluate_candidate_models(obs["history"], V_rms=env.V_DRIVE_RMS)
    submit_act = {
        "type": "submit",
        "model_probs": fit_result["model_probs"],
        "parameters": fit_result["parameters"],
    }
    _, reward, term, trunc, info = env.step(submit_act)
    return info


def run_adaptive_design_baseline(
    env,
    max_measurements: int = 6,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Active sequential experiment design baseline:

    Selects probe frequencies by maximizing expected information gain (predictive variance across
    candidate models, resonance zero-crossing tracking, and space-filling exploration).
    """
    obs, _ = env.reset(seed=seed)

    # Initial 2 anchor points: low and high extremes of band
    initial_probes = [env.BAND_MIN_HZ, env.BAND_MAX_HZ]
    for freq in initial_probes:
        if len(obs["history"]) >= max_measurements:
            break
        act = {"type": "measure", "frequency_hz": float(freq)}
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break

    # Candidate evaluation grid for active selection
    grid_freqs = np.geomspace(env.BAND_MIN_HZ, env.BAND_MAX_HZ, 100)

    # Sequential active loop
    while len(obs["history"]) < max_measurements and not (term or trunc):
        # Fit current candidate models
        fit_res = evaluate_candidate_models(obs["history"], V_rms=env.V_DRIVE_RMS)
        probs = fit_res.get("model_probs", {"RC": 0.33, "RL": 0.33, "RLC": 0.34})
        fits = fit_res.get("fits", {})
        p_rc, p_rl, p_rlc = probs["RC"], probs["RL"], probs["RLC"]

        # Early stopping for high budgets when posterior is converged (saves query cost penalty)
        if len(obs["history"]) >= 5 and max(p_rc, p_rl, p_rlc) > 0.98:
            sel = fit_res["selected_type"]
            if fits.get(sel, {}).get("rss", float("inf")) < 5e-7:
                break

        # 3rd query: target resonance peak if RLC suspected, else geometric center
        if len(obs["history"]) == 2:
            p1 = obs["history"][0]["reading"]["current_phase_rad"]
            p2 = obs["history"][1]["reading"]["current_phase_rad"]
            if p1 is not None and p2 is not None and p1 > 0.1 and p2 < -0.1:
                # Phase zero crossing (RLC resonance peak)
                frac = p1 / (p1 - p2 + 1e-6)
                log_f0 = np.log10(env.BAND_MIN_HZ) + frac * (np.log10(env.BAND_MAX_HZ) - np.log10(env.BAND_MIN_HZ))
                chosen_freq = float(np.clip(10**log_f0, 45.0, 22000.0))
            else:
                chosen_freq = float(np.sqrt(env.BAND_MIN_HZ * env.BAND_MAX_HZ))
        else:
            # Predict across grid for all 3 models
            I_rc = compute_complex_current(
                CircuitType.RC, grid_freqs, env.V_DRIVE_RMS,
                fits["RC"]["R"], C_F=fits["RC"]["C_uF"] * 1e-6
            )
            I_rl = compute_complex_current(
                CircuitType.RL, grid_freqs, env.V_DRIVE_RMS,
                fits["RL"]["R"], L_H=fits["RL"]["L_mH"] * 1e-3
            )
            I_rlc = compute_complex_current(
                CircuitType.RLC, grid_freqs, env.V_DRIVE_RMS,
                fits["RLC"]["R"], L_H=fits["RLC"]["L_mH"] * 1e-3, C_F=fits["RLC"]["C_uF"] * 1e-6
            )

            # Posterior-weighted complex variance
            I_bar = p_rc * I_rc + p_rl * I_rl + p_rlc * I_rlc
            var_pred = p_rc * np.abs(I_rc - I_bar)**2 + p_rl * np.abs(I_rl - I_bar)**2 + p_rlc * np.abs(I_rlc - I_bar)**2

            # Space-filling exploration bonus
            visited = np.array([h["query_input"]["frequency_hz"] for h in obs["history"]])
            log_v = np.log10(visited)
            log_g = np.log10(grid_freqs)
            min_dist = np.min(np.abs(log_g[:, None] - log_v[None, :]), axis=1)

            acq = (var_pred / (np.max(var_pred) + 1e-12)) + 0.6 * min_dist
            for vf in visited:
                dist = np.abs(log_g - np.log10(vf))
                acq *= (1.0 - 0.99 * np.exp(-0.5 * (dist / 0.18)**2))

            best_freq_idx = int(np.argmax(acq))
            chosen_freq = float(grid_freqs[best_freq_idx])

        act = {"type": "measure", "frequency_hz": chosen_freq}
        obs, _, term, trunc, _ = env.step(act)

    # Final fit & submission
    fit_result = evaluate_candidate_models(obs["history"], V_rms=env.V_DRIVE_RMS)
    submit_act = {
        "type": "submit",
        "model_probs": fit_result["model_probs"],
        "parameters": fit_result["parameters"],
    }
    _, reward, term, trunc, info = env.step(submit_act)
    return info
