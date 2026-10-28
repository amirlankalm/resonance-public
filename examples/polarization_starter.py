"""Polarization starter script and 100-seed numerical development check."""

import numpy as np
from scipy.optimize import minimize
from resonance.envs.polarization import PolarizationEnv
from resonance.physics.optics import malus_law


def fit_polarization(angles: np.ndarray, intensities: np.ndarray):
    """Fit A, b, phi on measured points."""
    def loss(p):
        A, b, phi = p
        if A <= 0 or b < 0:
            return 1e6
        pred = malus_law(angles, A, b, phi)
        return float(np.sum((pred - intensities) ** 2))

    res = minimize(
        loss,
        x0=[0.7, 0.05, 45.0],
        bounds=[(0.4, 1.0), (0.0, 0.15), (0.0, 180.0)],
        method="L-BFGS-B"
    )
    return res.x[0], res.x[1], res.x[2]


def run_100_seeds():
    num_seeds = 100
    fixed_angles = np.linspace(0.0, 150.0, 6)

    fixed_nrmses = []
    fixed_rewards = []
    random_nrmses = []
    random_rewards = []

    for seed in range(num_seeds):
        # 1. Fixed design
        env = PolarizationEnv()
        env.reset(seed=seed)
        for a in fixed_angles:
            env.step({"type": "measure", "angle_deg": float(a)})
        meas_angles = np.array([h["query_input"]["angle_deg"] for h in env.history])
        meas_y = np.array([h["reading"]["intensity"] for h in env.history])
        A_f, b_f, phi_f = fit_polarization(meas_angles, meas_y)
        _, rew, _, _, info = env.step({"type": "submit", "amplitude": A_f, "offset": b_f, "phi_deg": phi_f})
        fixed_nrmses.append(info["nrmse"])
        fixed_rewards.append(rew)

        # 2. Random design
        env = PolarizationEnv()
        env.reset(seed=seed)
        rng = np.random.default_rng(seed + 999)
        rand_angles = rng.uniform(0.0, 180.0, size=6)
        for a in rand_angles:
            env.step({"type": "measure", "angle_deg": float(a)})
        meas_angles = np.array([h["query_input"]["angle_deg"] for h in env.history])
        meas_y = np.array([h["reading"]["intensity"] for h in env.history])
        A_r, b_r, phi_r = fit_polarization(meas_angles, meas_y)
        _, rew, _, _, info = env.step({"type": "submit", "amplitude": A_r, "offset": b_r, "phi_deg": phi_r})
        random_nrmses.append(info["nrmse"])
        random_rewards.append(rew)

    print("=== Polarization Starter Development Results (100 Seeds) ===")
    print(f"Fixed Design (6 calls):  Mean NRMSE = {np.mean(fixed_nrmses):.7f}, Mean Reward = {np.mean(fixed_rewards):.7f}")
    print(f"Random Design (6 calls): Mean NRMSE = {np.mean(random_nrmses):.7f}, Mean Reward = {np.mean(random_rewards):.7f}")


if __name__ == "__main__":
    run_100_seeds()
