"""Stream-isolated Random Number Generator manager for Resonance.

Ensures that world generation, sensor noise, and evaluation frequencies
use strictly independent, non-interfering RNG streams, preventing seed leakage
and state reconstruction.
"""

from __future__ import annotations
import hashlib
from typing import Tuple
import numpy as np


class IsolatedRNGManager:
    """Manages independent PRNG streams for world generation, sensor noise, and evaluation."""

    def __init__(self, master_seed: int | None = None):
        if master_seed is None:
            ss = np.random.SeedSequence()
            master_seed = int(ss.generate_state(1)[0])
        self.master_seed = int(master_seed)

        # Derive independent stream seeds using cryptographic hashing to prevent correlations
        self.world_seed = self._derive_seed(self.master_seed, "resonance_world_v1")
        self.sensor_seed = self._derive_seed(self.master_seed, "resonance_sensor_v1")
        self.eval_seed = self._derive_seed(self.master_seed, "resonance_eval_v1")

        self.rng_world = np.random.default_rng(self.world_seed)
        self.rng_sensor = np.random.default_rng(self.sensor_seed)
        self.rng_eval = np.random.default_rng(self.eval_seed)

    @staticmethod
    def _derive_seed(seed: int, domain: str) -> int:
        h = hashlib.sha256(f"{seed}:{domain}".encode("utf-8")).digest()
        # Take 32-bit unsigned integer
        return int.from_bytes(h[:4], byteorder="big")

    def reseed_sensor(self, run_seed: int) -> None:
        """Reseed only the sensor noise stream (e.g. for multiple noisy rollouts of same world)."""
        new_seed = self._derive_seed(run_seed, "resonance_sensor_reseed")
        self.rng_sensor = np.random.default_rng(new_seed)
