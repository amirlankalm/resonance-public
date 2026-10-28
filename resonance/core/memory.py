"""Dynamic memory guard and system RAM adaptation for Resonance.

Monitors physical RAM, unified Apple Silicon memory pressure, and process RSS
to dynamically scale batch sizes, context lengths, and garbage collection,
preventing Out-Of-Memory (OOM) crashes and system freezes.
"""

from __future__ import annotations
import os
import gc
import platform
import subprocess
from typing import Dict, Any, Tuple
import psutil


class MemoryGuard:
    """Monitors system memory and enforces adaptive memory safety caps."""

    def __init__(self, target_memory_headroom_mb: int = 1500):
        self.target_memory_headroom_mb = target_memory_headroom_mb
        self.is_macos = platform.system() == "Darwin"

    @staticmethod
    def get_system_memory_info() -> Dict[str, float]:
        """Query physical RAM, available RAM, and process RSS in Megabytes (MB)."""
        vm = psutil.virtual_memory()
        proc = psutil.Process(os.getpid())
        proc_rss_mb = proc.memory_info().rss / (1024 * 1024)

        return {
            "total_ram_mb": round(vm.total / (1024 * 1024), 1),
            "available_ram_mb": round(vm.available / (1024 * 1024), 1),
            "used_ram_mb": round(vm.used / (1024 * 1024), 1),
            "percent_used": vm.percent,
            "process_rss_mb": round(proc_rss_mb, 1),
        }

    def get_safe_training_config(self) -> Dict[str, Any]:
        """Compute memory-adapted hyperparameters based on available RAM headroom.

        Ensures that low available RAM automatically scales down batch size
        while increasing gradient accumulation steps to maintain the same effective batch size in expectation!
        """
        mem = self.get_system_memory_info()
        avail_mb = mem["available_ram_mb"]

        # Tight memory conditions (< 2500 MB available)
        if avail_mb < 2500:
            batch_size = 1
            grad_accum_steps = 4
            max_tokens = 512
            gc_frequency = 1  # gc every step
            level = "CRITICAL_CONSERVATIVE"
        # Moderate memory conditions (2500 - 5000 MB available)
        elif avail_mb < 5000:
            batch_size = 2
            grad_accum_steps = 2
            max_tokens = 768
            gc_frequency = 2
            level = "MODERATE"
        # Plentiful memory (> 5000 MB available)
        else:
            batch_size = 4
            grad_accum_steps = 1
            max_tokens = 1024
            gc_frequency = 5
            level = "STANDARD"

        return {
            "mode": level,
            "available_ram_mb": avail_mb,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "max_tokens": max_tokens,
            "gc_frequency": gc_frequency,
        }

    def check_memory_or_fail(self, min_headroom_mb: float = 500.0) -> None:
        """Fail closed if available memory drops below safety threshold."""
        info = self.get_system_memory_info()
        avail = info["available_ram_mb"]
        if avail < min_headroom_mb:
            raise MemoryError(
                f"Available system RAM ({avail:.1f} MB) is below minimum safe threshold "
                f"({min_headroom_mb:.1f} MB). Failing closed to prevent OOM crash."
            )

    @staticmethod
    def cleanup_memory() -> None:
        """Force garbage collection and release PyTorch/MPS cache if torch is installed."""
        gc.collect()
        try:
            import torch
            if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except ImportError:
            pass
