"""Unit tests for Resonance MemoryGuard."""

import unittest
from resonance.core.memory import MemoryGuard


class TestMemoryGuard(unittest.TestCase):
    def test_get_system_memory_info(self):
        info = MemoryGuard.get_system_memory_info()
        self.assertIn("total_ram_mb", info)
        self.assertIn("available_ram_mb", info)
        self.assertIn("used_ram_mb", info)
        self.assertIn("percent_used", info)
        self.assertIn("process_rss_mb", info)
        self.assertGreater(info["total_ram_mb"], 1000)
        self.assertGreater(info["process_rss_mb"], 0)

    def test_safe_training_config(self):
        guard = MemoryGuard()
        cfg = guard.get_safe_training_config()
        self.assertIn("mode", cfg)
        self.assertIn("batch_size", cfg)
        self.assertIn("grad_accum_steps", cfg)
        self.assertIn("max_tokens", cfg)
        self.assertIn("gc_frequency", cfg)
        self.assertGreaterEqual(cfg["batch_size"], 1)
        self.assertGreaterEqual(cfg["grad_accum_steps"], 1)

    def test_cleanup_memory(self):
        # Should execute cleanly without raising exception
        MemoryGuard.cleanup_memory()


if __name__ == "__main__":
    unittest.main()
