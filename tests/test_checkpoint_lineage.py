import hashlib
import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

class TestCheckpointLineage(unittest.TestCase):
    def test_checkpoint_hashes_match_readme_and_disk(self):
        sft_path = REPO_ROOT / "checkpoints" / "resonance_qwen_sft" / "adapter_model.safetensors"
        rl_path = REPO_ROOT / "checkpoints" / "resonance_qwen_rl" / "adapter_model.safetensors"
        readme_path = REPO_ROOT / "README.md"
        manifest_path = REPO_ROOT / "artifacts" / "checkpoint_binary_hashes.json"

        # Check recorded manifest artifact
        self.assertTrue(manifest_path.is_file(), f"Missing manifest at {manifest_path}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = data.get("expected_sha256")
        self.assertTrue(data.get("comparison", {}).get("bytes_identical"))

        # Verify README mentions the expected hash
        readme_text = readme_path.read_text(encoding="utf-8")
        self.assertIn(
            expected_hash,
            readme_text,
            f"Expected hash {expected_hash} not found in README.md"
        )

        # If disk weights are present (local research workspace), verify live byte hash
        if sft_path.is_file() and rl_path.is_file():
            sft_hash = hashlib.sha256(sft_path.read_bytes()).hexdigest()
            rl_hash = hashlib.sha256(rl_path.read_bytes()).hexdigest()
            self.assertEqual(sft_hash, rl_hash, "SFT and RL adapter weights must have identical SHA-256")
            self.assertEqual(sft_hash, expected_hash, "Disk SHA-256 must match expected hash")
        else:
            # Checkpoints externalized to Hugging Face Hub in public distribution
            self.assertTrue(manifest_path.is_file())

if __name__ == "__main__":
    unittest.main()
