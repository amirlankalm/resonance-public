"""Unit test for R2: Assistance flags, execution_mode retention, and solver provenance."""

import json
import os
import unittest
from pathlib import Path


class TestAssistanceFlags(unittest.TestCase):
    """Verify that assistance flags do not hide failure episodes and mode filtering retains 50/50 records."""

    def test_assisted_mode_filtering_retains_all_records(self):
        repo_root = Path(__file__).resolve().parent.parent
        assisted_files = [
            repo_root / "artifacts/model_eval_qwen_sft/episodes.jsonl",
            repo_root / "artifacts/model_eval_qwen_rl/episodes.jsonl",
            repo_root / "artifacts/model_eval_qwen_base/episodes.jsonl",
        ]

        for path in assisted_files:
            self.assertTrue(path.exists(), f"Artifact missing: {path}")
            with open(path, "r", encoding="utf-8") as f:
                episodes = [json.loads(line) for line in f]

            self.assertEqual(len(episodes), 50, f"{path} should contain exactly 50 evaluation worlds")
            
            # Key R2 check: Filtering on execution_mode == 'assisted' retains 50/50 records
            assisted_mode_eps = [e for e in episodes if e.get("execution_mode") == "assisted"]
            self.assertEqual(
                len(assisted_mode_eps), 50,
                f"Filtering by execution_mode == 'assisted' in {path} must retain all 50 records without dropping failures"
            )

            # Check that failed episodes carry solver_used == False
            failed_eps = [e for e in episodes if e.get("submitted") is not True]
            for fail_ep in failed_eps:
                self.assertFalse(
                    fail_ep.get("solver_used"),
                    f"Episode {fail_ep.get('world_id')} failed submission, solver_used must be False"
                )

    def test_strict_mode_filtering_retains_all_records(self):
        repo_root = Path(__file__).resolve().parent.parent
        strict_files = [
            repo_root / "artifacts/model_eval_qwen_sft_strict/episodes.jsonl",
            repo_root / "artifacts/model_eval_qwen_rl_strict/episodes.jsonl",
            repo_root / "artifacts/model_eval_qwen_base_strict/episodes.jsonl",
        ]

        for path in strict_files:
            self.assertTrue(path.exists(), f"Artifact missing: {path}")
            with open(path, "r", encoding="utf-8") as f:
                episodes = [json.loads(line) for line in f]

            self.assertEqual(len(episodes), 50, f"{path} should contain exactly 50 evaluation worlds")
            
            # Key R2 check: Filtering on execution_mode == 'strict' retains 50/50 records
            strict_mode_eps = [e for e in episodes if e.get("execution_mode") == "strict"]
            self.assertEqual(
                len(strict_mode_eps), 50,
                f"Filtering by execution_mode == 'strict' in {path} must retain all 50 records"
            )

            # Strict mode never uses numerical solver
            for ep in episodes:
                self.assertFalse(ep.get("solver_used", False))
                self.assertIsNone(ep.get("fit_source"))

    def test_unified_headline_eval_assistance_provenance(self):
        repo_root = Path(__file__).resolve().parent.parent
        unified_path = repo_root / "artifacts/unified_headline_eval.json"
        self.assertTrue(unified_path.exists())

        with open(unified_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sec_name in ["sft_model_assisted", "rl_model_assisted", "base_model_assisted"]:
            eps = data[sec_name]["episodes"]
            self.assertEqual(len(eps), 50)
            retained = [e for e in eps if e.get("execution_mode") == "assisted"]
            self.assertEqual(len(retained), 50, f"{sec_name} must retain 50 records under execution_mode == 'assisted'")


if __name__ == "__main__":
    unittest.main()
