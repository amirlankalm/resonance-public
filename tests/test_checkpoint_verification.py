"""Tests for checkpoint verification, recorded-metadata inspection, and integrity checks."""

from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.verify_checkpoint_identity import (
    verify_checkpoints,
    validate_manifest_metadata,
    PINNED_EXPECTED_SHA256,
    PINNED_EXPECTED_SIZE_BYTES,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCheckpointVerification(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.canonical_manifest = REPO_ROOT / "artifacts" / "checkpoint_binary_hashes.json"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_canonical_manifest_structure_and_validity(self):
        """Canonical committed manifest must pass metadata validation."""
        self.assertTrue(self.canonical_manifest.is_file(), "Canonical manifest must exist")
        with open(self.canonical_manifest, "r") as f:
            data = json.load(f)
        valid, errors = validate_manifest_metadata(data)
        self.assertTrue(valid, f"Canonical manifest metadata validation failed: {errors}")
        self.assertEqual(len(errors), 0)

    def test_recorded_metadata_inspection_fallback(self):
        """When binaries are absent, verify_checkpoints performs metadata inspection."""
        report = verify_checkpoints(
            sft_path="/nonexistent/sft.safetensors",
            rl_path="/nonexistent/rl.safetensors",
            output_artifact=str(self.canonical_manifest),
        )
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["verification_type"], "recorded-metadata inspection")
        self.assertFalse(report["binaries_present_on_disk"])
        self.assertFalse(report["byte_identity_checked"])
        self.assertTrue(report["manifest_metadata_verified"])
        self.assertEqual(report["expected_sha256"], PINNED_EXPECTED_SHA256)
        self.assertEqual(report["expected_size_bytes"], PINNED_EXPECTED_SIZE_BYTES)
        self.assertIn("byte identity was not checked", report["validation_notes"])

    def test_tampered_manifest_expected_sha256(self):
        """Tampered expected_sha256 must be detected and rejected."""
        with open(self.canonical_manifest, "r") as f:
            data = json.load(f)
        data["expected_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"

        tampered_path = Path(self.tmp_dir.name) / "tampered.json"
        with open(tampered_path, "w") as f:
            json.dump(data, f)

        report = verify_checkpoints(
            sft_path="/nonexistent/sft.safetensors",
            rl_path="/nonexistent/rl.safetensors",
            output_artifact=str(tampered_path),
        )
        self.assertEqual(report["status"], "tampered")
        self.assertFalse(report["manifest_metadata_verified"])
        self.assertTrue(any("Expected SHA-256 mismatch" in err for err in report["errors"]))

    def test_tampered_manifest_adapter_hash(self):
        """Tampered adapter sha256 must be detected and rejected."""
        with open(self.canonical_manifest, "r") as f:
            data = json.load(f)
        data["sft_adapter"]["sha256"] = "1111111111111111111111111111111111111111111111111111111111111111"

        tampered_path = Path(self.tmp_dir.name) / "tampered.json"
        with open(tampered_path, "w") as f:
            json.dump(data, f)

        report = verify_checkpoints(
            sft_path="/nonexistent/sft.safetensors",
            rl_path="/nonexistent/rl.safetensors",
            output_artifact=str(tampered_path),
        )
        self.assertEqual(report["status"], "tampered")
        self.assertFalse(report["manifest_metadata_verified"])
        self.assertTrue(any("sft_adapter SHA-256 mismatch" in err for err in report["errors"]))

    def test_tampered_manifest_size(self):
        """Tampered adapter size_bytes must be detected and rejected."""
        with open(self.canonical_manifest, "r") as f:
            data = json.load(f)
        data["rl_adapter"]["size_bytes"] = 123456

        tampered_path = Path(self.tmp_dir.name) / "tampered.json"
        with open(tampered_path, "w") as f:
            json.dump(data, f)

        report = verify_checkpoints(
            sft_path="/nonexistent/sft.safetensors",
            rl_path="/nonexistent/rl.safetensors",
            output_artifact=str(tampered_path),
        )
        self.assertEqual(report["status"], "tampered")
        self.assertFalse(report["manifest_metadata_verified"])
        self.assertTrue(any("rl_adapter size_bytes mismatch" in err for err in report["errors"]))

    def test_tampered_manifest_bytes_identical_false(self):
        """Manifest asserting bytes_identical=False must be rejected."""
        with open(self.canonical_manifest, "r") as f:
            data = json.load(f)
        data["comparison"]["bytes_identical"] = False

        tampered_path = Path(self.tmp_dir.name) / "tampered.json"
        with open(tampered_path, "w") as f:
            json.dump(data, f)

        report = verify_checkpoints(
            sft_path="/nonexistent/sft.safetensors",
            rl_path="/nonexistent/rl.safetensors",
            output_artifact=str(tampered_path),
        )
        self.assertEqual(report["status"], "tampered")
        self.assertFalse(report["manifest_metadata_verified"])
        self.assertTrue(any("bytes_identical" in err for err in report["errors"]))

    def test_malformed_json_manifest(self):
        """Non-JSON file as manifest must return failed report."""
        bad_path = Path(self.tmp_dir.name) / "bad.json"
        bad_path.write_text("NOT_JSON{{{")

        report = verify_checkpoints(
            sft_path="/nonexistent/sft.safetensors",
            rl_path="/nonexistent/rl.safetensors",
            output_artifact=str(bad_path),
        )
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["manifest_metadata_verified"])
        self.assertTrue(any("Malformed JSON" in err for err in report["errors"]))

    def test_missing_manifest_keys(self):
        """Manifest missing required keys must fail schema validation."""
        incomplete_path = Path(self.tmp_dir.name) / "incomplete.json"
        incomplete_path.write_text(json.dumps({"status": "verified"}))

        report = verify_checkpoints(
            sft_path="/nonexistent/sft.safetensors",
            rl_path="/nonexistent/rl.safetensors",
            output_artifact=str(incomplete_path),
        )
        self.assertEqual(report["status"], "tampered")
        self.assertFalse(report["manifest_metadata_verified"])
        self.assertTrue(any("Missing required manifest key" in err for err in report["errors"]))

    def test_downloader_check_only_reports_status(self):
        """Downloader --check-only must truthfully report local binary presence or absence."""
        script_path = REPO_ROOT / "scripts" / "download_checkpoints.py"
        res = subprocess.run(
            [sys.executable, str(script_path), "--check-only"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        sft_path = REPO_ROOT / "checkpoints" / "resonance_qwen_sft" / "adapter_model.safetensors"
        if sft_path.is_file():
            self.assertEqual(res.returncode, 0)
            self.assertIn("local binary verified", res.stdout)
            self.assertIn("8676008 bytes", res.stdout)
        else:
            self.assertEqual(res.returncode, 1)
            self.assertIn("NOT PRESENT ON DISK", res.stdout)
            self.assertIn("byte identity not checked", res.stdout)

    def test_downloader_check_only_when_binaries_absent(self):
        """When binaries are absent, downloader logic reports NOT PRESENT ON DISK and fails (exit code 1)."""
        import shutil
        from scripts.download_checkpoints import verify_checkpoint
        empty_dir = Path(self.tmp_dir.name) / "empty_ckpt"
        empty_dir.mkdir(parents=True, exist_ok=True)
        self.assertFalse(verify_checkpoint("resonance_qwen_rl", empty_dir))

        # Copy canonical manifest to tmp_dir/artifacts
        artifacts_dir = Path(self.tmp_dir.name) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.canonical_manifest, artifacts_dir / "checkpoint_binary_hashes.json")

        # Run python snippet with mocked REPO_ROOT pointing to tmp_dir
        py_code = (
            "import sys; from pathlib import Path; "
            "import scripts.download_checkpoints as dc; "
            f"dc.REPO_ROOT = Path({json.dumps(str(self.tmp_dir.name))}); "
            "ret = dc.main(); "
            "sys.exit(ret)"
        )
        res = subprocess.run(
            [sys.executable, "-c", py_code, "--check-only"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 1, "Must exit 1 when binaries are absent in check-only mode")
        self.assertIn("NOT PRESENT ON DISK", res.stdout)
        self.assertIn("byte identity not checked", res.stdout)

    def test_zero_dependency_wording_in_readme(self):
        """README must contain exact approved dependency scope wording and no zero-dependency claims."""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        expected_phrase = (
            "The environment and numerical baselines run locally without hosted APIs "
            "or downloaded checkpoints, after installing runtime dependencies. Training requires the ML extras."
        )
        self.assertIn(expected_phrase, readme)
        self.assertNotIn("zero-dependency", readme)

    def test_no_signed_manifest_claims(self):
        """README and scripts must not claim a signed release manifest without public key verification."""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("signed release manifest", readme)
        self.assertNotIn("signed manifest", readme)

        verify_script = (REPO_ROOT / "scripts" / "verify_checkpoint_identity.py").read_text(encoding="utf-8")
        self.assertNotIn("signed release manifest", verify_script)
        self.assertNotIn("signed manifest", verify_script)


if __name__ == "__main__":
    unittest.main()
