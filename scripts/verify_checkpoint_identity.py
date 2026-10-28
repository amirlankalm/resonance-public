"""Verifies binary identity of checkpoint adapter weights and writes an auditable verification artifact."""

from __future__ import annotations
import os
import sys
import json
import hashlib
import filecmp
import subprocess
from typing import Dict, Any, List, Tuple

PINNED_EXPECTED_SHA256 = "22582b459d3523fc60ba7f7ec3afaf9792ec7b74e336fc75c26c5dcd74d459cd"
PINNED_EXPECTED_SIZE_BYTES = 8676008


def hash_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_blob_id(path: str) -> str:
    try:
        out = subprocess.check_output(["git", "hash-object", path], text=True).strip()
        return out
    except Exception:
        return "N/A"


def validate_manifest_metadata(manifest_data: Any) -> Tuple[bool, List[str]]:
    """Strictly validate manifest dictionary schema, pinned hashes, expected sizes, and identity flags."""
    errors: List[str] = []
    if not isinstance(manifest_data, dict):
        return False, ["Manifest root must be a JSON object / dict"]

    # Schema checks
    required_keys = ["expected_sha256", "sft_adapter", "rl_adapter", "comparison"]
    for key in required_keys:
        if key not in manifest_data:
            errors.append(f"Missing required manifest key: '{key}'")

    if errors:
        return False, errors

    # Pinned expected SHA-256 check
    if manifest_data.get("expected_sha256") != PINNED_EXPECTED_SHA256:
        errors.append(
            f"Expected SHA-256 mismatch in manifest: got '{manifest_data.get('expected_sha256')}', "
            f"pinned trusted is '{PINNED_EXPECTED_SHA256}'"
        )

    # sft_adapter checks
    sft = manifest_data.get("sft_adapter", {})
    if not isinstance(sft, dict):
        errors.append("'sft_adapter' must be an object")
    else:
        if sft.get("sha256") != PINNED_EXPECTED_SHA256:
            errors.append(
                f"sft_adapter SHA-256 mismatch: got '{sft.get('sha256')}', "
                f"pinned trusted is '{PINNED_EXPECTED_SHA256}'"
            )
        if sft.get("size_bytes") != PINNED_EXPECTED_SIZE_BYTES:
            errors.append(
                f"sft_adapter size_bytes mismatch: got '{sft.get('size_bytes')}', "
                f"pinned trusted is '{PINNED_EXPECTED_SIZE_BYTES}'"
            )
        if sft.get("matches_expected") is not True:
            errors.append("sft_adapter 'matches_expected' flag must be True")

    # rl_adapter checks
    rl = manifest_data.get("rl_adapter", {})
    if not isinstance(rl, dict):
        errors.append("'rl_adapter' must be an object")
    else:
        if rl.get("sha256") != PINNED_EXPECTED_SHA256:
            errors.append(
                f"rl_adapter SHA-256 mismatch: got '{rl.get('sha256')}', "
                f"pinned trusted is '{PINNED_EXPECTED_SHA256}'"
            )
        if rl.get("size_bytes") != PINNED_EXPECTED_SIZE_BYTES:
            errors.append(
                f"rl_adapter size_bytes mismatch: got '{rl.get('size_bytes')}', "
                f"pinned trusted is '{PINNED_EXPECTED_SIZE_BYTES}'"
            )
        if rl.get("matches_expected") is not True:
            errors.append("rl_adapter 'matches_expected' flag must be True")

    # comparison checks
    comp = manifest_data.get("comparison", {})
    if not isinstance(comp, dict):
        errors.append("'comparison' must be an object")
    else:
        if comp.get("sha256_identical") is not True:
            errors.append("comparison 'sha256_identical' must be True")
        if comp.get("bytes_identical") is not True:
            errors.append("comparison 'bytes_identical' must be True")

    return len(errors) == 0, errors


def verify_checkpoints(
    sft_path: str = "checkpoints/resonance_qwen_sft/adapter_model.safetensors",
    rl_path: str = "checkpoints/resonance_qwen_rl/adapter_model.safetensors",
    output_artifact: str = "artifacts/checkpoint_binary_hashes.json",
) -> Dict[str, Any]:
    """Verify checkpoints either via direct binary rehash or recorded-metadata inspection."""
    binaries_present = os.path.exists(sft_path) and os.path.exists(rl_path)

    # If binary weights are externalized (e.g. in public distribution before download),
    # perform recorded-metadata inspection against pinned trusted values.
    if not binaries_present:
        if not os.path.exists(output_artifact):
            raise FileNotFoundError(
                f"Checkpoint binaries absent ({sft_path}, {rl_path}) and manifest {output_artifact} not found."
            )
        try:
            with open(output_artifact, "r") as fp:
                manifest_data = json.load(fp)
        except Exception as e:
            return {
                "status": "failed",
                "verification_type": "recorded-metadata inspection",
                "binaries_present_on_disk": False,
                "byte_identity_checked": False,
                "manifest_metadata_verified": False,
                "errors": [f"Malformed JSON manifest: {e}"],
            }

        valid, errors = validate_manifest_metadata(manifest_data)
        if not valid:
            return {
                "status": "tampered",
                "verification_type": "recorded-metadata inspection",
                "binaries_present_on_disk": False,
                "byte_identity_checked": False,
                "manifest_metadata_verified": False,
                "expected_sha256": PINNED_EXPECTED_SHA256,
                "expected_size_bytes": PINNED_EXPECTED_SIZE_BYTES,
                "errors": errors,
                "manifest_data": manifest_data,
            }

        return {
            "status": "verified",
            "verification_type": "recorded-metadata inspection",
            "binaries_present_on_disk": False,
            "byte_identity_checked": False,
            "manifest_metadata_verified": True,
            "expected_sha256": PINNED_EXPECTED_SHA256,
            "expected_size_bytes": PINNED_EXPECTED_SIZE_BYTES,
            "sft_adapter": manifest_data["sft_adapter"],
            "rl_adapter": manifest_data["rl_adapter"],
            "comparison": manifest_data["comparison"],
            "validation_notes": (
                "All manifest fields, schema structure, SHA-256 hashes, and file sizes "
                "validated against pinned trusted release constants. Local weights are externalized; "
                "byte identity was not checked."
            ),
        }

    # Binary re-hash on local disk
    sft_sha256 = hash_file_sha256(sft_path)
    rl_sha256 = hash_file_sha256(rl_path)
    sft_size = os.path.getsize(sft_path)
    rl_size = os.path.getsize(rl_path)
    sft_blob = get_git_blob_id(sft_path)
    rl_blob = get_git_blob_id(rl_path)

    bytes_identical = filecmp.cmp(sft_path, rl_path, shallow=False)
    hashes_match = (sft_sha256 == rl_sha256)
    sft_matches_pinned = (sft_sha256 == PINNED_EXPECTED_SHA256 and sft_size == PINNED_EXPECTED_SIZE_BYTES)
    rl_matches_pinned = (rl_sha256 == PINNED_EXPECTED_SHA256 and rl_size == PINNED_EXPECTED_SIZE_BYTES)

    is_verified = (bytes_identical and hashes_match and sft_matches_pinned and rl_matches_pinned)

    report = {
        "status": "verified" if is_verified else "mismatch",
        "verification_type": "binary-byte-rehash",
        "binaries_present_on_disk": True,
        "byte_identity_checked": True,
        "expected_sha256": PINNED_EXPECTED_SHA256,
        "expected_size_bytes": PINNED_EXPECTED_SIZE_BYTES,
        "sft_adapter": {
            "path": sft_path,
            "sha256": sft_sha256,
            "size_bytes": sft_size,
            "git_blob_id": sft_blob,
            "matches_expected": sft_matches_pinned,
        },
        "rl_adapter": {
            "path": rl_path,
            "sha256": rl_sha256,
            "size_bytes": rl_size,
            "git_blob_id": rl_blob,
            "matches_expected": rl_matches_pinned,
        },
        "comparison": {
            "sha256_identical": hashes_match,
            "bytes_identical": bytes_identical,
            "git_blob_identical": (sft_blob == rl_blob and sft_blob != "N/A"),
            "root_cause_disclosure": (
                "Commit b4174fe initialized the SFT checkpoint directory by copying the adapter "
                "from the RL checkpoint directory. Consequently, on-disk adapter_model.safetensors "
                "files are byte-for-byte duplicate copies sharing identical weights."
            ),
            "scientific_interpretation": (
                "Evaluations under SFT Pass A and RL Pass B represent stochastic evaluation passes "
                "(T=0.7, top_p=0.8) over the locked 50 dev worlds under a single shared adapter checkpoint. "
                "They are not an SFT vs. RL post-training causal ablation."
            ),
        },
    }

    os.makedirs(os.path.dirname(output_artifact), exist_ok=True)
    with open(output_artifact, "w") as fp:
        json.dump(report, fp, indent=2)

    return report


def main():
    report = verify_checkpoints()
    print(json.dumps(report, indent=2))
    if report["status"] != "verified":
        print("\nERROR: Checkpoint verification failed!")
        if "errors" in report:
            for err in report["errors"]:
                print(f"  - {err}")
        sys.exit(1)
    elif not report.get("binaries_present_on_disk", False):
        print(
            "\nSUCCESS: Recorded-metadata inspection verified manifest against pinned checksums and sizes "
            "(note: local binaries absent; byte identity was not checked)."
        )
    else:
        print("\nSUCCESS: Binary re-hashing and byte-level comparison independently verified on local disk.")


if __name__ == "__main__":
    main()
