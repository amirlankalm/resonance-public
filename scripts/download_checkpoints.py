#!/usr/bin/env python3
"""Download and verify Resonance model checkpoints from Hugging Face Hub or official mirror.

This script fetches binary adapter weights into `checkpoints/` and performs
cryptographic SHA-256 integrity checks against committed golden hashes.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Optional
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_CONFIGS = {
    "resonance_qwen_rl": {
        "repo_id": "harriot-ai/resonance-qwen2.5-0.5b-adapter",
        "filename": "adapter_model.safetensors",
        "expected_sha256": "22582b459d3523fc60ba7f7ec3afaf9792ec7b74e336fc75c26c5dcd74d459cd",
        "expected_bytes": 8676008,
    },
    "resonance_qwen_sft": {
        "repo_id": "harriot-ai/resonance-qwen2.5-0.5b-adapter",
        "filename": "adapter_model.safetensors",
        "expected_sha256": "22582b459d3523fc60ba7f7ec3afaf9792ec7b74e336fc75c26c5dcd74d459cd",
        "expected_bytes": 8676008,
    },
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(ckpt_name: str, target_dir: Path) -> bool:
    cfg = CHECKPOINT_CONFIGS[ckpt_name]
    target_file = target_dir / cfg["filename"]
    if not target_file.is_file():
        return False
    digest = compute_sha256(target_file)
    size = target_file.stat().st_size
    return digest == cfg["expected_sha256"] and size == cfg["expected_bytes"]


def download_checkpoint(ckpt_name: str, target_dir: Path, force: bool = False, token: Optional[str] = None) -> bool:
    cfg = CHECKPOINT_CONFIGS[ckpt_name]
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / cfg["filename"]

    if target_file.is_file() and not force:
        if verify_checkpoint(ckpt_name, target_dir):
            print(f"[{ckpt_name}] Checkpoint already present and verified (SHA-256: {cfg['expected_sha256'][:8]}...)")
            return True
        else:
            print(f"[{ckpt_name}] Existing file failed verification, redownloading...")

    # Primary URL: Hugging Face Hub direct resolve
    url = f"https://huggingface.co/{cfg['repo_id']}/resolve/main/{cfg['filename']}"
    print(f"[{ckpt_name}] Attempting download from {url}...")
    headers = {}
    auth_token = token or os.environ.get("HF_TOKEN")
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(target_file, "wb") as out_f:
            while chunk := resp.read(65536):
                out_f.write(chunk)

        if verify_checkpoint(ckpt_name, target_dir):
            print(f"[{ckpt_name}] Download verified successfully.")
            return True
        else:
            print(f"[{ckpt_name}] Checksum mismatch after download!")
            if target_file.exists():
                target_file.unlink()
            return False
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            print(f"[{ckpt_name}] Note: Checkpoint weights at '{cfg['repo_id']}' are currently unavailable in this public research preview (HTTP {e.code}).")
            print(f"[{ckpt_name}] The Resonance physics environment and numerical baselines run locally without hosted APIs or downloaded checkpoints, after installing runtime dependencies. Training requires the ML extras.")
            print(f"[{ckpt_name}] Manifest checksums can be independently verified via: python3 scripts/verify_checkpoint_identity.py")
            if target_file.exists() and target_file.stat().st_size == 0:
                target_file.unlink()
            return False
        else:
            print(f"[{ckpt_name}] Download failed with HTTP {e.code}: {e.reason}")
            return False
    except Exception as e:
        print(f"[{ckpt_name}] Download failed ({e}). Environment and baselines remain fully functional without checkpoints.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify Resonance model checkpoints.")
    parser.add_argument("--check-only", action="store_true", help="Only verify existing checkpoints without downloading.")
    parser.add_argument("--force", action="store_true", help="Force redownload even if present.")
    parser.add_argument("--token", type=str, default=None, help="Optional Hugging Face authentication token.")
    args = parser.parse_args()

    manifest_path = REPO_ROOT / "artifacts" / "checkpoint_binary_hashes.json"
    manifest_valid = False
    manifest_errors: list[str] = []
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)
            try:
                from scripts.verify_checkpoint_identity import validate_manifest_metadata
            except ImportError:
                from verify_checkpoint_identity import validate_manifest_metadata
            manifest_valid, manifest_errors = validate_manifest_metadata(manifest_data)
        except Exception as e:
            manifest_errors = [str(e)]

    all_ok = True
    for ckpt_name, cfg in CHECKPOINT_CONFIGS.items():
        ckpt_dir = REPO_ROOT / "checkpoints" / ckpt_name
        if args.check_only:
            target_file = ckpt_dir / cfg["filename"]
            if target_file.is_file():
                if verify_checkpoint(ckpt_name, ckpt_dir):
                    status = f"OK (local binary verified, {cfg['expected_bytes']} bytes)"
                    ok = True
                else:
                    status = "FAIL (local binary checksum or size mismatch)"
                    ok = False
            else:
                if manifest_valid:
                    status = (
                        f"NOT PRESENT ON DISK (externalized in preview; recorded metadata verified: "
                        f"{cfg['expected_sha256'][:8]}..., byte identity not checked)"
                    )
                else:
                    status = f"NOT PRESENT ON DISK (manifest validation failed: {'; '.join(manifest_errors)})"
                ok = False
            print(f"{ckpt_name:20s}: {status}")
            if not ok:
                all_ok = False
        else:
            if not download_checkpoint(ckpt_name, ckpt_dir, force=args.force, token=args.token):
                all_ok = False

    if args.check_only:
        if all_ok:
            print("\nSUCCESS: All checkpoints present and verified on local disk.")
            return 0
        else:
            print("\nNOTE: One or more checkpoint binaries are not present and verified on local disk.")
            return 1

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
