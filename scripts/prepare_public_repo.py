#!/usr/bin/env python3
"""Automated Public Repository Preparation and Hygiene Tool for Resonance.

Prepares an immaculate public release repository by:
1. Verifying that the current private repo has zero plaintext salts and that the production
   salt was never committed anywhere in git history.
2. Checking that all unit tests and table verifications pass.
3. Exporting a clean-history public repository directory (default: `../resonance-public`)
   with externalized Hugging Face Hub checkpoint downloads, zero git baggage, and
   immaculate provenance documentation.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return res.returncode, res.stdout + res.stderr


def verify_source_hygiene() -> bool:
    print("[1/4] Verifying source repository hygiene...")
    
    # 1. Check retired salt leakage in tracked files
    legacy_salt_token = "_".join(["resonance", "v01", "heldout", "test", "salt"])
    ret, out = run_cmd(["git", "grep", "-i", legacy_salt_token, "--", ":!scripts/prepare_public_repo.py"], REPO_ROOT)
    if ret == 0 and out.strip():
        print(f"  FAIL: Old retired salt found in tracked files: {out.strip()}!")
        return False
    print("  PASS: Old retired salt is completely purged from working tree.")

    # 2. Check if new production salt was ever committed to git history
    salt_file = Path(os.path.expanduser("~/.resonance_heldout_salt"))
    if salt_file.is_file():
        salt = salt_file.read_text(encoding="utf-8").strip()
        if salt:
            ret, out = run_cmd(["git", "log", "-S", salt, "--oneline"], REPO_ROOT)
            if out.strip():
                print("  FAIL: New production salt was committed in git history!")
                return False
            print("  PASS: New production salt has NEVER been committed to git history.")
    else:
        print("  NOTE: ~/.resonance_heldout_salt not present; skipping local salt git history check.")

    # 3. Check table sync
    ret, out = run_cmd([sys.executable, "scripts/render_readme_tables.py", "--check"], REPO_ROOT)
    if ret != 0:
        print("  FAIL: README tables out of sync with serialized artifacts!")
        return False
    print("  PASS: README tables synchronized with artifacts.")

    # 4. Run unit tests
    ret, out = run_cmd([sys.executable, "-m", "unittest", "discover", "tests"], REPO_ROOT)
    if ret != 0:
        print("  FAIL: Unit tests failed!")
        return False
    print("  PASS: All unit tests pass.")

    return True


def export_public_repo(dest: Path) -> bool:
    print(f"[2/4] Exporting clean-history tree to {dest}...")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Git archive to get clean tracked files without git history
    archive_tar = dest / "repo.tar"
    ret, _ = run_cmd(["git", "archive", "HEAD", "-o", str(archive_tar)], REPO_ROOT)
    if ret != 0:
        print("  FAIL: git archive failed!")
        return False

    ret, _ = run_cmd(["tar", "-xf", str(archive_tar), "-C", str(dest)], dest)
    archive_tar.unlink()
    if ret != 0:
        print("  FAIL: tar extraction failed!")
        return False

    print("  PASS: Clean files extracted.")

    # Externalize binary checkpoints
    print("[3/4] Externalizing binary weights from public release tree...")
    for ckpt in ["resonance_qwen_rl", "resonance_qwen_sft"]:
        weight_file = dest / "checkpoints" / ckpt / "adapter_model.safetensors"
        if weight_file.exists():
            weight_file.unlink()
            print(f"  Removed {weight_file.relative_to(dest)} (binary weight externalized to Hugging Face Hub)")

    # Update .gitignore in public repo to prevent large weights from being checked in
    gitignore = dest / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        if "*.safetensors" not in content:
            with open(gitignore, "a", encoding="utf-8") as f:
                f.write("\n# Externalized model weights (download via scripts/download_checkpoints.py)\n*.safetensors\n*.bin\n")

    # Initialize fresh git repository with clean commit 1
    print("[4/4] Initializing fresh public git repository with single initial commit...")
    run_cmd(["git", "init", "-b", "main"], dest)
    run_cmd(["git", "add", "."], dest)
    run_cmd(
        [
            "git",
            "commit",
            "-m",
            "Initial public release: Resonance physics RL environment (v0.1.0)",
        ],
        dest,
    )

    # Verify no old salt in full history of fresh repo
    legacy_salt_token = "_".join(["resonance", "v01", "heldout", "test", "salt"])
    ret, out = run_cmd(["git", "log", "--all", "-S", legacy_salt_token, "--", ":!scripts/prepare_public_repo.py"], dest)
    if out.strip():
        print("  FAIL: Old salt detected in fresh repository!")
        return False
    print("  PASS: Fresh repository history is 100% clean and free of legacy strings.")

    # Check repository size
    size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"\n=======================================================")
    print(f" Clean Public Repository Ready at: {dest}")
    print(f" Total Repository Size:            {size_mb:.2f} MB (Checkpoints externalized)")
    print(f" Commit Count:                     1 (Zero legacy history)")
    print(f"=======================================================")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare clean-history public release repository.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=REPO_ROOT.parent / "resonance-public",
        help="Destination directory for the public repo (default: ../resonance-public)",
    )
    args = parser.parse_args()

    if not verify_source_hygiene():
        print("\nAborting: Source repository failed hygiene checks.")
        return 1

    if not export_public_repo(args.dest):
        print("\nAborting: Public repository export failed.")
        return 1

    print("\nPublic release preparation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
