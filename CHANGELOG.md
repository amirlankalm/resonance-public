# Changelog

All notable changes to the Resonance project are documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-27

### Added
- **Public API & Gymnasium Integration:** Registered `Resonance/CircuitSpectroscopy-v0` and `Resonance/Polarization-v0` environments with Gymnasium registry. Passed `gymnasium.utils.env_checker.check_env` with zero warnings.
- **External Lab Adapters:** Added Prime Intellect Verifiers adapter (`resonance/integrations/verifiers.py`) and Nous Research Atropos multi-turn environment (`resonance/integrations/atropos.py`).
- **Unified CLI:** Added `resonance eval`, `resonance report`, `resonance verify`, and `resonance doctor` subcommands.
- **One-Command Smoke Suite:** Implemented `python -m resonance.smoke` running teacher and random baselines in < 2 seconds on CPU.
- **Scripted Strict Reference Policy:** Implemented `resonance/baselines/strict_reference.py` demonstrating a proven 100% submission rate and 0.9294 mean reward ceiling in the strict unassisted JSON harness.
- **Adversarial Hacking Defense:** Implemented `tests/test_reward_hacking.py` covering 13 METR-style reward gaming vectors.
- **Golden Determinism Verification:** Checked in 20 golden worlds in `tests/golden/circuit_worlds_20_golden.json` with 1e-9 relative tolerance tests.
- **Split Management:** Implemented `resonance/splits.py` with rotated cryptographic test salt commitment (`1b747fa6aa61cefdd760780b659a96962faf57c6ebcef3c3ef3461ded79fa90a`) and pairwise disjointness assertions. Plaintext salt is maintained strictly outside the repository.
- **Salt Rotation & External Storage:** Rotated test split salt following external security review; purged plaintext strings from test files and committed SHA-256 zero-knowledge commitment.
- **Physics Derivations & Validation:** Created `docs/PHYSICS.md` and `tests/test_physics_validation.py` comparing against independent complex nodal solver across 100 random circuits to 1e-9 relative tolerance.
- **Comprehensive Documentation:** Added `docs/ENV_SPEC.md`, `docs/EVAL_PROTOCOL.md`, `docs/KNOWN_LIMITATIONS.md`, `docs/REWARD_HACKING.md`, and `docs/ENV_CARD.md`.

### Changed
- **Checkpoint Lineage Transparency:** Disclosed identical SHA-256 binary hash (`22582b45...`) between SFT and RL adapter files and added `tests/test_checkpoint_lineage.py`.
- **License and Security Hygiene:** Synchronized name ("Harriot Resonance Team") and version (`0.1.0`) across `LICENSE`, `SECURITY.md`, and `pyproject.toml`.
- **Starter Script Relocation:** Moved root starter scripts `environment.py` and `test_environment.py` into `examples/`.
