# Contributing to Resonance

Thank you for your interest in contributing to **Resonance**! We welcome contributions from researchers, machine learning engineers, and physicists interested in active experimental design, verifiable reinforcement learning environments, and scientific system identification.

---

## Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating in discussions or opening pull requests.

---

## Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, and 3.12)
- Git and virtual environment tools (`venv` or `conda`)
- (Optional) PyTorch with Apple Silicon MPS or CUDA support for model training

### 2. Local Setup
```bash
# Clone repository
git clone https://github.com/amirlankalm/resonance.git
cd resonance

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (core environment + physics simulation)
pip install -e .

# Or install with ML dependencies (PyTorch, Hugging Face Transformers, PEFT)
pip install -e '.[ml]'
```

### 3. Verify Local Setup
Run the full test suite to confirm everything is working properly:
```bash
# Run unit & regression tests
python3 -m unittest discover tests

# Starter Malus' Law environment smoke tests
python3 -m unittest -v test_environment.py

# Verify benchmark tables in README match evaluation artifacts
python3 scripts/render_readme_tables.py --check
```

---

## Development Workflow

1. **Fork & Branch:** Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make Atomic Changes:** Keep commits focused, well-documented, and descriptive.
3. **Follow Architectural Conventions:**
   - **Environment Isolation:** Environments must never leak hidden ground-truth physical parameters through observations or public attributes. Use private variables (`_hidden_params`, `_eval_grid`, `_rng`) and the public observation API.
   - **RNG Stream Separation:** Use isolated RNG streams (`IsolatedRNGManager`) with SHA-256 derived seeds to ensure world sampling and sensor noise remain statistically independent.
   - **Schema Rigor:** All actions must validate against strict Pydantic schemas with `extra='forbid'` to prevent malformed or malicious payload injections.
   - **Fail-Closed Principle:** When a model generates malformed actions, consume turn limits and apply `INVALID_ACTION_PENALTY` (-0.05) without crashing the runtime.
4. **Run Verification:** Ensure all tests pass and documentation tables are synchronized.
5. **Submit a Pull Request:** Open a PR against `main` using our [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).

---

## Coding Standards

- **Formatting & Style:** Adhere to PEP 8 standards with explicit type annotations on public functions.
- **Documentation:** Preserve existing comments and docstrings. Document new CLI parameters and environment configurations.
- **Secrets & Privacy:** Never commit `.env` files, credentials, private API keys, or absolute user file paths. Ensure `.gitignore` rules are respected.
- **Auditability:** Any new evaluation or benchmark artifact must store complete episode records and summary statistics, keeping individual files under 1 MB to ensure seamless GitHub API and clone auditability.

---

## Adding New Physical Systems

We actively welcome contributions of new physical environments for active experimental reasoning (e.g., optical interferometry, acoustic resonance, fluid damping). New environments should:
1. Inherit from `gymnasium.Env` or `resonance.envs.base.BaseExperimentEnv`.
2. Implement discrete or continuous probe actions within a physically realistic parameter regime.
3. Provide realistic sensor noise and detection floors.
4. Supply unit tests verifying reproducibility, state isolation, and budget enforcement.

---

## Reporting Issues

If you discover a bug, unexpected behavior, or statistical discrepancy, please file an issue using our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md). For feature suggestions or benchmark ideas, please use the [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md).

For security vulnerabilities, please refer to [SECURITY.md](SECURITY.md).
