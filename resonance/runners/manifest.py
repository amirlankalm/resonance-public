"""Run artifact logger and manifest generator for Resonance.

Produces the 4 required run artifacts:
1. run_manifest.json
2. episodes.jsonl
3. summary.json
4. failures.md
"""

from __future__ import annotations
import json
import os
import platform
import time
from typing import Dict, Any, List, Optional
import numpy as np


class RunArtifactLogger:
    """Records and exports full diagnostic run artifacts."""

    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        model_name: str,
        decoding_params: Dict[str, Any],
        prompt_text: str,
        task_name: str = "circuit_spectroscopy",
        mode: str = "assisted",
        is_mock: bool = False,
        loader_confirmed: Optional[bool] = None,
    ):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.decoding_params = decoding_params
        self.prompt_text = prompt_text
        self.task_name = task_name
        self.mode = mode
        self.is_mock = is_mock
        self.loader_confirmed = loader_confirmed

        os.makedirs(self.output_dir, exist_ok=True)
        self.episodes: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def set_loader_confirmed(self, confirmed: Optional[bool]) -> None:
        """Set explicit runtime model loader verification state."""
        self.loader_confirmed = bool(confirmed) if confirmed is not None else None

    def record_episode(self, episode_data: Dict[str, Any]) -> None:
        """Record a completed episode trace."""
        self.episodes.append(episode_data)
        if episode_data.get("failure_category"):
            self.failures.append(episode_data)

    def write_artifacts(self) -> Dict[str, str]:
        """Write all 4 artifact files to output_dir."""
        elapsed = time.time() - self.start_time

        # Git commit provenance
        git_commit = "unknown"
        try:
            import subprocess
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            pass

        # Checkpoint hash provenance (R7 & Oct 26 Review P1)
        checkpoint_hash = None
        hash_target = None
        weight_sha256 = None
        adapter_config_sha256 = None
        tokenizer_hash = None

        if os.path.exists(self.model_name):
            import hashlib
            try:
                def _file_hash(p: str) -> Optional[str]:
                    if os.path.exists(p):
                        h = hashlib.sha256()
                        with open(p, "rb") as fp:
                            while chunk := fp.read(65536):
                                h.update(chunk)
                        return h.hexdigest()
                    return None

                if os.path.isdir(self.model_name):
                    for fname in ["adapter_model.safetensors", "adapter_model.bin", "model.safetensors"]:
                        fpath = os.path.join(self.model_name, fname)
                        if os.path.exists(fpath):
                            hash_target = fpath
                            weight_sha256 = _file_hash(fpath)
                            checkpoint_hash = weight_sha256
                            break
                    cfg_path = os.path.join(self.model_name, "adapter_config.json")
                    if os.path.exists(cfg_path):
                        adapter_config_sha256 = _file_hash(cfg_path)
                    tok_path = os.path.join(self.model_name, "tokenizer.json")
                    if os.path.exists(tok_path):
                        tokenizer_hash = _file_hash(tok_path)
                    # Note: We strictly separate config hashing from weight provenance (Oct 26 Review P1).
                    # If weight binaries are missing, weight_sha256 and checkpoint_hash remain None.
                elif os.path.isfile(self.model_name):
                    hash_target = self.model_name
                    checkpoint_hash = _file_hash(self.model_name)
                    weight_sha256 = checkpoint_hash
            except Exception:
                pass

        # Software dependency versions (R7)
        def _get_ver(pkg: str) -> Optional[str]:
            try:
                import importlib.metadata
                return importlib.metadata.version(pkg)
            except Exception:
                return None

        # load_confirmation is an explicit loader runtime signal, not inferred from file existence (Oct 26 Review P1)
        load_confirmation = self.loader_confirmed if self.loader_confirmed is not None else False
        if self.is_mock:
            load_confirmation = False

        # 1. run_manifest.json
        manifest = {
            "experiment_name": self.experiment_name,
            "task_name": self.task_name,
            "model_name": self.model_name,
            "mode": self.mode,
            "is_mock": self.is_mock,
            "git_commit": git_commit,
            "checkpoint_hash": checkpoint_hash,
            "hash_target": hash_target,
            "weight_sha256": weight_sha256,
            "adapter_config_sha256": adapter_config_sha256,
            "tokenizer_hash": tokenizer_hash,
            "load_confirmation": load_confirmation,
            "system_prompt": self.prompt_text,
            "decoding_params": self.decoding_params,
            "prompt_length_chars": len(self.prompt_text),
            "software_versions": {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
                "torch": _get_ver("torch"),
                "transformers": _get_ver("transformers"),
                "peft": _get_ver("peft"),
                "pydantic": _get_ver("pydantic"),
                "numpy": _get_ver("numpy"),
                "scipy": _get_ver("scipy"),
                "psutil": _get_ver("psutil"),
            },
            "total_episodes": len(self.episodes),
            "elapsed_seconds": round(elapsed, 3),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest_path = os.path.join(self.output_dir, "run_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # 2. episodes.jsonl
        episodes_path = os.path.join(self.output_dir, "episodes.jsonl")
        with open(episodes_path, "w") as f:
            for ep in self.episodes:
                # Convert any non-serializable objects and compact observation logging
                compact_ep = self._compact_episode(ep)
                clean_ep = self._make_serializable(compact_ep)
                f.write(json.dumps(clean_ep) + "\n")

        # 3. summary.json
        summary = self._compute_summary()
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # 4. failures.md
        failures_path = os.path.join(self.output_dir, "failures.md")
        self._write_failures_md(failures_path, summary)

        return {
            "manifest": manifest_path,
            "episodes": episodes_path,
            "summary": summary_path,
            "failures": failures_path,
        }

    def _compute_summary(self) -> Dict[str, Any]:
        N = len(self.episodes)
        if N == 0:
            return {"total_episodes": 0}

        terminal_rewards = [ep.get("terminal_reward", ep.get("reward", 0.0)) for ep in self.episodes]
        episode_returns = [ep.get("episode_return", ep.get("reward", 0.0)) for ep in self.episodes]
        calls = [ep.get("calls_made", 0) for ep in self.episodes]
        turns = [ep.get("steps", 0) for ep in self.episodes]
        valid_rates = [ep.get("valid_action_rate", 1.0) for ep in self.episodes]

        # Submissions
        submitted_eps = [ep for ep in self.episodes if ep.get("submitted") is True]
        n_submitted = len(submitted_eps)

        # Unconditional accuracy: denominator is ALL assigned worlds N
        unconditional_correct = sum(1.0 for ep in self.episodes if ep.get("is_correct_type") is True)
        unconditional_type_acc = unconditional_correct / N

        # Conditional metrics on submitted worlds
        submitted_nrmses = [ep["nrmse"] for ep in submitted_eps if ep.get("nrmse") is not None]
        submitted_briers = [ep["brier_score"] for ep in submitted_eps if ep.get("brier_score") is not None]

        category_counts: Dict[str, int] = {}
        for ep in self.episodes:
            cat = ep.get("failure_category", "none")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_episodes": N,
            "submitted_episodes": n_submitted,
            "submission_rate": float(n_submitted / N),
            "mean_terminal_reward": float(np.mean(terminal_rewards)),
            "terminal_reward_std": float(np.std(terminal_rewards)),
            "mean_episode_return": float(np.mean(episode_returns)),
            "episode_return_std": float(np.std(episode_returns)),
            "mean_reward": float(np.mean(terminal_rewards)),  # backwards compatibility
            "reward_std": float(np.std(terminal_rewards)),
            "unconditional_type_accuracy": float(unconditional_type_acc),
            "conditional_mean_nrmse": float(np.mean(submitted_nrmses)) if submitted_nrmses else None,
            "conditional_mean_brier_score": float(np.mean(submitted_briers)) if submitted_briers else None,
            "type_classification_accuracy": float(unconditional_type_acc),  # backwards compatibility
            "mean_nrmse": float(np.mean(submitted_nrmses)) if submitted_nrmses else None,
            "mean_brier_score": float(np.mean(submitted_briers)) if submitted_briers else None,
            "mean_valid_action_rate": float(np.mean(valid_rates)),
            "mean_instrument_calls": float(np.mean(calls)),
            "mean_turn_steps": float(np.mean(turns)),
            "failure_counts": category_counts,
        }

    def _write_failures_md(self, filepath: str, summary: Dict[str, Any]) -> None:
        lines = [
            f"# Failure Diagnosis Report: {self.experiment_name}",
            "",
            f"**Model:** {self.model_name}",
            f"**Total Episodes:** {summary.get('total_episodes', 0)}",
            f"**Mean Reward:** {summary.get('mean_reward', 0.0):.4f}",
            "",
            "## Failure Breakdown by Category",
            "",
            "| Failure Category | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ]
        N = summary.get("total_episodes", 1) or 1
        for cat, cnt in summary.get("failure_counts", {}).items():
            pct = (cnt / N) * 100
            lines.append(f"| `{cat}` | {cnt} | {pct:.1f}% |")

        lines.extend([
            "",
            "## Detailed Failure Case Studies",
            "",
        ])

        if not self.failures:
            lines.append("No failures recorded in this run. All episodes completed successfully.")
        else:
            for idx, ep in enumerate(self.failures[:10]):  # Top 10 sample failures
                lines.extend([
                    f"### Case {idx + 1}: World {ep.get('world_id', 'unknown')}",
                    f"- **Category:** `{ep.get('failure_category')}`",
                    f"- **Suspected Cause:** {ep.get('failure_detail', 'N/A')}",
                    f"- **Calls Made:** {ep.get('calls_made', 0)}",
                    f"- **Raw Error:** `{ep.get('error', 'None')}`",
                    "",
                ])

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    def _compact_episode(self, ep: Dict[str, Any]) -> Dict[str, Any]:
        """Compact episode record to prevent quadratic blowup of static prompt formulas in observation logs.
        
        Preserves all actions, measurements, completions, rewards, parameters, and metrics,
        while keeping file size comfortably under GitHub's 1MB API limit for auditability.
        """
        clean_ep = dict(ep)
        if "observations" in clean_ep and isinstance(clean_ep["observations"], list):
            compact_obs = []
            for i, obs in enumerate(clean_ep["observations"]):
                if isinstance(obs, dict):
                    hist = obs.get("history", [])
                    latest = hist[-1] if hist else None
                    c = {
                        "step": i,
                        "remaining_calls": obs.get("remaining_calls"),
                        "remaining_turns": obs.get("remaining_turns"),
                    }
                    if latest:
                        c["measurement"] = latest
                    compact_obs.append(c)
                else:
                    compact_obs.append(obs)
            clean_ep["observations"] = compact_obs
        return clean_ep

    def _make_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(x) for x in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj) if np.isfinite(obj) else None
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return self._make_serializable(obj.tolist())
        return obj
