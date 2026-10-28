"""Sandboxed bounded execution runner for Resonance."""

from __future__ import annotations
import time
from typing import Dict, Any, Optional, List
import numpy as np

from resonance.envs.base import BasePhysicsEnv
from resonance.models.base import BaseAgent
from resonance.runners.manifest import RunArtifactLogger


class EpisodeRunner:
    """Orchestrates an agent interacting with a physics environment."""

    def __init__(
        self,
        env: BasePhysicsEnv,
        agent: BaseAgent,
        artifact_logger: Optional[RunArtifactLogger] = None,
        execution_mode: Optional[str] = None,
    ):
        self.env = env
        self.agent = agent
        self.artifact_logger = artifact_logger
        self.execution_mode = execution_mode
        if self.artifact_logger is not None and hasattr(self.agent, "loader_confirmed"):
            self.artifact_logger.set_loader_confirmed(self.agent.loader_confirmed)

    def run_episode(
        self,
        world_seed: Optional[int] = None,
        world_id: str = "world_0",
        sensor_seed: Optional[int] = None,
        execution_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a single sandboxed episode."""
        start_time = time.time()
        obs, info = self.env.reset(seed=world_seed)
        if sensor_seed is not None and hasattr(self.env, "reseed_sensor"):
            self.env.reseed_sensor(sensor_seed)
            obs = self.env.get_public_observation()

        self.agent.reset()

        actions_log: List[Dict[str, Any]] = []
        observations_log: List[Dict[str, Any]] = [obs]
        errors_log: List[str] = []
        step_rewards: List[float] = []
        raw_completions: List[str] = []
        total_episode_return: float = 0.0
        invalid_action_count: int = 0
        consecutive_invalid: int = 0
        episode_assisted: bool = False
        fit_source: Optional[str] = None

        terminated = False
        truncated = False
        reward = 0.0
        final_info: Dict[str, Any] = {}

        while not (terminated or truncated):
            # Agent action
            try:
                action = self.agent.act(obs)
            except Exception as e:
                action = {"type": "invalid", "error": f"Agent exception: {e}"}
                errors_log.append(str(e))

            actions_log.append(action)

            # Store every raw completion from agent (R7)
            raw_text = getattr(self.agent, "last_raw_completion", None)
            if raw_text is not None and str(raw_text).strip() != "":
                raw_completions.append(str(raw_text))
            elif isinstance(action, dict) and "raw" in action:
                raw_completions.append(str(action["raw"]))
            else:
                raw_completions.append(json.dumps(action) if isinstance(action, dict) else str(action))

            # Track assistance metadata separately from action payload (R1)
            meta = getattr(self.agent, "last_action_metadata", {}) or {}
            if meta.get("assisted"):
                episode_assisted = True
                fit_source = meta.get("fit_source", "numerical_fitter")

            # Step environment
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            observations_log.append(obs)
            step_rewards.append(float(reward))
            total_episode_return += float(reward)

            if not step_info.get("is_valid", True) or (isinstance(action, dict) and action.get("type") == "invalid"):
                invalid_action_count += 1
                consecutive_invalid += 1
            else:
                consecutive_invalid = 0

            if "error" in step_info:
                errors_log.append(step_info["error"])

            if terminated or truncated:
                final_info = step_info
                break

            # Fail fast if agent is stalled emitting repeated invalid syntax actions in strict mode
            if consecutive_invalid >= 5:
                truncated = True
                final_info = {"error": "Truncated after 5 consecutive invalid action schema violations.", "is_valid": False}
                break

        elapsed = time.time() - start_time
        terminal_reward = float(reward)
        episode_return = float(total_episode_return)
        submitted = bool(terminated and any(isinstance(a, dict) and a.get("type") == "submit" for a in actions_log))
        valid_action_rate = (len(actions_log) - invalid_action_count) / max(1, len(actions_log))

        # Determine exact stop reason (R8): canonical precise source of episode termination
        if submitted:
            stop_reason = "submitted"
        elif consecutive_invalid >= 5:
            stop_reason = "invalid_action_limit"
        elif self.env.step_count >= getattr(self.env, "max_turns", 20):
            stop_reason = "turn_horizon"
        elif self.env.instrument_calls >= getattr(self.env, "max_calls", 12):
            stop_reason = "measurement_limit"
        elif truncated:
            stop_reason = "invalid_action_limit" if invalid_action_count >= 5 else "turn_horizon"
        else:
            stop_reason = "terminated"

        # Failure classification:
        # stop_reason is the canonical, precise source of episode termination.
        # failure_category preserves backward compatibility where non-submission truncations
        # are mapped to their specific stop reason, and legacy aggregates represent non-submission count.
        failure_category = None
        failure_detail = None

        if truncated and not terminated:
            failure_category = stop_reason if stop_reason in ("invalid_action_limit", "turn_horizon", "measurement_limit") else "non_submission_truncation"
            if stop_reason == "invalid_action_limit":
                failure_detail = f"Episode truncated early after {consecutive_invalid} consecutive schema violations."
            elif stop_reason == "turn_horizon":
                failure_detail = f"Episode truncated upon reaching the 20-turn environment horizon without submitting."
            elif stop_reason == "measurement_limit":
                failure_detail = f"Episode reached 12-call measurement limit without submitting."
            else:
                failure_detail = f"Episode truncated at {self.env.step_count} turns without submitting."
        elif errors_log and not terminated:
            failure_category = "parser_schema_failure"
            failure_detail = f"Schema or parsing errors: {errors_log[:2]}"
        elif terminated:
            if not final_info.get("is_correct_type", True):
                failure_category = "wrong_physics_assumptions"
                failure_detail = f"Classified type as {final_info.get('predicted', {}).get('circuit_type')}, but truth was different."
            elif final_info.get("nrmse", 0.0) > 0.15:
                failure_category = "incorrect_parameter_fitting"
                failure_detail = f"NRMSE of {final_info.get('nrmse', 0.0):.4f} exceeded threshold 0.15."

        mode_val = (
            execution_mode
            or self.execution_mode
            or getattr(self.agent, "track", None)
            or getattr(self.agent, "mode", None)
            or ("assisted" if episode_assisted else "strict")
        )
        if mode_val not in ("strict", "assisted"):
            mode_val = "assisted" if episode_assisted else "strict"

        solver_used = bool(fit_source is not None or (episode_assisted and submitted))

        episode_record = {
            "world_id": world_id,
            "world_seed": world_seed,
            "sensor_seed": sensor_seed,
            "steps": self.env.step_count,
            "calls_made": self.env.instrument_calls,
            "terminated": terminated,
            "truncated": truncated,
            "submitted": submitted,
            "terminal_reward": terminal_reward,
            "episode_return": episode_return,
            "reward": terminal_reward,  # backwards compatibility
            "step_rewards": step_rewards,
            "execution_mode": mode_val,
            "assisted": episode_assisted,
            "solver_used": solver_used,
            "fit_source": fit_source,
            "stop_reason": stop_reason,
            "invalid_action_count": invalid_action_count,
            "valid_action_rate": valid_action_rate,
            "nrmse": final_info.get("nrmse"),
            "rmse": final_info.get("rmse"),
            "brier_score": final_info.get("brier_score"),
            "is_correct_type": final_info.get("is_correct_type"),
            "failure_category": failure_category,
            "failure_detail": failure_detail,
            "elapsed_seconds": round(elapsed, 4),
            "actions": actions_log,
            "observations": observations_log,
            "raw_completions": raw_completions,
            "errors": errors_log,
        }

        if self.artifact_logger:
            self.artifact_logger.record_episode(episode_record)

        return episode_record
