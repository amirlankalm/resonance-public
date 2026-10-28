# Resonance Evaluation Artifacts & Run Provenance

This directory stores evaluation logs, trajectory traces, benchmark comparison outputs, and cryptographic checksum records for the Resonance physics reinforcement learning environment.

## Historical Artifact Provenance & Legacy Fields

Historical run artifacts retain the legacy `budget_overrun` failure category. Use `stop_reason` for termination-cause analysis. In the assisted Pass A/B runs, all 22/27 non-submissions are `invalid_action_limit` stops. Legacy `budget_overrun` aggregates count non-submissions, not exclusively 20-turn horizon exhaustion. New runner outputs use the precise stop cause.

### Historical Dev World Runs Summary ($n = 50$ locked development worlds)

| Stored Run Directory | Track | Submitted | Mean Reward | Actual Stop Causes (`stop_reason`) | Legacy Aggregate (`failure_category`) |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `model_eval_qwen_sft/` (Assisted A) | assisted | 28 / 50 | 0.4833 | 28 submitted, 22 invalid_action_limit | 22 budget_overrun, 4 incorrect_parameter_fitting, 24 None |
| `model_eval_qwen_rl/` (Assisted B) | assisted | 23 / 50 | 0.4024 | 23 submitted, 27 invalid_action_limit | 27 budget_overrun, 23 None |
| `model_eval_qwen_base_strict/` | strict | 0 / 50 | -0.0500 | 50 invalid_action_limit | 50 budget_overrun |
| `model_eval_qwen_sft_strict/` (Strict A) | strict | 0 / 50 | -0.0490 | 49 invalid_action_limit, 1 turn_horizon | 50 budget_overrun |
| `model_eval_qwen_rl_strict/` (Strict B) | strict | 0 / 50 | -0.0480 | 48 invalid_action_limit, 2 turn_horizon | 50 budget_overrun |

- **Invalid-Action Stops:** Episodes halting because the model produced 5 consecutive schema violations (`stop_reason="invalid_action_limit"`).
- **Turn Horizon Stops:** Episodes halting because the 20-turn conversational horizon was reached without submitting (`stop_reason="turn_horizon"`).
- **Submissions:** Episodes that emitted a valid hypothesis submission (`stop_reason="submitted"`).
