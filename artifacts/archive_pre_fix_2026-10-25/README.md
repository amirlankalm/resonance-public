# Archive of Pre-Fix Evaluation Artifacts (25 October 2026)

These run artifacts were produced prior to the technical review and bug fixes implemented on 25 October 2026:
- **Inverted Policy Gradient Loss Sign:** Initial checkpoint was trained with `loss = -adv * out.loss` (reducing probabilities of positive-advantage actions).
- **Non-Greedy JSON Parsing:** Actions were parsed using non-greedy regex `r"\{.*?\}"` which truncated nested JSON.
- **Assisted Fitting Asymmetry:** Evaluation silently used numerical fitting in a single track without strict/assisted track separation.
- **Sensor Reseed Overwrite:** Sensor RNG reseed was overwritten during multi-turn group rollouts.
- **Metric Accounting:** Failure to separate `terminal_reward` from `episode_return` and improper unconditional accuracy denominators.

All subsequent evaluations must use the corrected codebase and explicit `strict` and `assisted` evaluation tracks.
