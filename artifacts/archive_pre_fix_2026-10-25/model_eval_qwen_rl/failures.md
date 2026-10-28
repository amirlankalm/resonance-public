# Failure Diagnosis Report: unified_eval_Resonance RL (Qwen-2.5-0.5B)

**Model:** checkpoints/resonance_qwen_rl
**Total Episodes:** 10
**Mean Reward:** 0.8296

## Failure Breakdown by Category

| Failure Category | Count | Percentage |
| :--- | :--- | :--- |
| `None` | 9 | 90.0% |
| `budget_overrun` | 1 | 10.0% |

## Detailed Failure Case Studies

### Case 1: World dev_world_3
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 20 turns without submitting.
- **Calls Made:** 10
- **Raw Error:** `None`
