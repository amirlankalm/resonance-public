# Failure Diagnosis Report: unified_eval_Qwen2.5-0.5B (SFT-Only)_assisted

**Model:** checkpoints/resonance_qwen_sft
**Total Episodes:** 50
**Mean Reward:** 0.4833

## Failure Breakdown by Category

| Failure Category | Count | Percentage |
| :--- | :--- | :--- |
| `budget_overrun` | 22 | 44.0% |
| `None` | 24 | 48.0% |
| `incorrect_parameter_fitting` | 4 | 8.0% |

## Detailed Failure Case Studies

### Case 1: World dev_world_0
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 14 turns without submitting.
- **Calls Made:** 9
- **Raw Error:** `None`

### Case 2: World dev_world_7
- **Category:** `incorrect_parameter_fitting`
- **Suspected Cause:** NRMSE of 0.6106 exceeded threshold 0.15.
- **Calls Made:** 4
- **Raw Error:** `None`

### Case 3: World dev_world_10
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 16 turns without submitting.
- **Calls Made:** 11
- **Raw Error:** `None`

### Case 4: World dev_world_11
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 10 turns without submitting.
- **Calls Made:** 5
- **Raw Error:** `None`

### Case 5: World dev_world_17
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 14 turns without submitting.
- **Calls Made:** 9
- **Raw Error:** `None`

### Case 6: World dev_world_19
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 15 turns without submitting.
- **Calls Made:** 10
- **Raw Error:** `None`

### Case 7: World dev_world_21
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 10 turns without submitting.
- **Calls Made:** 5
- **Raw Error:** `None`

### Case 8: World dev_world_22
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 7 turns without submitting.
- **Calls Made:** 2
- **Raw Error:** `None`

### Case 9: World dev_world_23
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 8 turns without submitting.
- **Calls Made:** 3
- **Raw Error:** `None`

### Case 10: World dev_world_25
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 17 turns without submitting.
- **Calls Made:** 9
- **Raw Error:** `None`
