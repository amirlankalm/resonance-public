# Failure Diagnosis Report: qwen_active_circuit_eval

**Model:** Qwen/Qwen2.5-0.5B-Instruct
**Total Episodes:** 5
**Mean Reward:** 0.3474

## Failure Breakdown by Category

| Failure Category | Count | Percentage |
| :--- | :--- | :--- |
| `budget_overrun` | 3 | 60.0% |
| `None` | 1 | 20.0% |
| `wrong_physics_assumptions` | 1 | 20.0% |

## Detailed Failure Case Studies

### Case 1: World dev_world_0
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 20 turns without submitting.
- **Calls Made:** 12
- **Raw Error:** `None`

### Case 2: World dev_world_2
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 20 turns without submitting.
- **Calls Made:** 12
- **Raw Error:** `None`

### Case 3: World dev_world_3
- **Category:** `budget_overrun`
- **Suspected Cause:** Episode truncated at 20 turns without submitting.
- **Calls Made:** 12
- **Raw Error:** `None`

### Case 4: World dev_world_4
- **Category:** `wrong_physics_assumptions`
- **Suspected Cause:** Classified type as RLC, but truth was different.
- **Calls Made:** 0
- **Raw Error:** `None`
