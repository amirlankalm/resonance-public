---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
- reinforcement-learning
- scientific-reasoning
- physics
---

# Resonance Qwen2.5-0.5B RL Checkpoint

This repository contains the LoRA adapter checkpoint trained using **Group-Relative Policy Gradient (GRPO-style advantage normalization)** on the **Resonance Active Circuit Spectroscopy** environment.

## Model Details

- **Base Model:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Apache-2.0 license)
- **Adapter Type:** LoRA (`peft` library)
- **Target Modules:** `q_proj`, `v_proj`
- **LoRA Hyperparameters:** $r=16$, $\alpha=32$, dropout $=0.05$
- **Language:** English
- **License:** Apache-2.0 (compatible with upstream Qwen2.5 base weights)
- **Repository:** https://github.com/amirlankalm/resonance

## Training Recipe

- **Algorithm:** Lightweight Group-Relative Policy Gradient (advantage-weighted cross-entropy with leave-one-out / group baseline normalization across $G=4$ parallel sensor rollouts).
- **Environment:** `CircuitSpectroscopyEnv` (driven series RLC spectroscopy under $V_{\text{drive}} = 0.1\text{ V}_{\text{rms}}$, 12-query measurement budget, 20-turn horizon).
- **Seeds:** Training seeds $30000+$, strictly isolated from development seeds ($10000..10049$).
- **Optimizer:** AdamW ($\text{lr} = 5\times 10^{-5}$, weight decay $0.01$).

## Checkpoint File Checksums (SHA-256)

| File | SHA-256 Checksum |
| :--- | :--- |
| `adapter_model.safetensors` | `22582b459d3523fc60ba7f7ec3afaf9792ec7b74e336fc75c26c5dcd74d459cd` |
| `adapter_config.json` | `cf58104381024781e248c2f1a9e9f9b6fb4b14f173ed4abdebeb48a48d0a9b24` |
| `tokenizer.json` | `3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8` |

## Checkpoint Provenance & Identity Disclosure (Research Preview)

- **Shared Checkpoint Binary:** Note that `checkpoints/resonance_qwen_rl/adapter_model.safetensors` shares the exact same SHA-256 digest (`22582b459d3523fc60ba7f7ec3afaf9792ec7b74e336fc75c26c5dcd74d459cd`) with `checkpoints/resonance_qwen_sft/adapter_model.safetensors`. In commit `b4174fe`, the SFT baseline folder was initialized by copying this adapter file.
- **Evaluation Mechanism:** The stored RL evaluation logs (`artifacts/model_eval_qwen_rl/`) represent an independent stochastic evaluation pass under this adapter checkpoint ($T=0.7$, $\text{top\_p}=0.8$) across the 50 locked worlds. The outcome divergence between SFT ($28$ submissions, $0.4833$ reward) and RL ($23$ submissions, $0.4024$ reward) reflects rollout sampling variability under identical weights, rather than a separate post-training weight state.
- **Research Preview Scope:** Released as part of the Resonance research preview to provide a reproducible audit of the recorded runs. Distinct two-stage training should be executed via `scripts/train_rl.py`.

## Evaluation Scope & Findings

Evaluated across 50 locked development worlds ($N=50$, seeds $10000..10049$):
- **Assisted Track:** Mean terminal reward **$0.4024$** (95% CI `[0.2624, 0.5367]`), type accuracy **$46.0\%$**, conditional NRMSE **$0.0330$** on completed episodes ($23/50$ submissions).
- **Strict Track:** Mean terminal reward **$-0.0480$**, $0/50$ submissions ($50/50$ budget overruns).
- **Key Takeaway:** In Pass B, completed episodes yielded low parameter estimation error ($0.0330$ conditional NRMSE), but recorded fewer overall submissions ($23/50$) due to probing near the resonance peak past the turn budget. Because Pass A and Pass B share identical adapter weights, this divergence demonstrates rollout trajectory variance rather than a comparative training effect.
