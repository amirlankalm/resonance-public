"""Two-stage SFT + Multi-Turn Group-Relative Policy Gradient Training Pipeline for Resonance.

Trains Qwen2.5-0.5B-Instruct locally on Apple Silicon MPS / CPU:
Stage 1: Supervised Fine-Tuning (SFT) on scripted numerical-teacher demonstrations (adaptive design + least squares,
         not human expert verification) with assistant token loss masking.
Stage 2: Group-Relative Policy Gradient (advantage-normalized REINFORCE across parallel rollouts;
         a lightweight training recipe without PPO ratio clipping or reference-model KL, not an exact standard GRPO replication).
"""

from __future__ import annotations
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.training.trajectory_collector import MultiTurnTrajectoryCollector
from resonance.training.grpo import compute_grpo_advantages
from resonance.core.memory import MemoryGuard
from resonance.core.prompts import (
    SYSTEM_PROMPT_CIRCUIT,
    format_observation_user_prompt,
    build_initial_chat_history,
)


class TrajectoryDataset(Dataset):
    """Dataset of turn-level prompt-action pairs with target token loss masking."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    messages = record["messages"]
                    sys_msg = messages[0]
                    history = [sys_msg]
                    for i in range(1, len(messages) - 1, 2):
                        user_msg = messages[i]
                        asst_msg = messages[i + 1]
                        prompt_msgs = history + [user_msg]
                        self.examples.append((prompt_msgs, asst_msg))
                        history.extend([user_msg, asst_msg])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt_msgs, asst_msg = self.examples[idx]
        prompt_str = self.tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

        asst_text = f"{asst_msg['content']}<|im_end|>\n"
        asst_ids = self.tokenizer.encode(asst_text, add_special_tokens=False)

        # Protect assistant target tokens from right-truncation loss masking (R7)
        if len(prompt_ids) + len(asst_ids) > self.max_length:
            max_prompt_len = max(0, self.max_length - len(asst_ids))
            if max_prompt_len <= 0:
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[-max_prompt_len:]

        if len(asst_ids) > self.max_length:
            asst_ids = asst_ids[:self.max_length]

        input_ids = prompt_ids + asst_ids
        labels = [-100] * len(prompt_ids) + asst_ids

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        assert any(l != -100 for l in labels), "Dataset item must have at least one unmasked target label!"

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    padded_inputs = []
    padded_labels = []
    attention_masks = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len
        padded_inputs.append(torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_masks.append(torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_masks),
    }


def train_sft_stage(
    model,
    tokenizer,
    dataset_path: str,
    device: str,
    epochs: int = 2,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum_steps: int = 2,
    gc_freq: int = 2,
):
    mem_info = MemoryGuard.get_system_memory_info()
    print(f"\n--- [Stage 1] Supervised Fine-Tuning (SFT) ---")
    print(f"  [RAM Monitor] Total RAM: {mem_info['total_ram_mb']} MB, Available: {mem_info['available_ram_mb']} MB ({mem_info['percent_used']}% used)")
    print(f"  [RAM Adaptation] Configured batch_size={batch_size}, grad_accum_steps={grad_accum_steps}")

    dataset = TrajectoryDataset(dataset_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    optimizer.zero_grad()

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        t0 = time.time()
        for step_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps

            loss.backward()

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            steps += 1

            if (step_idx + 1) % 20 == 0:
                print(f"    [SFT Step {step_idx+1}/{len(dataloader)}] Current Batch Loss: {loss.item() * grad_accum_steps:.4f}", flush=True)

            if step_idx % gc_freq == 0:
                MemoryGuard.cleanup_memory()

        avg_loss = total_loss / max(1, steps)
        elapsed = time.time() - t0
        cur_mem = MemoryGuard.get_system_memory_info()
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s | Process RSS: {cur_mem['process_rss_mb']} MB", flush=True)


def train_grpo_stage(
    model,
    tokenizer,
    device: str,
    num_worlds: int = 15,
    group_size: int = 4,
    lr: float = 5e-5,
    submission_timing_bonus: float = 0.0,
):
    """Run Group-Relative Policy Gradient on active experiment episodes.
    
    Uses group-normalized advantages across G parallel rollouts per world with
    direct advantage-weighted cross-entropy optimization.
    """
    print(f"\n--- [Stage 2] Live Multi-Turn Group-Relative Policy Gradient RL Optimization ---")
    print(f"Note: 8-15 worlds x 4 rollouts is an on-premise developer smoke test. (submission_timing_bonus={submission_timing_bonus})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    env = CircuitSpectroscopyEnv(submission_timing_bonus=submission_timing_bonus)

    total_rollouts = 0
    group_rewards_history = []

    # GRPO uses training seeds 30000+ (completely isolated from dev 10000+ and locked test 20000+)
    for w_idx in range(num_worlds):
        world_seed = 30000 + w_idx
        group_trajectories = []
        group_rewards = []

        # Roll out G trajectories for this world
        for g in range(group_size):
            obs, _ = env.reset(seed=world_seed)
            # Reseed sensor noise for diverse exploration
            env.rng_mgr.reseed_sensor(world_seed + 100 * (g + 1))

            chat_messages = build_initial_chat_history()
            step_tokens = []
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # Standardized multi-turn chat prompt byte-identical with evaluation and SFT (R5)
                user_prompt = format_observation_user_prompt(obs)
                chat_messages.append({"role": "user", "content": user_prompt})

                prompt_str = tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                p_enc = tokenizer(prompt_str, return_tensors="pt").to(device)

                # Set eval mode during generation to disable LoRA dropout (R7)
                model.eval()
                with torch.no_grad():
                    gen_ids = model.generate(
                        **p_enc,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = gen_ids[0][p_enc["input_ids"].shape[1]:]
                action_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                chat_messages.append({"role": "assistant", "content": action_text})

                # Parse JSON action directly from generated tokens using shared robust parser
                from resonance.core.schemas import extract_and_parse_action
                parsed_action = extract_and_parse_action(action_text)

                # No silent script fallback! If invalid, pass invalid action payload
                if parsed_action and isinstance(parsed_action, dict) and "type" in parsed_action:
                    action = parsed_action
                else:
                    action = {"type": "invalid", "raw": action_text[:80]}

                obs, reward, terminated, truncated, info = env.step(action)
                step_tokens.append((p_enc["input_ids"], new_tokens))

                # If action was invalid format, terminate early to penalize format failures
                if not info.get("is_valid", True):
                    terminated = True

            group_rewards.append(float(reward))
            group_trajectories.append(step_tokens)

        # GRPO group-normalized advantages
        advantages, is_zero_var = compute_grpo_advantages(group_rewards)
        mean_r = float(np.mean(group_rewards))
        group_rewards_history.append(mean_r)

        if not is_zero_var:
            # Policy gradient update on advantage across ALL action decisions in the trajectory
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            n_updates = 0

            for g in range(group_size):
                adv = float(advantages[g])
                traj = group_trajectories[g]
                for p_ids, act_ids in traj:
                    if len(act_ids) == 0:
                        continue
                    input_seq = torch.cat([p_ids[0].to(device), act_ids.to(device)]).unsqueeze(0)
                    labels = torch.cat([
                        torch.full((len(p_ids[0]),), -100, dtype=torch.long, device=device),
                        act_ids.to(device),
                    ]).unsqueeze(0)

                    out = model(input_ids=input_seq, labels=labels)
                    # out.loss is cross-entropy: -log p(a)
                    # Policy gradient minimizes: -adv * log p(a) = adv * out.loss (F1 fix)
                    loss = adv * out.loss
                    total_loss += loss
                    n_updates += 1

            if n_updates > 0:
                (total_loss / n_updates).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_rollouts += group_size
        MemoryGuard.cleanup_memory()
        cur_mem = MemoryGuard.get_system_memory_info()
        print(f"  World {w_idx+1:2d}/{num_worlds} | Mean Group Reward: {mean_r:.4f} | Process RSS: {cur_mem['process_rss_mb']} MB | Rewards: {[round(r, 3) for r in group_rewards]}", flush=True)

    print(f"  GRPO Finished. Average Reward over training: {np.mean(group_rewards_history):.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train Resonance Policy with SFT + Group-Relative Policy Gradient (GRPO Baseline)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="checkpoints/resonance_qwen_rl")
    parser.add_argument("--sft_output_dir", type=str, default="checkpoints/resonance_qwen_sft")
    parser.add_argument("--sft_samples", type=int, default=30)
    parser.add_argument("--grpo_worlds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")  # CPU avoids MPS backward sync stalls
    parser.add_argument(
        "--submission_timing_bonus",
        type=float,
        default=0.0,
        help="Explicit reward bonus for timely submission before turn exhaustion (default: 0.0)"
    )
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    print(f"=== Starting Resonance RL Policy Training on {device} (Seed={args.seed}) ===")

    guard = MemoryGuard()
    safe_cfg = guard.get_safe_training_config()
    print(f"[Memory Guard] Operating Mode: {safe_cfg['mode']}")
    print(f"[Memory Guard] Available RAM: {safe_cfg['available_ram_mb']} MB -> Auto-adapted: batch_size={safe_cfg['batch_size']}, grad_accum={safe_cfg['grad_accum_steps']}")

    # 1. Generate / load SFT gold trajectory dataset
    sft_data_path = "artifacts/sft_trajectories.jsonl"
    print(f"[1/4] Preparing {args.sft_samples} verified expert trajectories (seeds 50000+)...")
    from resonance.training.sft import generate_sft_dataset
    generate_sft_dataset(
        num_trajectories=args.sft_samples,
        output_jsonl=sft_data_path,
        base_seed=50000,
    )

    # 2. Load model and tokenizer
    print(f"[2/4] Loading {args.base_model} and initializing LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        device_map=device if device != "mps" else None,
        trust_remote_code=True,
    )
    if device == "mps":
        model = model.to("mps")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # 3. Train SFT Stage
    train_sft_stage(
        model,
        tokenizer,
        sft_data_path,
        device,
        epochs=args.epochs,
        lr=2e-4,
        batch_size=safe_cfg["batch_size"],
        grad_accum_steps=safe_cfg["grad_accum_steps"],
        gc_freq=safe_cfg["gc_frequency"],
    )

    # Save SFT-only checkpoint (R3 ablation checkpoint)
    print(f"\n[3/4] Saving SFT-only ablation checkpoint to {args.sft_output_dir}...")
    os.makedirs(args.sft_output_dir, exist_ok=True)
    model.save_pretrained(args.sft_output_dir)
    tokenizer.save_pretrained(args.sft_output_dir)
    
    sft_cfg = {
        "stage": "sft_only",
        "base_model": args.base_model,
        "sft_samples": args.sft_samples,
        "epochs": args.epochs,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(args.sft_output_dir, "training_config.json"), "w") as f:
        json.dump(sft_cfg, f, indent=2)
    print(f"Successfully saved SFT-only checkpoint to {args.sft_output_dir}!")

    # 4. Train GRPO RL Stage
    train_grpo_stage(
        model,
        tokenizer,
        device,
        num_worlds=args.grpo_worlds,
        group_size=4,
        lr=5e-5,
        submission_timing_bonus=args.submission_timing_bonus,
    )

    # 5. Save final RL model
    print(f"\n[4/4] Saving final RL model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    rl_cfg = {
        "stage": "sft_plus_group_relative_rl",
        "base_model": args.base_model,
        "sft_samples": args.sft_samples,
        "grpo_worlds": args.grpo_worlds,
        "submission_timing_bonus": args.submission_timing_bonus,
        "epochs": args.epochs,
        "seed": args.seed,
        "optimization": "Group-Relative Policy Gradient (advantage-weighted cross-entropy with group baseline normalization)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(rl_cfg, f, indent=2)
    print(f"Successfully saved RL checkpoint to {args.output_dir}!")


if __name__ == "__main__":
    main()
