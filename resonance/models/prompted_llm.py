"""Prompted Open-Source LLM Agent for Resonance active physics experimentation."""

from __future__ import annotations
import json
import os
import re
from typing import Dict, Any, Optional, List
import numpy as np

from resonance.models.base import BaseAgent
from resonance.core.memory import MemoryGuard
from resonance.core.prompts import (
    SYSTEM_PROMPT_CIRCUIT,
    format_observation_user_prompt,
    build_initial_chat_history,
)


class PromptedLLMAgent(BaseAgent):
    """LLM agent driven by Hugging Face Transformers pipeline or callable model."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.8,
        mode: str = "assisted",  # "strict" or "assisted" (or prompt mode)
        track: Optional[str] = None,
        allow_mock: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Track separation (strict vs assisted)
        if track in ("strict", "assisted"):
            self.track = track
        elif mode in ("strict", "assisted"):
            self.track = mode
        else:
            self.track = "assisted"

        self.prompt_mode = mode if mode in ("thinking", "non-thinking") else "non-thinking"
        self.allow_mock = allow_mock
        self.is_mock = False
        self.loader_confirmed = False

        self.pipeline = None
        self.messages: List[Dict[str, str]] = build_initial_chat_history()
        self.last_action_metadata: Dict[str, Any] = {"assisted": False, "fit_source": None}
        self.last_raw_completion: str = ""
        self.consecutive_invalid: int = 0
        self._init_model()

    def _init_model(self) -> None:
        """Initialize HuggingFace pipeline or Ollama backend lazily."""
        if self.model_name_or_path.startswith("ollama:"):
            self.backend = "ollama"
            self.ollama_model = self.model_name_or_path.split("ollama:", 1)[1]
            self.loader_confirmed = True
            print(f"[PromptedLLMAgent] Initialized Ollama backend for model '{self.ollama_model}'.")
            return

        self.backend = "transformers"
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            print(f"[PromptedLLMAgent] Loading model: {self.model_name_or_path}...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            )

            # Determine device
            if self.device == "auto":
                if torch.backends.mps.is_available():
                    dev = "mps"
                elif torch.cuda.is_available():
                    dev = "cuda"
                else:
                    dev = "cpu"
            else:
                dev = self.device

            # Check if this is a PEFT LoRA adapter directory
            adapter_config_path = os.path.join(self.model_name_or_path, "adapter_config.json")
            if os.path.isdir(self.model_name_or_path) and os.path.exists(adapter_config_path):
                import json
                with open(adapter_config_path) as fp:
                    ad_cfg = json.load(fp)
                base_name = ad_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
                print(f"[PromptedLLMAgent] Detected PEFT adapter. Loading base model: {base_name}...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_name,
                    torch_dtype=torch.float16 if dev in ("mps", "cuda") else torch.float32,
                    trust_remote_code=True,
                )
                from peft import PeftModel
                model = PeftModel.from_pretrained(base_model, self.model_name_or_path)
                model = model.merge_and_unload()
                if dev == "mps":
                    model = model.to("mps")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype=torch.float16 if dev in ("mps", "cuda") else torch.float32,
                    device_map=dev if dev != "mps" else None,
                    trust_remote_code=True,
                )
                if dev == "mps":
                    model = model.to("mps")

            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            self.loader_confirmed = True
            print(f"[PromptedLLMAgent] Successfully loaded {self.model_name_or_path} on {dev}.")
        except Exception as e:
            self.loader_confirmed = False
            if not self.allow_mock:
                raise RuntimeError(
                    f"[PromptedLLMAgent] Failed to load model pipeline for '{self.model_name_or_path}': {e}. "
                    "Fail-closed policy is active. Pass allow_mock=True only in synthetic unit tests."
                ) from e
            print(f"[PromptedLLMAgent] Could not load model pipeline: {e} (Mock allowed)")
            self.pipeline = None
            self.is_mock = True

    def reset(self) -> None:
        self.messages = build_initial_chat_history()
        self.last_action_metadata = {"assisted": False, "fit_source": None}
        self.last_raw_completion = ""
        self.consecutive_invalid = 0

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Direct text generation for physics benchmark questions.
        
        Supports general Q&A, multiple-choice, equation discovery,
        and system identification prompts without forcing circuit schema.
        """
        if getattr(self, "backend", "") == "ollama":
            import urllib.request
            import json
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt})
            req_data = {
                "model": self.ollama_model,
                "messages": msgs,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": max_new_tokens or self.max_new_tokens,
                },
            }
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    return result.get("message", {}).get("content", "").strip()
            except Exception as e:
                print(f"[PromptedLLMAgent] Ollama error: {e}")
                return f"Error: {e}"

        if self.pipeline is None:
            if not self.allow_mock:
                raise RuntimeError("Model pipeline is not loaded and allow_mock is False.")
            # Fallback mock generator
            return "Based on physical principles, the answer is A."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            tokenizer = self.pipeline.tokenizer
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            toks = max_new_tokens or self.max_new_tokens
            out = self.pipeline(
                prompt_str,
                max_new_tokens=toks,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_text = out[0]["generated_text"]
            if prompt_str in generated_text:
                generated_text = generated_text[len(prompt_str):]
            return generated_text.strip()
        except Exception as e:
            print(f"[PromptedLLMAgent] Benchmark generation error: {e}")
            return f"Error: {e}"

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Format prompt, invoke LLM, parse JSON action."""
        user_prompt = format_observation_user_prompt(observation)
        self.messages.append({"role": "user", "content": user_prompt})

        raw_output = self._generate_completion()
        self.last_raw_completion = raw_output
        self.messages.append({"role": "assistant", "content": raw_output})

        action = self._parse_action_json(raw_output, observation)
        return action

    def _generate_completion(self) -> str:
        if getattr(self, "backend", "") == "ollama":
            import urllib.request
            import json
            req_data = {
                "model": self.ollama_model,
                "messages": self.messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_new_tokens,
                },
            }
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    return result.get("message", {}).get("content", "").strip()
            except Exception as e:
                print(f"[PromptedLLMAgent] Ollama error: {e}")
                return '```json\n{"type": "submit"}\n```'

        if self.pipeline is None:
            if not self.allow_mock:
                raise RuntimeError("Model pipeline not loaded and allow_mock is False.")
            # Fallback mock generator
            return '```json\n{"type": "measure", "frequency_hz": 1000.0}\n```'

        try:
            tokenizer = self.pipeline.tokenizer
            prompt_str = tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            out = self.pipeline(
                prompt_str,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_text = out[0]["generated_text"]
            # Extract assistant turn
            if prompt_str in generated_text:
                generated_text = generated_text[len(prompt_str):]
            return generated_text.strip()
        except Exception as e:
            if not self.allow_mock:
                raise RuntimeError(f"[PromptedLLMAgent] Generation error: {e}") from e
            print(f"[PromptedLLMAgent] Generation error: {e}")
            return '```json\n{"type": "measure", "frequency_hz": 1000.0}\n```'

    def _parse_action_json(self, response_text: str, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and sanitize JSON action from LLM output.
        
        Supports two explicit evaluation tracks:
        - 'strict': The model must supply all required fields itself. No auto-fit or forced submission.
                    Incomplete or unparseable output fails closed.
        - 'assisted': Disclosed numerical least-squares fitter assists if submit action lacks parameters
                      or when measurement calls run out. Note: Provenance is stored separately in
                      self.last_action_metadata and NEVER injected into the action dict itself (R1).
        """
        from resonance.core.schemas import extract_and_parse_action
        from resonance.baselines.numerical import evaluate_candidate_models

        rem_calls = obs.get("remaining_calls", 0)
        parsed = extract_and_parse_action(response_text)

        if self.track == "strict":
            self.last_action_metadata = {"assisted": False, "fit_source": None}
            if not isinstance(parsed, dict) or "type" not in parsed:
                return {"type": "invalid", "raw": response_text[:100]}
            # In strict mode, pass parsed directly without auto-fit assistance!
            return parsed

        # assisted track
        is_valid_format = isinstance(parsed, dict) and parsed.get("type") in ("measure", "submit")
        if not is_valid_format:
            self.consecutive_invalid += 1
        else:
            self.consecutive_invalid = 0

        if rem_calls <= 0 or obs.get("remaining_turns", 20) <= 1 or self.consecutive_invalid >= 3:
            if not (isinstance(parsed, dict) and parsed.get("type") == "submit" and "parameters" in parsed and "model_probs" in parsed):
                fit_result = evaluate_candidate_models(obs.get("history", []))
                self.last_action_metadata = {"assisted": True, "fit_source": "numerical_fitter"}
                # Return clean action dictionary adhering strictly to SubmitCircuitAction schema (R1)
                return {
                    "type": "submit",
                    "model_probs": fit_result["model_probs"],
                    "parameters": fit_result["parameters"],
                }

        if isinstance(parsed, dict):
            act_type = parsed.get("type")
            if act_type == "measure":
                self.last_action_metadata = {"assisted": False, "fit_source": None}
                return parsed
            elif act_type == "submit":
                if "parameters" not in parsed or "model_probs" not in parsed:
                    fit_result = evaluate_candidate_models(obs.get("history", []))
                    self.last_action_metadata = {"assisted": True, "fit_source": "numerical_fitter"}
                    return {
                        "type": "submit",
                        "model_probs": fit_result["model_probs"],
                        "parameters": fit_result["parameters"],
                    }
                else:
                    self.last_action_metadata = {"assisted": False, "fit_source": "model"}
                    return parsed

        self.last_action_metadata = {"assisted": False, "fit_source": None}
        return {"type": "invalid", "raw": response_text[:100]}
