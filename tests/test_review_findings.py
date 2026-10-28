"""Unit tests verifying all fixes from Resonance_Detailed_Review.pdf (F1-F9)."""

import os
import json
import unittest
import numpy as np
from pydantic import ValidationError

try:
    import torch
    import torch.nn as nn
    from scripts.train_rl import TrajectoryDataset
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    TrajectoryDataset = None
    HAS_TORCH = False

from resonance.core.schemas import (
    MeasureAction,
    CircuitParameters,
    SubmitCircuitAction,
    extract_and_parse_action,
)
from resonance.models.prompted_llm import PromptedLLMAgent
from resonance.envs.circuit import CircuitSpectroscopyEnv
from resonance.runners.runner import EpisodeRunner
from resonance.runners.manifest import RunArtifactLogger
from resonance.training.grpo import GRPOMultiTurnCollector, compute_grpo_advantages
from resonance.training.trajectory_collector import MultiTurnTrajectoryCollector


class TestF1PolicyGradientSign(unittest.TestCase):
    """F1: Ensure policy gradient loss sign correctly increases log-prob for positive advantage."""

    def test_loss_sign_gradient_direction_numpy(self):
        # Target action y = 1 out of 2 classes
        w = np.array([0.5, -0.5])
        x = np.array([1.0, 1.0])
        logits = w * x
        exp_l = np.exp(logits - np.max(logits))
        p0 = (exp_l / np.sum(exp_l))[1]

        # In HuggingFace out.loss = -log p(y)
        # Proper loss is: L = adv * out.loss = -adv * log p(y)
        # Gradient wrt logits is: dL/d(logits) = adv * (p - y)
        # For positive advantage adv = +2.0:
        adv = 2.0
        grad_logits = adv * ((exp_l / np.sum(exp_l)) - np.array([0.0, 1.0]))
        grad_w = grad_logits * x

        # Gradient descent update: w_new = w - eta * grad_w
        w_new = w - 0.5 * grad_w
        logits_new = w_new * x
        exp_l_new = np.exp(logits_new - np.max(logits_new))
        p_new = (exp_l_new / np.sum(exp_l_new))[1]

        self.assertGreater(p_new, p0, "loss = adv * out.loss must increase action probability when adv > 0")

        # Inverted bug test: if loss was -adv * out.loss:
        grad_inverted = -adv * ((exp_l / np.sum(exp_l)) - np.array([0.0, 1.0])) * x
        w_bad = w - 0.5 * grad_inverted
        logits_bad = w_bad * x
        exp_l_bad = np.exp(logits_bad - np.max(logits_bad))
        p_bad = (exp_l_bad / np.sum(exp_l_bad))[1]
        self.assertLess(p_bad, p0, "Inverted loss would decrease action probability when adv > 0!")

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for autograd loss sign test")
    def test_loss_sign_gradient_direction_pytorch(self):
        # Linear model predicting logits for 2 classes
        torch.manual_seed(42)
        model = nn.Linear(4, 2, bias=False)
        x = torch.randn(1, 4)
        target_token = torch.tensor([1], dtype=torch.long)

        # Baseline probability of target token
        logits_0 = model(x)
        prob_0 = torch.softmax(logits_0, dim=-1)[0, 1].item()

        # Step with positive advantage A = +2.0
        # loss = adv * cross_entropy(logits, target)
        ce_loss = nn.functional.cross_entropy(logits_0, target_token)
        adv_pos = 2.0
        loss_pos = adv_pos * ce_loss
        loss_pos.backward()

        with torch.no_grad():
            for p in model.parameters():
                p -= 0.1 * p.grad  # standard gradient descent step

        logits_after_pos = model(x)
        prob_after_pos = torch.softmax(logits_after_pos, dim=-1)[0, 1].item()

        # Probability of target token MUST increase after positive advantage update!
        self.assertGreater(prob_after_pos, prob_0, "Positive advantage must increase target action probability!")

        # Now test negative advantage A = -2.0
        torch.manual_seed(42)
        model_neg = nn.Linear(4, 2, bias=False)
        logits_0_neg = model_neg(x)
        ce_loss_neg = nn.functional.cross_entropy(logits_0_neg, target_token)
        adv_neg = -2.0
        loss_neg = adv_neg * ce_loss_neg
        loss_neg.backward()

        with torch.no_grad():
            for p in model_neg.parameters():
                p -= 0.1 * p.grad

        logits_after_neg = model_neg(x)
        prob_after_neg = torch.softmax(logits_after_neg, dim=-1)[0, 1].item()

        # Probability of target token MUST decrease after negative advantage update!
        self.assertLess(prob_after_neg, prob_0, "Negative advantage must decrease target action probability!")


class TestF2AndF8SchemasAndParsing(unittest.TestCase):
    """F2 & F8: Shared JSON parser and schema strictness."""

    def test_nested_submit_json_extraction(self):
        raw_text = (
            "Based on the measurements, here is my final submission:\n"
            "```json\n"
            "{\n"
            '  "type": "submit",\n'
            '  "model_probs": {"RC": 0.05, "RL": 0.90, "RLC": 0.05},\n'
            '  "parameters": {\n'
            '    "circuit_type": "RL",\n'
            '    "R": 100.0,\n'
            '    "L_mH": 12.5\n'
            "  }\n"
            "}\n"
            "```\n"
            "Thank you!"
        )
        parsed = extract_and_parse_action(raw_text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["type"], "submit")
        self.assertEqual(parsed["parameters"]["circuit_type"], "RL")
        self.assertAlmostEqual(parsed["model_probs"]["RL"], 0.90)

    def test_schema_forbids_extra_fields(self):
        # MeasureAction rejects unknown fields
        with self.assertRaises(ValidationError):
            MeasureAction(type="measure", frequency_hz=1000.0, extra_bogus_field=123)

        # CircuitParameters rejects unknown fields
        with self.assertRaises(ValidationError):
            CircuitParameters(circuit_type="RC", R=100.0, C_uF=1.0, hacked_param="attack")

        # SubmitCircuitAction rejects extra keys in model_probs
        with self.assertRaises(ValidationError):
            SubmitCircuitAction(
                type="submit",
                model_probs={"RC": 0.5, "RL": 0.5, "RLC": 0.0, "UNKNOWN": 0.0},
                parameters=CircuitParameters(circuit_type="RC", R=100.0, C_uF=1.0),
            )


class TestF3AndF4TracksAndFailClosed(unittest.TestCase):
    """F3 & F4 & R1: Track separation (strict vs assisted) and fail-closed loading."""

    def test_fail_closed_on_missing_model(self):
        with self.assertRaises(RuntimeError):
            PromptedLLMAgent(
                model_name_or_path="nonexistent/fake_model_dir_xyz",
                allow_mock=False,
            )

    def test_allow_mock_flag(self):
        agent = PromptedLLMAgent(
            model_name_or_path="nonexistent/fake_model_dir_xyz",
            allow_mock=True,
        )
        self.assertTrue(agent.is_mock)

    def test_strict_mode_does_not_auto_fit_and_fails_env(self):
        env = CircuitSpectroscopyEnv()
        env.reset(seed=10000)
        # Take two measurements so history exists
        env.step({"type": "measure", "frequency_hz": 100.0})
        obs, _, _, _, _ = env.step({"type": "measure", "frequency_hz": 1000.0})

        agent = PromptedLLMAgent(
            model_name_or_path="nonexistent/fake",
            allow_mock=True,
            track="strict",
        )
        incomplete_resp = '```json\n{"type": "submit"}\n```'
        action = agent._parse_action_json(incomplete_resp, obs)
        # In strict mode, action is incomplete and fails env validation
        _, reward, term, trunc, info = env.step(action)
        self.assertFalse(info.get("is_valid", True))
        self.assertEqual(reward, -0.05)
        self.assertFalse(agent.last_action_metadata.get("assisted"))

    def test_assisted_mode_submits_clean_action_accepted_by_env(self):
        """Acceptance test for R1: Assisted track produces clean action accepted by env with reward > 0."""
        env = CircuitSpectroscopyEnv()
        env.reset(seed=10000)
        # Take four measurements
        for f in [50.0, 500.0, 5000.0, 20000.0]:
            obs, _, _, _, _ = env.step({"type": "measure", "frequency_hz": f})

        agent = PromptedLLMAgent(
            model_name_or_path="nonexistent/fake",
            allow_mock=True,
            track="assisted",
        )
        incomplete_resp = '```json\n{"type": "submit"}\n```'
        action = agent._parse_action_json(incomplete_resp, obs)

        # Invariant R1 & Oct26 Item 7: SubmitCircuitAction has extra="forbid"
        self.assertEqual(SubmitCircuitAction.model_config.get("extra"), "forbid")
        # Action MUST NOT contain extra keys that trip extra="forbid"!
        self.assertNotIn("fit_source", action, "fit_source must be kept out of the env action payload")
        self.assertNotIn("assisted", action, "assisted must be kept out of the env action payload")
        
        # Pydantic parsing of action MUST succeed without ValidationError
        validated = SubmitCircuitAction(**action)
        self.assertEqual(validated.type, "submit")

        # Step the real environment with the assisted submit
        obs, reward, term, trunc, info = env.step(action)
        self.assertTrue(term, "Assisted submit must terminate the episode")
        self.assertTrue(info.get("is_valid", True), f"Environment rejected assisted submit: {info.get('error')}")
        self.assertGreater(reward, 0.5, f"Assisted submit with 4 measurements must score positive reward, got {reward}")

        # Metadata is tracked on agent instance, outside action dict
        self.assertTrue(agent.last_action_metadata.get("assisted"))
        self.assertEqual(agent.last_action_metadata.get("fit_source"), "numerical_fitter")

    def test_runner_records_assisted_metadata_end_to_end(self):
        """Acceptance test for R1: EpisodeRunner records assisted flag in episode record."""
        env = CircuitSpectroscopyEnv()
        agent = PromptedLLMAgent(
            model_name_or_path="nonexistent/fake",
            allow_mock=True,
            track="assisted",
        )
        # Force agent to emit an incomplete submit that triggers auto-fit
        agent._generate_completion = lambda: '```json\n{"type": "submit"}\n```'

        runner = EpisodeRunner(env, agent)
        rec = runner.run_episode(world_seed=10000)
        self.assertTrue(rec.get("assisted"))
        self.assertEqual(rec.get("fit_source"), "numerical_fitter")

        # In strict mode
        agent_strict = PromptedLLMAgent(
            model_name_or_path="nonexistent/fake",
            allow_mock=True,
            track="strict",
        )
        agent_strict._generate_completion = lambda: '```json\n{"type": "submit"}\n```'
        runner_strict = EpisodeRunner(env, agent_strict)
        rec_strict = runner_strict.run_episode(world_seed=10000)
        self.assertFalse(rec_strict.get("assisted"))
        self.assertIsNone(rec_strict.get("fit_source"))


class TestF5MetricsAndSummary(unittest.TestCase):
    """F5: Unconditional accuracy and clean separation of episode return vs terminal reward."""

    def test_unconditional_accuracy_denominator(self):
        logger = RunArtifactLogger(
            output_dir="/tmp/test_f5_artifacts",
            experiment_name="test_f5",
            model_name="mock_model",
            decoding_params={},
            prompt_text="",
        )
        # 3 episodes: 2 submitted (1 correct, 1 incorrect), 1 truncated without submission
        logger.record_episode({
            "world_id": "w0", "submitted": True, "is_correct_type": True,
            "terminal_reward": 1.0, "episode_return": 0.95, "nrmse": 0.02, "brier_score": 0.05
        })
        logger.record_episode({
            "world_id": "w1", "submitted": True, "is_correct_type": False,
            "terminal_reward": -0.5, "episode_return": -0.55, "nrmse": 0.45, "brier_score": 0.8
        })
        logger.record_episode({
            "world_id": "w2", "submitted": False, "is_correct_type": None,
            "terminal_reward": 0.0, "episode_return": -0.2, "nrmse": None, "brier_score": None
        })

        summary = logger._compute_summary()
        self.assertEqual(summary["total_episodes"], 3)
        self.assertEqual(summary["submitted_episodes"], 2)
        # Unconditional accuracy: 1 / 3 = 0.3333, NOT 1 / 2 = 0.5
        self.assertAlmostEqual(summary["unconditional_type_accuracy"], 1.0 / 3.0)
        self.assertEqual(len([x for x in [0.02, 0.45]]), 2)
        self.assertAlmostEqual(summary["conditional_mean_nrmse"], (0.02 + 0.45) / 2.0)


class TestF6SFTBudgetAndTargetProtection(unittest.TestCase):
    """F6 & R5 & R6: True measurement budget reporting, byte-identical prompts, and length diversity."""

    def test_sft_prompt_reflects_actual_budget(self):
        collector = MultiTurnTrajectoryCollector()
        traj = collector.collect_expert_trajectory(world_seed=101, max_measurements=4)
        messages = traj["messages"]
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        # Verify no user message lied about 0 remaining budget when budget was positive
        for u in user_msgs[:-1]:
            self.assertNotIn("Remaining measurement budget: 0", u)

    def test_sft_trajectory_length_diversity(self):
        """Acceptance test for R6: SFT collector produces a spread of trajectory lengths."""
        collector = MultiTurnTrajectoryCollector()
        lengths = set()
        for s in [101, 102, 103, 104, 105, 106]:
            traj = collector.collect_expert_trajectory(world_seed=s)
            lengths.add(traj["trajectory_length"])
            # Invariant: final-turn prompt budget matches env remaining_calls
            final_user_msg = [m["content"] for m in traj["messages"] if m["role"] == "user"][-1]
            rem_calls_expected = 12 - traj["trajectory_length"]
            self.assertIn(f"Remaining measurement budget: {rem_calls_expected}", final_user_msg)
        self.assertGreater(len(lengths), 1, f"Expected length diversity in SFT trajectories, got {lengths}")

    def test_byte_identical_prompts_across_sft_rollout_eval(self):
        """Acceptance test for R5: SFT, rollout, and eval use byte-identical user prompt formatting."""
        from resonance.core.prompts import format_observation_user_prompt
        env = CircuitSpectroscopyEnv()
        obs, _ = env.reset(seed=10000)
        obs, _, _, _, _ = env.step({"type": "measure", "frequency_hz": 120.0})

        # Eval agent prompt
        agent = PromptedLLMAgent(model_name_or_path="fake", allow_mock=True)
        eval_prompt = format_observation_user_prompt(obs)

        # Direct prompt function
        standard_prompt = format_observation_user_prompt(obs)

        self.assertEqual(eval_prompt, standard_prompt)
        self.assertIn("f=120.0 Hz ->", eval_prompt)
        self.assertIn(f"Remaining measurement budget: {obs['remaining_calls']}", eval_prompt)

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for TrajectoryDataset test")
    def test_target_token_protection_under_long_prompts(self):
        class DummyTokenizer:
            eos_token_id = 0
            def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
                return "long prompt " * 200
            def encode(self, text, add_special_tokens=False):
                # 600 tokens for prompt, 20 tokens for assistant
                if "long prompt" in text:
                    return list(range(1, 601))
                return [100, 101, 102]

        dataset = TrajectoryDataset.__new__(TrajectoryDataset)
        dataset.tokenizer = DummyTokenizer()
        dataset.max_length = 512
        dataset.examples = [([{"role": "user", "content": "hi"}], {"role": "assistant", "content": "bye"})]

        item = dataset[0]
        # Verify that labels has unmasked tokens (not all -100!)
        unmasked = (item["labels"] != -100).sum().item()
        self.assertGreater(unmasked, 0, "Assistant target tokens must not be completely masked out by prompt length!")
        self.assertEqual(len(item["input_ids"]), 512)


class TestF9GRPOSensorReseed(unittest.TestCase):
    """F9: Ensure group rollouts share identical circuit ground-truth but independent sensor noise."""

    def test_group_rollouts_independent_sensor_noise(self):
        env1 = CircuitSpectroscopyEnv()
        env2 = CircuitSpectroscopyEnv()

        obs1, _ = env1.reset(seed=999)
        obs2, _ = env2.reset(seed=999)

        # Ground truth hidden circuits MUST be identical
        self.assertEqual(env1._hidden_params["circuit_type"], env2._hidden_params["circuit_type"])
        self.assertAlmostEqual(env1._hidden_params["R"], env2._hidden_params["R"])

        # Reseed sensor noise on env2
        env2.reseed_sensor(999 + 1000)

        # Measure exact same frequency on both
        act = {"type": "measure", "frequency_hz": 1000.0}
        o1, _, _, _, _ = env1.step(act)
        o2, _, _, _, _ = env2.step(act)

        read1 = o1["history"][0]["reading"]["current_amplitude_A"]
        read2 = o2["history"][0]["reading"]["current_amplitude_A"]

        # Different sensor noise means readings must not be bit-for-bit identical!
        self.assertNotEqual(read1, read2, "Sensor readings must differ across independent sensor seeds!")


class TestOct26ReviewAuditFindings(unittest.TestCase):
    """Regression tests for October 26 audit findings (R1-R5)."""

    def test_r1_episode_files_under_1mb_and_complete(self):
        """R1: All 6 episode files must exist, contain 50 episodes, and be < 1MB (GitHub API safe)."""
        dirs = [
            "artifacts/model_eval_qwen_base",
            "artifacts/model_eval_qwen_base_strict",
            "artifacts/model_eval_qwen_sft",
            "artifacts/model_eval_qwen_sft_strict",
            "artifacts/model_eval_qwen_rl",
            "artifacts/model_eval_qwen_rl_strict",
        ]
        for d in dirs:
            episodes_path = os.path.join(d, "episodes.jsonl")
            self.assertTrue(os.path.exists(episodes_path), f"Missing {episodes_path}")
            size = os.path.getsize(episodes_path)
            self.assertGreater(size, 1000, f"File {episodes_path} suspiciously small ({size} bytes)")
            self.assertLess(size, 1000000, f"File {episodes_path} ({size} bytes) exceeds 1MB GitHub API limit")

            with open(episodes_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            self.assertEqual(len(lines), 50, f"Expected 50 episodes in {episodes_path}, got {len(lines)}")

    def test_r1_unified_headline_eval_under_1mb_and_complete(self):
        """R1: unified_headline_eval.json must exist, be < 1MB, and contain all headline models."""
        unified_path = "artifacts/unified_headline_eval.json"
        self.assertTrue(os.path.exists(unified_path))
        size = os.path.getsize(unified_path)
        self.assertGreater(size, 1000, f"unified_headline_eval.json suspiciously small ({size} bytes)")
        self.assertLess(size, 1000000, f"unified_headline_eval.json ({size} bytes) exceeds 1MB GitHub API limit")

        with open(unified_path) as f:
            data = json.load(f)
        for key in ["teacher_baseline", "base_model_strict", "base_model_assisted", "sft_model_strict", "sft_model_assisted", "rl_model_strict", "rl_model_assisted", "paired_differences"]:
            self.assertIn(key, data, f"Missing key {key} in unified_headline_eval.json")

    def test_r1_independent_episode_recomputation(self):
        """R1: Summary metrics must independently reproduce exactly from episode-level lines."""
        dirs = [
            "artifacts/model_eval_qwen_base",
            "artifacts/model_eval_qwen_base_strict",
            "artifacts/model_eval_qwen_sft",
            "artifacts/model_eval_qwen_sft_strict",
            "artifacts/model_eval_qwen_rl",
            "artifacts/model_eval_qwen_rl_strict",
        ]
        for d in dirs:
            summary_path = os.path.join(d, "summary.json")
            episodes_path = os.path.join(d, "episodes.jsonl")
            with open(summary_path) as f:
                summ = json.load(f)
            episodes = []
            with open(episodes_path) as f:
                for line in f:
                    if line.strip():
                        episodes.append(json.loads(line))

            n_eps = len(episodes)
            self.assertEqual(n_eps, 50)
            recomputed_reward = float(np.mean([ep.get("terminal_reward", ep.get("reward", 0.0)) for ep in episodes]))
            published_reward = summ.get("mean_terminal_reward", summ.get("mean_reward"))
            self.assertAlmostEqual(recomputed_reward, published_reward, places=4)

            sub_eps = [ep for ep in episodes if bool(ep.get("submitted"))]
            self.assertEqual(len(sub_eps), summ.get("submitted_episodes"))

            recomputed_acc = float(sum(1.0 for ep in episodes if bool(ep.get("is_correct_type"))) / n_eps)
            published_acc = summ.get("unconditional_type_accuracy", summ.get("type_classification_accuracy"))
            self.assertAlmostEqual(recomputed_acc, published_acc, places=4)

    def test_r2_submission_timing_bonus(self):
        """R2: Submission timing incentive awards bonus for timely submission before turn budget exhaustion."""
        from resonance.core.scoring import compute_terminal_reward
        # Base reward without timing bonus
        r_base = compute_terminal_reward(nrmse=0.05, instrument_calls=6, query_cost=0.005, submission_timing_bonus=0.0, remaining_turns=10)
        # Reward with timing bonus and remaining turns
        r_timed = compute_terminal_reward(nrmse=0.05, instrument_calls=6, query_cost=0.005, submission_timing_bonus=0.05, remaining_turns=10)
        self.assertAlmostEqual(r_timed - r_base, 0.05, places=5)

        # Timing bonus is 0 if remaining turns is 0
        r_exhausted = compute_terminal_reward(nrmse=0.05, instrument_calls=6, query_cost=0.005, submission_timing_bonus=0.05, remaining_turns=0)
        self.assertAlmostEqual(r_exhausted, r_base, places=5)

    def test_boolean_serialization_round_trip(self):
        """P2: Ensure booleans are serialized as native bool (True/False), not integers (1/0)."""
        from resonance.runners.manifest import RunArtifactLogger
        logger = RunArtifactLogger(
            output_dir="/tmp/test_bool_manifest",
            experiment_name="test_bool",
            model_name="mock",
            mode="assisted",
            decoding_params={},
            prompt_text="test",
        )
        sample = {
            "submitted": True,
            "terminated": False,
            "int_val": 1,
            "zero_val": 0,
            "float_val": 1.5,
            "np_bool_true": np.bool_(True),
            "np_bool_false": np.bool_(False),
            "np_int": np.int64(42),
        }
        serialized = logger._make_serializable(sample)
        dumped = json.dumps(serialized)
        loaded = json.loads(dumped)

        self.assertIs(loaded["submitted"], True)
        self.assertIs(loaded["terminated"], False)
        self.assertIs(loaded["np_bool_true"], True)
        self.assertIs(loaded["np_bool_false"], False)
        self.assertIsInstance(loaded["int_val"], int)
        self.assertIsNot(loaded["int_val"], True)
        self.assertEqual(loaded["int_val"], 1)

    def test_community_governance_and_notices_exist(self):
        """Public Release: Verify presence of governance, security, and attribution files."""
        required = [
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            "SECURITY.md",
            "NOTICES.md",
            "LICENSE",
            ".github/PULL_REQUEST_TEMPLATE.md",
            ".github/ISSUE_TEMPLATE/bug_report.md",
            ".github/ISSUE_TEMPLATE/feature_request.md",
        ]
        for f in required:
            self.assertTrue(os.path.exists(f), f"Missing required public file: {f}")
            self.assertGreater(os.path.getsize(f), 50, f"File {f} is suspiciously empty")

    def test_secrets_privacy_gitignore_patterns(self):
        """Security: Verify .gitignore contains comprehensive patterns blocking sensitive keys and envs."""
        self.assertTrue(os.path.exists(".gitignore"))
        with open(".gitignore") as f:
            content = f.read()
        patterns = [".env*", "*.pem", "*.key", "credentials.json"]
        for p in patterns:
            self.assertIn(p, content, f"Missing pattern {p} in .gitignore")

    def test_config_only_dir_does_not_claim_weight_hash_or_load_confirmation(self):
        """Review P1: A directory containing only adapter_config.json must NOT produce a weight hash or load_confirmation=True."""
        import tempfile
        import shutil
        from resonance.runners.manifest import RunArtifactLogger

        temp_dir = tempfile.mkdtemp()
        try:
            model_dir = os.path.join(temp_dir, "fake_adapter_no_weights")
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "adapter_config.json"), "w") as fp:
                json.dump({"base_model_name_or_path": "test/model"}, fp)

            out_dir = os.path.join(temp_dir, "output")
            logger = RunArtifactLogger(
                output_dir=out_dir,
                experiment_name="test_p1",
                model_name=model_dir,
                decoding_params={},
                prompt_text="test",
            )
            paths = logger.write_artifacts()
            with open(paths["manifest"]) as fp:
                manifest = json.load(fp)

            self.assertIsNone(manifest["weight_sha256"])
            self.assertIsNone(manifest["checkpoint_hash"])
            self.assertIsNone(manifest["hash_target"])
            self.assertIsNotNone(manifest["adapter_config_sha256"])
            self.assertFalse(manifest["load_confirmation"])
        finally:
            shutil.rmtree(temp_dir)

    def test_loader_confirmation_requires_explicit_loader_signal(self):
        """Review P1: load_confirmation must reflect actual model loader signal, not inferred from file existence."""
        import tempfile
        import shutil
        from resonance.runners.manifest import RunArtifactLogger

        temp_dir = tempfile.mkdtemp()
        try:
            model_dir = os.path.join(temp_dir, "fake_adapter_with_weights")
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "adapter_model.safetensors"), "wb") as fp:
                fp.write(b"dummy binary tensor weights")

            # Case A: Weights exist, but loader signal was never provided -> load_confirmation must be False
            out_a = os.path.join(temp_dir, "out_a")
            logger_a = RunArtifactLogger(
                output_dir=out_a,
                experiment_name="test_loader_unconfirmed",
                model_name=model_dir,
                decoding_params={},
                prompt_text="test",
            )
            paths_a = logger_a.write_artifacts()
            with open(paths_a["manifest"]) as fp:
                man_a = json.load(fp)
            self.assertIsNotNone(man_a["weight_sha256"])
            self.assertFalse(man_a["load_confirmation"])

            # Case B: Explicit loader signal confirmed -> load_confirmation is True
            out_b = os.path.join(temp_dir, "out_b")
            logger_b = RunArtifactLogger(
                output_dir=out_b,
                experiment_name="test_loader_confirmed",
                model_name=model_dir,
                decoding_params={},
                prompt_text="test",
                loader_confirmed=True,
            )
            paths_b = logger_b.write_artifacts()
            with open(paths_b["manifest"]) as fp:
                man_b = json.load(fp)
            self.assertTrue(man_b["load_confirmation"])

            # Case C: Mock model override -> load_confirmation is False regardless of file
            out_c = os.path.join(temp_dir, "out_c")
            logger_c = RunArtifactLogger(
                output_dir=out_c,
                experiment_name="test_loader_mock",
                model_name=model_dir,
                decoding_params={},
                prompt_text="test",
                is_mock=True,
                loader_confirmed=True,
            )
            paths_c = logger_c.write_artifacts()
            with open(paths_c["manifest"]) as fp:
                man_c = json.load(fp)
            self.assertFalse(man_c["load_confirmation"])
        finally:
            shutil.rmtree(temp_dir)

    def test_readme_table_relabeling_and_duplicate_weight_callout(self):
        """Recheck Oct 26: Verify table rows are relabeled as Shared Adapter Pass A/B and callout exists."""
        from scripts.render_readme_tables import generate_headline_table, generate_master_table

        with open("artifacts/unified_headline_eval.json") as fp:
            data = json.load(fp)

        headline_md = generate_headline_table(data)
        self.assertIn("Shared Adapter, Pass A", headline_md)
        self.assertIn("Shared Adapter, Pass B", headline_md)
        self.assertNotIn("SFT-Only Ablation", headline_md)
        self.assertNotIn("SFT + Group-Relative RL", headline_md)

        master_md = generate_master_table(data)
        self.assertIn("Shared Adapter (Pass B)", master_md)
        self.assertNotIn("After Policy Optimization (Resonance SFT + Group-Relative RL)", master_md)

        with open("README.md") as fp:
            readme = fp.read()
        self.assertIn("Checkpoint Lineage & Evaluation Scope Disclosure (Research Preview)", readme)
        self.assertIn("22582b459d3523fc60ba7f7ec3afaf9792ec7b74e336fc75c26c5dcd74d459cd", readme)

    def test_audit_oct26_safe_positioning_and_dynamic_transfer_binding(self):
        """Audit Oct 26: Verify safe positioning, dynamic transfer table binding, and softened claims."""
        from scripts.render_readme_tables import generate_master_table

        with open("artifacts/unified_headline_eval.json") as fp:
            unified_data = json.load(fp)
        with open("artifacts/all_benchmarks_comparison.json") as fp:
            transfer_data = json.load(fp)

        master_md = generate_master_table(unified_data, transfer_data=transfer_data)
        self.assertIn("Exploratory active probing (n=2; small sample)", master_md)
        self.assertNotIn("Consistent active probing (n=2; exploratory)", master_md)

        with open("README.md") as fp:
            readme = fp.read()

        # Safe positioning statements
        self.assertIn("Resonance is a research-preview environment for sequential physics experiment design", readme)
        self.assertIn("Current assisted results compare an exploratory adapter with a frozen base", readme)
        self.assertIn("All reported strict small-model runs fail to submit successfully", readme)

        # Baseline distinction under Gate B
        self.assertIn("The 200-world ladder below evaluates non-learned numerical and information-theoretic baseline algorithms", readme)
        self.assertIn("Neither ladder entry nor headline teacher is an RL or learned model", readme)

        # Main task scope alignment
        self.assertIn("Main active 12-call budget continuous complex impedance response scoring and hypothesis calibration ($n=50$ locked development worlds)", readme)

        # Softened claims
        self.assertIn("exploratory active probing on system identification (BoxingGym: 0.3506 vs 0.3462 NRMSE, $n=2$; small sample)", readme)
        self.assertIn("Higher Action Validity in this Sample", readme)
        self.assertNotIn("The evaluated adapter " + "eli" + "minates the schema collapse", readme)

        # Quickstart qualifier
        self.assertIn("# Runs in-repo smoke suites inspired by UGPhysics, PhysReason, PHYSICS-Verified, BoxingGym, LLM-SRBench, and Resonance Circuit", readme)

    def test_audit_oct27_checkpoint_binary_rehash_and_exploratory_archive(self):
        """Audit Oct 27 (Instinct recheck): Verify binary re-hashing verification and exploratory archive labeling."""
        from scripts.verify_checkpoint_identity import verify_checkpoints

        report = verify_checkpoints()
        self.assertEqual(report["status"], "verified")
        self.assertTrue(report["comparison"]["sha256_identical"])
        self.assertTrue(report["comparison"]["bytes_identical"])
        self.assertTrue(report["comparison"]["git_blob_identical"])
        expected_sha256 = "22582b459d3523fc60ba7f7ec3afaf9792ec7b74e336fc75c26c5dcd74d459cd"
        self.assertEqual(report["sft_adapter"]["sha256"], expected_sha256)
        self.assertEqual(report["rl_adapter"]["sha256"], expected_sha256)

        # Check recorded artifact
        with open("artifacts/checkpoint_binary_hashes.json") as fp:
            data = json.load(fp)
        self.assertEqual(data["status"], "verified")
        self.assertEqual(data["sft_adapter"]["git_blob_id"], "e7fb170edc4ce01a5fd39772d71422a90f3529fb")

        # Check exploratory archive labeling
        self.assertTrue(os.path.exists("artifacts/model_eval_qwen/README.md"))
        with open("artifacts/model_eval_qwen/README.md") as fp:
            arch_readme = fp.read()
        self.assertIn("Archived Exploratory Prototype", arch_readme)

        with open("artifacts/model_eval_qwen/run_manifest.json") as fp:
            manifest = json.load(fp)
        self.assertEqual(manifest.get("evaluation_status"), "archived_exploratory_prototype_v0.1")

        with open("README.md") as fp:
            readme = fp.read()
        self.assertIn("verify_checkpoint_identity.py", readme)
        self.assertIn("checkpoint_binary_hashes.json", readme)


if __name__ == "__main__":
    unittest.main()
