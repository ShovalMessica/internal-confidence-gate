"""Offline tests for loading user-edited task configurations."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import yaml

from src.config import ConfigurationError, load_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.dataset = self.root / "dataset.jsonl"
        # Record validation belongs to the next stage; this loader must not parse it.
        self.dataset.write_text("not a JSON record", encoding="utf-8")
        self.source = self.root / "task.yaml"
        self.required = {
            "model_name_or_path": "organization/model",
            "dataset_path": str(self.dataset),
            "reasoning_mode": "direct",
        }

    def write(self, values):
        self.source.write_text(yaml.safe_dump(values), encoding="utf-8")
        return self.source

    def test_minimal_configuration_defaults_and_no_side_effects(self):
        self.write(self.required)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            config = load_config(self.source)
        self.assertEqual(config.dataset_path, self.dataset)
        self.assertEqual(config.output_dir, self.root / "outputs")
        self.assertIsNone(config.model_revision)
        self.assertIsNone(config.system_prompt_path)
        self.assertIsNone(config.system_prompt)
        self.assertIsNone(config.system_prompt_sha256)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.dtype, "auto")
        self.assertEqual(config.reasoning_max_new_tokens, 1024)
        self.assertEqual(config.answer_max_new_tokens, 64)
        self.assertTrue(config.allow_abstention)
        self.assertIsNone(config.answer_matcher_path)
        self.assertIsNone(config.custom_metrics_path)
        self.assertEqual(config.generation_seed, 42)
        self.assertEqual(config.direct_batch_size, 8)
        self.assertEqual(config.decoding_strategy, "model_default")
        self.assertEqual(config.probe_seed, 42)
        self.assertIsNone(config.probe_positions)
        self.assertIsNone(config.probe_layers)
        self.assertEqual(config.probe_regularization_c, 1.0)
        self.assertEqual(config.probe_class_weight, "balanced")
        self.assertEqual(config.target_tpr, 0.90)
        self.assertEqual(config.split_seed, 42)
        self.assertEqual((config.split_ratios.train, config.split_ratios.validation, config.split_ratios.test), (0.7, 0.15, 0.15))
        self.assertFalse(config.output_dir.exists())
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(self.dataset.read_text(encoding="utf-8"), "not a JSON record")

    def test_explicit_settings_and_local_checkpoint(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        matcher = self.root / "matcher.py"
        matcher.write_text("def answer_match(prediction, target_answer):\n    return True\n")
        metrics = self.root / "metrics.py"
        metrics.write_text("def compute_metrics(records):\n    return {}\n")
        output = self.root / "custom-results"
        values = dict(self.required, model_name_or_path=str(checkpoint),
                      model_revision="checkpoint-v1", output_dir=str(output),
                      reasoning_mode="reasoning", reasoning_max_new_tokens=256,
                      answer_max_new_tokens=16, allow_abstention=False,
                      answer_matcher_path=str(matcher),
                      custom_metrics_path=str(metrics),
                      generation_seed=7, direct_batch_size=4, probe_seed=11,
                      probe_positions=["prompt_end"], probe_layers=[35],
                      probe_regularization_c=0.3, probe_class_weight="none",
                      target_tpr=0.95,
                      split_seed=0,
                      device="cuda:1", dtype="bfloat16",
                      split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1})
        config = load_config(self.write(values))
        self.assertEqual(config.model_name_or_path, str(checkpoint))
        self.assertEqual(config.model_revision, "checkpoint-v1")
        self.assertEqual(config.reasoning_mode, "reasoning")
        self.assertEqual(config.output_dir, output)
        self.assertEqual(config.device, "cuda:1")
        self.assertEqual(config.dtype, "bfloat16")
        self.assertEqual((config.reasoning_max_new_tokens, config.answer_max_new_tokens), (256, 16))
        self.assertFalse(config.allow_abstention)
        self.assertEqual(config.answer_matcher_path, matcher)
        self.assertEqual(config.custom_metrics_path, metrics)
        self.assertEqual((config.generation_seed, config.direct_batch_size), (7, 4))
        self.assertEqual(config.probe_seed, 11)
        self.assertEqual(config.probe_positions, ("prompt_end",))
        self.assertEqual(config.probe_layers, (35,))
        self.assertEqual(config.probe_regularization_c, 0.3)
        self.assertEqual(config.probe_class_weight, "none")
        self.assertEqual(config.target_tpr, 0.95)
        self.assertEqual(config.split_seed, 0)
        self.assertEqual(config.split_ratios.train, 0.8)
        self.assertFalse(output.exists())

    def test_fixed_system_prompt_is_loaded_and_hashed(self):
        prompt = self.root / "system.md"
        prompt_bytes = "Fixed task instructions.\n".encode("utf-8")
        prompt.write_bytes(prompt_bytes)

        config = load_config(
            self.write(dict(self.required, system_prompt_path=str(prompt)))
        )

        self.assertEqual(config.system_prompt_path, prompt)
        self.assertEqual(config.system_prompt, prompt_bytes.decode("utf-8"))
        self.assertEqual(
            config.system_prompt_sha256,
            hashlib.sha256(prompt_bytes).hexdigest(),
        )

    def test_all_missing_required_fields_are_reported_together(self):
        with self.assertRaises(ConfigurationError) as caught:
            load_config(self.write({}))
        self.assertEqual(len(caught.exception.errors), 3)
        for field in self.required:
            self.assertIn(field, str(caught.exception))

    def test_shipped_configuration_has_intentional_unfilled_placeholders(self):
        shipped = Path(__file__).resolve().parents[1] / "configs" / "task.yaml"
        with self.assertRaises(ConfigurationError) as caught:
            load_config(shipped)
        self.assertEqual(len(caught.exception.errors), 3)
        for field in self.required:
            self.assertIn(field, str(caught.exception))

    def test_invalid_scalar_fields(self):
        cases = {
            "reasoning_mode": [None, "thinking", True, []],
            "reasoning_max_new_tokens": [None, 0, -1, True, 1.5, "1024"],
            "answer_max_new_tokens": [None, 0, False, "64"],
            "generation_seed": [None, -1, True, 1.5, "42"],
            "direct_batch_size": [None, 0, -1, True, 1.5, "8"],
            "decoding_strategy": [None, "sample", True],
            "probe_seed": [None, -1, True, 1.5, "42"],
            "probe_positions": [None, [], ["unknown"], ["prompt_end", "prompt_end"]],
            "probe_layers": [None, [], [-1], [True], [35, 35]],
            "probe_regularization_c": [None, 0, -1, True, "0.3", float("nan")],
            "probe_class_weight": [None, "auto", True],
            "target_tpr": [None, 0, -0.1, 1.01, True, "0.9", float("nan"), float("inf")],
            "split_seed": [None, -1, True, 1.5],
            "allow_abstention": [None, 1, "true"],
            "model_revision": [None, "", "  ", 1, False],
            "device": [None, "", "gpu", "cuda:-1", "cuda:one", 0],
            "dtype": [None, "fp16", "int8", 16],
            "model_name_or_path": [None, "", "  ", 123, "./checkpoint", "a/b/c", "a\\b"],
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ConfigurationError) as caught:
                        load_config(self.write(dict(self.required, **{field: value})))
                    self.assertIn(field, str(caught.exception))

    def test_greedy_decoding_is_supported_in_both_modes(self):
        config = load_config(
            self.write(dict(self.required, decoding_strategy="greedy"))
        )
        self.assertEqual(config.decoding_strategy, "greedy")
        reasoning = load_config(
            self.write(
                dict(
                    self.required,
                    reasoning_mode="reasoning",
                    decoding_strategy="greedy",
                )
            )
        )
        self.assertEqual(reasoning.decoding_strategy, "greedy")

    def test_hub_revision_is_optional_and_local_revision_is_required(self):
        config = load_config(self.write(dict(self.required, model_revision="abc123")))
        self.assertEqual(config.model_revision, "abc123")

        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        with self.assertRaisesRegex(ConfigurationError, "model_revision is required"):
            load_config(self.write(dict(
                self.required,
                model_name_or_path=str(checkpoint),
            )))
        local = load_config(self.write(dict(
            self.required,
            model_name_or_path=str(checkpoint),
            model_revision="checkpoint-v1",
        )))
        self.assertEqual(local.model_revision, "checkpoint-v1")

    def test_unknown_fields_and_other_errors_are_combined(self):
        values = dict(self.required, answer_max_new_token=99, temperature=0.8, split_seed=-1)
        with self.assertRaises(ConfigurationError) as caught:
            load_config(self.write(values))
        for field in ("answer_max_new_token", "temperature", "split_seed"):
            self.assertIn(field, str(caught.exception))
        self.assertEqual(len(caught.exception.errors), 3)

    def test_invalid_paths(self):
        checkpoint_file = self.root / "checkpoint.bin"
        checkpoint_file.touch()
        cases = [
            ("dataset_path", None), ("dataset_path", "relative.jsonl"),
            ("dataset_path", str(self.root / "missing.jsonl")),
            ("dataset_path", str(self.root)), ("dataset_path", "bad\0path"),
            ("output_dir", "relative-output"), ("output_dir", None),
            ("output_dir", str(checkpoint_file)),
            ("model_name_or_path", str(self.root / "missing-checkpoint")),
            ("model_name_or_path", str(checkpoint_file)),
            ("answer_matcher_path", None),
            ("answer_matcher_path", "relative.py"),
            ("answer_matcher_path", str(self.root / "missing.py")),
            ("answer_matcher_path", str(self.dataset)),
            ("custom_metrics_path", None),
            ("custom_metrics_path", "relative.py"),
            ("custom_metrics_path", str(self.root / "missing.py")),
            ("custom_metrics_path", str(self.dataset)),
            ("system_prompt_path", None),
            ("system_prompt_path", "relative.md"),
            ("system_prompt_path", str(self.root / "missing-system.md")),
            ("system_prompt_path", str(self.root)),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ConfigurationError) as caught:
                    load_config(self.write(dict(self.required, **{field: value})))
                self.assertIn(field, str(caught.exception))

        empty_prompt = self.root / "empty-system.md"
        empty_prompt.write_text(" \n", encoding="utf-8")
        with self.assertRaisesRegex(ConfigurationError, "empty prompt"):
            load_config(
                self.write(
                    dict(self.required, system_prompt_path=str(empty_prompt))
                )
            )

    def test_invalid_ratios(self):
        cases = [None, [], {}, {"train": 0.7, "validation": 0.3},
                 {"train": 0.7, "validation": 0.15, "test": 0.15, "extra": 0.1},
                 {"train": 0.7, "validation": 0.2, "test": 0.2}]
        for value in (True, "0.7", float("nan"), float("inf"), 0, -0.1, 1):
            cases.append({"train": value, "validation": 0.15, "test": 0.15})
        for ratios in cases:
            with self.subTest(ratios=ratios):
                with self.assertRaises(ConfigurationError) as caught:
                    load_config(self.write(dict(self.required, split_ratios=ratios)))
                self.assertIn("split_ratios", str(caught.exception))

    def test_bad_yaml_shapes_and_unsafe_tags(self):
        for text in ("", "- item\n", "42\n", "input: [\n", "!!python/object:builtins.object {}", "1: value"):
            with self.subTest(text=text):
                self.source.write_text(text, encoding="utf-8")
                with self.assertRaises(ConfigurationError):
                    load_config(self.source)

    def test_duplicate_yaml_fields_are_not_silently_overwritten(self):
        for text in ("split_seed: 42\nsplit_seed: 9\n",
                     "split_ratios:\n  train: 0.7\n  train: 0.8\n"):
            with self.subTest(text=text):
                self.source.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ConfigurationError, "Duplicate YAML field.*line"):
                    load_config(self.source)

    def test_unreadable_configuration(self):
        for path in (self.root / "missing.yaml", self.root):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ConfigurationError, "Cannot read"):
                    load_config(path)

    def test_returned_settings_are_immutable(self):
        config = load_config(self.write(self.required))
        with self.assertRaises(FrozenInstanceError):
            config.answer_max_new_tokens = 3
        with self.assertRaises(FrozenInstanceError):
            config.split_ratios.train = 0.5


if __name__ == "__main__":
    unittest.main()
