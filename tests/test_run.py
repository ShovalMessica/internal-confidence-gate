"""Offline checks for preparation and generation orchestration."""

from contextlib import redirect_stderr, redirect_stdout
from collections import Counter
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from src.activation import ActivationError, ActivationResult
from src.config import TaskConfig
from src.run import _capture_activations, main, prepare_run
from src.generation import (
    ALLOW_ABSTENTION_INSTRUCTION,
    ANSWER_INSTRUCTION,
    GENERATION_PROTOCOL_VERSION,
    GENERATION_RECORD_SCHEMA_VERSION,
    GenerationUnit,
    render_initial_prompt,
)
from src.run_store import build_run_identity


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.dataset = self.root / "dataset.jsonl"
        self.config = self.root / "task.yaml"

    @staticmethod
    def record(example_id, split=None):
        record = {
            "id": example_id,
            "input": f"Classify example {example_id}.",
            "target_answer": "Positive",
        }
        if split:
            record["split"] = split
        return record

    def write_dataset(self, records):
        self.dataset.write_text(
            "\n".join(json.dumps(record) for record in records), encoding="utf-8"
        )

    def write_config(self, path=None, **changes):
        values = {
            "model_name_or_path": "organization/model",
            "dataset_path": str(self.dataset),
            "reasoning_mode": "direct",
        }
        values.update(changes)
        path = path or self.config
        path.write_text(yaml.safe_dump(values), encoding="utf-8")
        return path

    def invoke(self, path=None, *options):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main([str(path or self.config), "--prepare-only", *options])
        return code, stdout.getvalue(), stderr.getvalue()

    def invoke_full(self, path=None, *options):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            patch("src.run._capture_activations") as capture,
            patch("src.run._train_probes") as train_probes,
        ):
            code = main([str(path or self.config), *options])
        self.capture_mock = capture
        self.probe_mock = train_probes
        return code, stdout.getvalue(), stderr.getvalue()

    def run_directories(self):
        output = self.root / "outputs"
        return [path for path in output.iterdir() if (path / "run.json").is_file()]

    @staticmethod
    def generation_record(example, answer="A"):
        return {
            "schema_version": GENERATION_RECORD_SCHEMA_VERSION,
            "protocol_version": GENERATION_PROTOCOL_VERSION,
            "id": example["id"],
            "split": example["split"],
            "status": "success",
            "formatted_prompt_token_ids": [1],
            "formatted_prompt_tokens": 1,
            "reasoning": None,
            "final_control": {
                "token_ids": [2],
                "final_marker_span": [0, 1],
            },
            "answer": {
                "text": answer,
                "token_ids": [3],
                "token_logprobs": [-0.1],
                "tokens": 1,
                "stop_reason": "eos",
            },
        }

    @classmethod
    def probe_ready_generation_records(cls, examples):
        positions = Counter()
        records = []
        for example in examples:
            split = example["split"]
            answer = "Positive" if positions[split] % 2 == 0 else "Negative"
            positions[split] += 1
            records.append(cls.generation_record(example, answer))
        return tuple(records)

    def test_automatic_splits_create_minimal_run_record(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config(reasoning_mode="reasoning")

        code, stdout, stderr = self.invoke()

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        for text in (
            "Configuration valid.", "Model: organization/model",
            "Reasoning mode: reasoning", "Valid examples: 700",
            "Excluded examples: 0", "Split source: automatic",
            "train: 490", "validation: 105", "test: 105",
            "Run record: created",
            "Preparation complete. No model was run.",
        ):
            self.assertIn(text, stdout)

        run_directories = self.run_directories()
        self.assertEqual(len(run_directories), 1)
        record = json.loads((run_directories[0] / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["run_id"], run_directories[0].name)
        self.assertEqual(record["completed_stages"], ["preparation"])
        self.assertEqual(record["preparation"]["split_sizes"], {
            "train": 490, "validation": 105, "test": 105,
        })
        self.assertEqual(set(run_directories[0].iterdir()), {run_directories[0] / "run.json"})

    def test_user_splits_are_preserved(self):
        splits = ["train"] * 500 + ["validation"] * 100 + ["test"] * 100
        self.write_dataset(self.record(index, split) for index, split in enumerate(splits))
        self.write_config()

        prepared = prepare_run(self.config)
        code, stdout, _ = self.invoke()

        self.assertEqual(prepared.split_source, "dataset")
        self.assertEqual([row["split"] for row in prepared.dataset.examples], splits)
        self.assertEqual(code, 0)
        self.assertIn("Split source: provided by dataset", stdout)

    def test_every_exclusion_is_printed(self):
        valid = [self.record(index) for index in range(700)]
        raw = ["not json", json.dumps(self.record(900) | {"target_answer": "UNKNOWN"})]
        raw.extend(json.dumps(record) for record in valid)
        self.dataset.write_text("\n".join(raw), encoding="utf-8")
        self.write_config()

        code, stdout, stderr = self.invoke()

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("Excluded examples: 2", stdout)
        self.assertIn("Line 1, ID unavailable: Invalid record:", stdout)
        self.assertIn(
            "Line 2, ID 900: target_answer cannot be UNKNOWN while abstention is enabled; "
            "UNKNOWN is reserved for model abstention.",
            stdout,
        )

    def test_configuration_error_returns_one_without_traceback(self):
        self.write_dataset([self.record(1)])
        self.write_config(model_name_or_path=None)

        code, stdout, stderr = self.invoke()

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Configuration errors:", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertFalse((self.root / "outputs").exists())

    def test_dataset_error_returns_one_without_traceback(self):
        self.write_dataset(self.record(index) for index in range(10))
        self.write_config()

        code, stdout, stderr = self.invoke()

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("at least 700", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertFalse((self.root / "outputs").exists())

    def test_equivalent_config_and_dataset_paths_reuse_without_rewriting(self):
        self.write_dataset(self.record(index) for index in range(700))
        dataset_copy = self.root / "dataset-copy.jsonl"
        shutil.copyfile(self.dataset, dataset_copy)
        output = self.root / "shared-output"
        self.write_config(output_dir=str(output))

        first_code, first_stdout, _ = self.invoke()
        record_path = next(output.iterdir()) / "run.json"
        fixed_time = 1_600_000_000_000_000_000
        os.utime(record_path, ns=(fixed_time, fixed_time))

        second_config = self.root / "same-settings.yaml"
        self.write_config(
            second_config,
            dataset_path=str(dataset_copy),
            output_dir=str(output),
            reasoning_max_new_tokens=1024,
            answer_max_new_tokens=64,
            allow_abstention=True,
            split_ratios={"train": 0.70, "validation": 0.15, "test": 0.15},
            split_seed=42,
        )
        second_code, second_stdout, _ = self.invoke(second_config)

        self.assertEqual((first_code, second_code), (0, 0))
        self.assertIn("Run record: created", first_stdout)
        self.assertIn("Run record: reused", second_stdout)
        self.assertEqual(len(list(output.iterdir())), 1)
        self.assertEqual(record_path.stat().st_mtime_ns, fixed_time)

    def test_only_effective_settings_and_exact_dataset_bytes_define_identity(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config(reasoning_max_new_tokens=10)
        first = prepare_run(self.config)
        first_id = build_run_identity(
            first.config, first.dataset.content_sha256, first.split_source
        ).run_id

        self.write_config(reasoning_max_new_tokens=999)
        inactive_change = prepare_run(self.config)
        inactive_id = build_run_identity(
            inactive_change.config,
            inactive_change.dataset.content_sha256,
            inactive_change.split_source,
        ).run_id

        self.write_config(reasoning_max_new_tokens=999, probe_seed=7)
        probe_change = prepare_run(self.config)
        probe_seed_id = build_run_identity(
            probe_change.config,
            probe_change.dataset.content_sha256,
            probe_change.split_source,
        ).run_id

        self.write_config(reasoning_max_new_tokens=999, answer_max_new_tokens=65)
        active_change = prepare_run(self.config)
        active_id = build_run_identity(
            active_change.config,
            active_change.dataset.content_sha256,
            active_change.split_source,
        ).run_id

        self.dataset.write_bytes(self.dataset.read_bytes() + b"\n")
        data_change = prepare_run(self.config)
        data_id = build_run_identity(
            data_change.config, data_change.dataset.content_sha256, data_change.split_source
        ).run_id

        self.assertEqual(first_id, inactive_id)
        self.assertEqual(first_id, probe_seed_id)
        self.assertNotEqual(first_id, active_id)
        self.assertNotEqual(active_id, data_id)

    def test_model_loading_settings_define_identity(self):
        self.write_dataset(self.record(index) for index in range(700))
        identities = []
        for changes in (
            {},
            {"model_revision": "abc123"},
            {"device": "cpu"},
            {"dtype": "float32"},
        ):
            self.write_config(**changes)
            prepared = prepare_run(self.config)
            identities.append(build_run_identity(
                prepared.config,
                prepared.dataset.content_sha256,
                prepared.split_source,
            ).run_id)
        self.assertEqual(len(set(identities)), len(identities))

    def test_only_active_generation_settings_define_identity(self):
        self.write_dataset(self.record(index) for index in range(700))
        identities = []
        for changes in (
            {},
            {"generation_seed": 7},
            {"direct_batch_size": 4},
        ):
            self.write_config(**changes)
            prepared = prepare_run(self.config)
            identities.append(build_run_identity(
                prepared.config,
                prepared.dataset.content_sha256,
                prepared.split_source,
            ).run_id)
        self.assertEqual(len(set(identities)), 3)

        self.write_config(reasoning_mode="reasoning", direct_batch_size=2)
        first = prepare_run(self.config)
        self.write_config(reasoning_mode="reasoning", direct_batch_size=99)
        second = prepare_run(self.config)
        self.assertEqual(
            build_run_identity(
                first.config, first.dataset.content_sha256, first.split_source
            ).run_id,
            build_run_identity(
                second.config, second.dataset.content_sha256, second.split_source
            ).run_id,
        )

    def test_runner_does_not_load_model(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config()
        with patch("src.run.load_model") as load_model:
            code, _, _ = self.invoke()
        self.assertEqual(code, 0)
        load_model.assert_not_called()

    def test_full_run_saves_and_reuses_generation_without_reloading_model(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config()
        prepared = prepare_run(self.config)
        records = self.probe_ready_generation_records(prepared.dataset.examples)
        loaded = object()
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            "torch_version": "test",
            "transformers_version": "test",
            "generation_config": {},
        }
        with (
            patch("src.run.load_model", return_value=loaded) as load_model,
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                return_value=iter((GenerationUnit(records),)),
            ),
        ):
            code, stdout, stderr = self.invoke_full()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn(
            "Generation complete: 0 reused, 700 generated; 700 succeeded, 0 failed.",
            stdout,
        )
        self.assertIn("Evaluation artifact: created", stdout)
        self.assertIn("Model Behavior:", stdout)
        self.assertIn("Correct prediction", stdout)
        self.assertIn("Token-limit output", stdout)
        load_model.assert_called_once_with(prepared.config, pinned_revision=None)

        with patch("src.run.load_model") as load_model:
            code, stdout, stderr = self.invoke_full()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Generation artifact: reused (model not loaded).", stdout)
        self.assertIn("Evaluation artifact: reused", stdout)
        load_model.assert_not_called()

        run_directory = self.run_directories()[0]
        record = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(
            record["completed_stages"], ["preparation", "generation", "evaluation"]
        )
        self.assertEqual(record["model"]["resolved_revision"], "resolved-commit")
        self.assertEqual(record["generation"]["successful"], 700)
        self.assertEqual(
            len((run_directory / "generation-manifest.jsonl").read_text().splitlines()),
            700,
        )
        self.assertEqual(
            len(next((run_directory / "evaluations").iterdir()).read_text().splitlines()),
            700,
        )

    def test_probe_ready_run_continues_to_activation_with_loaded_model(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config()
        prepared = prepare_run(self.config)
        records = self.probe_ready_generation_records(prepared.dataset.examples)
        loaded = object()
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            "torch_version": "test",
            "transformers_version": "test",
            "generation_config": {},
        }
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            patch("src.run.load_model", return_value=loaded),
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                return_value=iter((GenerationUnit(records),)),
            ),
            patch("src.run._capture_activations") as capture,
            patch("src.run._train_probes") as train_probes,
        ):
            code = main([str(self.config)])

        self.assertEqual((code, stderr.getvalue()), (0, ""))
        capture.assert_called_once()
        self.assertIs(capture.call_args.args[2], loaded)
        train_probes.assert_called_once()

    def test_completed_activation_capture_is_reused_without_model_loading(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config()
        prepared = prepare_run(self.config)
        records = self.probe_ready_generation_records(prepared.dataset.examples)
        loaded = object()
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            "torch_version": "test",
            "transformers_version": "test",
            "generation_config": {},
        }

        def captured(_loaded, plan):
            tensors = {
                name: torch.zeros(2, len(positions), 3, dtype=torch.float16)
                for name, positions in plan.positions.items()
            }
            return ActivationResult(tensors, ("embedding", "hidden_state_1"), 0.0)

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            patch("src.run.load_model", return_value=loaded) as load_model,
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                return_value=iter((GenerationUnit(records),)),
            ),
            patch("src.run.capture_hidden_states", side_effect=captured) as capture,
        ):
            first_code = main([str(self.config)])

        self.assertEqual((first_code, stderr.getvalue()), (0, ""))
        self.assertEqual(capture.call_count, 700)
        load_model.assert_called_once()
        run_directory = self.run_directories()[0]
        run_record = json.loads(
            (run_directory / "run.json").read_text(encoding="utf-8")
        )
        self.assertIn("activation_capture", run_record["completed_stages"])
        capture_entry = run_record["activation_capture"]
        activation_file = run_directory / capture_entry["artifact"]
        import h5py

        with h5py.File(activation_file, "r") as source:
            self.assertEqual(len(source["examples"]), 700)

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            patch("src.run.load_model") as load_model,
            patch("src.run.capture_hidden_states") as capture,
        ):
            second_code = main([str(self.config)])

        self.assertEqual((second_code, stderr.getvalue()), (0, ""))
        self.assertIn("Activation artifact: reused (model not loaded).", stdout.getvalue())
        load_model.assert_not_called()
        capture.assert_not_called()

    def test_semantic_spans_are_captured_and_completed_run_skips_loading(self):
        class FastTokenizer:
            is_fast = True

            @staticmethod
            def _ids(text):
                return [ord(character) + 10 for character in text]

            def apply_chat_template(
                self, messages, tokenize, add_generation_prompt, enable_thinking
            ):
                rendered = f"<user>{messages[0]['content']}</user><assistant>"
                return self._ids(rendered) if tokenize else rendered

            def __call__(self, text, add_special_tokens, return_offsets_mapping):
                return {
                    "input_ids": self._ids(text),
                    "offset_mapping": [
                        (index, index + 1) for index in range(len(text))
                    ],
                }

        records = []
        dataset = []
        tokenizer = FastTokenizer()
        instruction = f"{ANSWER_INSTRUCTION}\n{ALLOW_ABSTENTION_INSTRUCTION}"
        for index in range(700):
            record = self.record(index)
            record["semantic_spans"] = {
                "span_1": {
                    "start_char": len("Classify example "),
                    "end_char": len(record["input"]) - 1,
                },
                "span_2": {"start_char": 0, "end_char": len("Classify")},
            }
            dataset.append(record)
        self.write_dataset(dataset)
        self.write_config()
        prepared = prepare_run(self.config)
        for generation in self.probe_ready_generation_records(
            prepared.dataset.examples
        ):
            example = prepared.dataset.examples[generation["id"]]
            generation["formatted_prompt_token_ids"] = render_initial_prompt(
                tokenizer,
                example["input"],
                instruction,
                reasoning=False,
            )
            generation["formatted_prompt_tokens"] = len(
                generation["formatted_prompt_token_ids"]
            )
            generation["final_control"]["answer_instruction"] = instruction
            records.append(generation)

        loaded = SimpleNamespace(tokenizer=tokenizer)
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "FastTokenizer",
            "dtype": "float32",
            "device": "cpu",
            "torch_version": "test",
            "transformers_version": "test",
            "generation_config": {},
        }

        def captured(_loaded, plan):
            tensors = {
                name: torch.zeros(2, len(positions), 3, dtype=torch.float16)
                for name, positions in plan.positions.items()
            }
            return ActivationResult(
                tensors, ("embedding", "hidden_state_1"), 0.0
            )

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            patch("src.run.load_model", return_value=loaded) as load_model,
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                return_value=iter((GenerationUnit(tuple(records)),)),
            ),
            patch("src.run.capture_hidden_states", side_effect=captured) as capture,
        ):
            first_code = main([str(self.config)])

        self.assertEqual((first_code, stderr.getvalue()), (0, ""))
        self.assertEqual(capture.call_count, 700)
        load_model.assert_called_once()
        run_directory = self.run_directories()[0]
        import h5py

        with h5py.File(run_directory / "activations.h5", "r") as source:
            self.assertEqual(source["examples/0/span_1"].shape, (2, 1, 3))
            self.assertEqual(source["examples/699/span_1"].shape, (2, 3, 3))
            self.assertEqual(source["examples/0/span_2"].shape, (2, 8, 3))
        run_record = json.loads(
            (run_directory / "run.json").read_text(encoding="utf-8")
        )
        self.assertIn("span_1", run_record["activation_capture"]["summary"]["positions"])
        self.assertIn("span_2", run_record["activation_capture"]["summary"]["positions"])
        self.assertIn("probe_training", run_record["completed_stages"])
        self.assertEqual(len(run_record["probe_trainings"]), 1)
        self.assertEqual(next(iter(run_record["probe_trainings"].values()))["seed"], 42)
        self.assertTrue((run_directory / "probes.h5").is_file())

        reused_stdout = io.StringIO()
        with (
            redirect_stdout(reused_stdout),
            redirect_stderr(io.StringIO()),
            patch("src.run.load_model") as load_model,
            patch("src.run.load_tokenizer") as load_tokenizer,
            patch("src.run.capture_hidden_states") as capture,
        ):
            second_code = main([str(self.config)])
        self.assertEqual(second_code, 0)
        load_model.assert_not_called()
        load_tokenizer.assert_not_called()
        capture.assert_not_called()
        self.assertIn("Probe artifact: reused.", reused_stdout.getvalue())

    def test_semantic_mapping_failure_precedes_activation_file_writes(self):
        run_directory = self.root / "run"
        run_directory.mkdir()
        config = TaskConfig(
            model_name_or_path="organization/model",
            dataset_path=self.dataset,
            reasoning_mode="direct",
            output_dir=self.root,
        )
        example = {
            "id": 1,
            "input": "target",
            "target_answer": "Positive",
            "split": "train",
            "semantic_spans": {
                "span_1": {"start_char": 0, "end_char": 6},
            },
        }
        prepared = SimpleNamespace(
            config=config,
            dataset=SimpleNamespace(examples=[example]),
            answer_matcher=object(),
        )
        registered = SimpleNamespace(directory=run_directory)
        generation = self.generation_record(example, "Positive")
        generation["final_control"]["answer_instruction"] = "final instruction"
        run_record = {
            "generation": {"artifact_sha256": "g" * 64},
            "model": {
                "identifier": "organization/model",
                "resolved_revision": "commit",
            },
        }
        slow_tokenizer = SimpleNamespace(is_fast=False)

        with (
            patch("src.run.load_run_record", return_value=run_record),
            patch(
                "src.run.build_evaluation_identity",
                return_value=SimpleNamespace(evaluation_id="evaluation"),
            ),
            patch(
                "src.run.load_evaluation_records",
                return_value={1: {"outcome": "correct"}},
            ),
            patch("src.run.validate_completed_activation_capture", return_value=False),
            patch(
                "src.run.load_generation_records",
                return_value=({1: generation}, False),
            ),
            patch("src.run.load_tokenizer", return_value=slow_tokenizer),
            patch("src.run.load_activation_records") as load_records,
        ):
            with self.assertRaisesRegex(ActivationError, "fast tokenizer"):
                _capture_activations(prepared, registered, None, False)

        load_records.assert_not_called()
        self.assertFalse((run_directory / "activations.h5").exists())

    def test_force_recompute_uses_recorded_revision(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config()
        prepared = prepare_run(self.config)
        records = self.probe_ready_generation_records(prepared.dataset.examples)
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            "torch_version": "test",
            "transformers_version": "test",
            "generation_config": {},
        }
        patches = (
            patch("src.run.load_model", return_value=object()),
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                return_value=iter((GenerationUnit(records),)),
            ),
        )
        with patches[0], patches[1], patches[2]:
            self.assertEqual(self.invoke_full()[0], 0)

        with (
            patch("src.run.load_model", return_value=object()) as load_model,
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                return_value=iter((GenerationUnit(records),)),
            ),
        ):
            code, _, stderr = self.invoke_full(None, "--force-recompute")
        self.assertEqual((code, stderr), (0, ""))
        load_model.assert_called_once_with(
            prepared.config, pinned_revision="resolved-commit"
        )

    def test_extended_dataset_reuses_unchanged_generations(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config()
        prepared = prepare_run(self.config)
        records = self.probe_ready_generation_records(prepared.dataset.examples)
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            "torch_version": "test",
            "transformers_version": "test",
            "generation_config": {},
        }
        with (
            patch("src.run.load_model", return_value=object()),
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                return_value=iter((GenerationUnit(records),)),
            ),
        ):
            self.assertEqual(self.invoke_full()[0], 0)

        changed_targets = []
        for index in range(700):
            record = self.record(index)
            record["target_answer"] = "Negative"
            changed_targets.append(record)
        self.write_dataset(changed_targets)
        with (
            patch("src.run.load_model") as load_model,
            patch("src.run.generation_units") as units,
        ):
            code, stdout, stderr = self.invoke_full()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Generation cache: 700 reused.", stdout)
        self.assertIn("model not loaded", stdout)
        load_model.assert_not_called()
        units.assert_not_called()

        self.write_dataset(self.record(index) for index in range(701))
        extended = prepare_run(self.config)
        new_record = self.generation_record(
            extended.dataset.examples[-1], "Positive"
        )
        completed_count = []

        def remaining_units(*_, completed_ids):
            completed_count.append(len(completed_ids))
            return iter((GenerationUnit((new_record,)),))

        with (
            patch("src.run.load_model", return_value=object()) as load_model,
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                side_effect=remaining_units,
            ),
        ):
            code, stdout, stderr = self.invoke_full()

        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Generation cache: 700 reused.", stdout)
        self.assertIn("Generation complete: 700 reused, 1 generated", stdout)
        load_model.assert_called_once_with(
            extended.config, pinned_revision="resolved-commit"
        )
        self.assertEqual(completed_count, [700])

    def test_evaluation_shortages_are_saved_before_runner_fails(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config()
        prepared = prepare_run(self.config)
        records = tuple(
            self.generation_record(example, "Wrong")
            for example in prepared.dataset.examples
        )
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            "torch_version": "test",
            "transformers_version": "test",
            "generation_config": {},
        }
        with (
            patch("src.run.load_model", return_value=object()),
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                return_value=iter((GenerationUnit(records),)),
            ),
        ):
            code, stdout, stderr = self.invoke_full()

        self.assertEqual(code, 1)
        self.assertIn("Evaluation artifact: created", stdout)
        self.assertIn("train needs 100 correct predictions", stderr)
        self.capture_mock.assert_not_called()
        run_directory = self.run_directories()[0]
        run_record = json.loads(
            (run_directory / "run.json").read_text(encoding="utf-8")
        )
        self.assertIn("evaluation", run_record["completed_stages"])
        evaluation = next(iter(run_record["evaluations"].values()))
        self.assertFalse(evaluation["summary"]["probe_ready"])

    def test_matcher_changes_reuse_generation_and_preserve_evaluations(self):
        self.write_dataset(self.record(index) for index in range(700))
        matcher = self.root / "matcher.py"
        source = (
            "def answer_match(prediction, target_answer):\n"
            "    return prediction.casefold() == target_answer.casefold()\n"
        )
        matcher.write_text(source, encoding="utf-8")
        self.write_config(answer_matcher_path=str(matcher))
        prepared = prepare_run(self.config)
        records = self.probe_ready_generation_records(prepared.dataset.examples)
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            "torch_version": "test",
            "transformers_version": "test",
            "generation_config": {},
        }
        with (
            patch("src.run.load_model", return_value=object()),
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                return_value=iter((GenerationUnit(records),)),
            ),
        ):
            self.assertEqual(self.invoke_full()[0], 0)

        copy = self.root / "matcher-copy.py"
        copy.write_text(source, encoding="utf-8")
        self.write_config(answer_matcher_path=str(copy))
        with patch("src.run.load_model") as load_model:
            code, stdout, stderr = self.invoke_full()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Evaluation artifact: reused", stdout)
        load_model.assert_not_called()

        copy.write_text(source + "# revised matcher\n", encoding="utf-8")
        with patch("src.run.load_model") as load_model:
            code, stdout, stderr = self.invoke_full()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Generation artifact: reused", stdout)
        self.assertIn("Evaluation artifact: created", stdout)
        load_model.assert_not_called()

        run_directory = self.run_directories()[0]
        record = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(len(record["evaluations"]), 2)
        self.assertEqual(len(list((run_directory / "evaluations").iterdir())), 2)

    def test_invalid_matcher_stops_before_model_loading(self):
        self.write_dataset(self.record(index) for index in range(700))
        matcher = self.root / "matcher.py"
        matcher.write_text("value = 1\n", encoding="utf-8")
        self.write_config(answer_matcher_path=str(matcher))

        with patch("src.run.load_model") as load_model:
            code, stdout, stderr = self.invoke_full()

        self.assertEqual((code, stdout), (1, ""))
        self.assertIn("Answer matcher must define answer_match", stderr)
        self.assertFalse((self.root / "outputs").exists())
        load_model.assert_not_called()

    def test_legacy_evaluation_is_rebuilt_without_model_loading(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config()
        prepared = prepare_run(self.config)
        records = self.probe_ready_generation_records(prepared.dataset.examples)
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            "torch_version": "test",
            "transformers_version": "test",
            "generation_config": {},
        }
        with (
            patch("src.run.load_model", return_value=object()),
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                return_value=iter((GenerationUnit(records),)),
            ),
        ):
            self.assertEqual(self.invoke_full()[0], 0)

        run_directory = self.run_directories()[0]
        run_path = run_directory / "run.json"
        run_record = json.loads(run_path.read_text(encoding="utf-8"))
        legacy = next(iter(run_record.pop("evaluations").values()))
        source = run_directory / legacy["artifact"]
        legacy_path = run_directory / "evaluations.jsonl"
        source.replace(legacy_path)
        (run_directory / "evaluations").rmdir()
        run_record["evaluation"] = legacy
        run_path.write_text(json.dumps(run_record), encoding="utf-8")

        with patch("src.run.load_model") as load_model:
            code, stdout, stderr = self.invoke_full()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Generation artifact: reused", stdout)
        self.assertIn("Evaluation artifact: created", stdout)
        load_model.assert_not_called()
        rebuilt = json.loads(run_path.read_text(encoding="utf-8"))
        self.assertIn("evaluations", rebuilt)

    def test_split_settings_are_inactive_when_dataset_supplies_splits(self):
        splits = ["train"] * 500 + ["validation"] * 100 + ["test"] * 100
        self.write_dataset(
            self.record(index, split) for index, split in enumerate(splits)
        )
        self.write_config(split_seed=1)
        first = prepare_run(self.config)
        first_id = build_run_identity(
            first.config, first.dataset.content_sha256, first.split_source
        ).run_id

        self.write_config(
            split_seed=999,
            split_ratios={"train": 0.60, "validation": 0.20, "test": 0.20},
        )
        second = prepare_run(self.config)
        second_id = build_run_identity(
            second.config, second.dataset.content_sha256, second.split_source
        ).run_id

        self.assertEqual(first_id, second_id)

    def test_empty_directory_recovers_but_unsafe_run_records_fail_cleanly(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config()
        prepared = prepare_run(self.config)
        identity = build_run_identity(
            prepared.config, prepared.dataset.content_sha256, prepared.split_source
        )
        run_directory = prepared.config.output_dir / identity.run_id
        run_directory.mkdir(parents=True)

        code, stdout, stderr = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("Run record: created", stdout)

        (run_directory / "run.json").write_text("not json", encoding="utf-8")
        code, stdout, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Run storage error:", stderr)
        self.assertNotIn("Traceback", stderr)

        shutil.rmtree(run_directory)
        run_directory.mkdir()
        (run_directory / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        code, stdout, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("nonempty but has no run.json", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_usage_error_uses_argparse_exit_code(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main([])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
