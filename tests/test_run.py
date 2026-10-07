"""Offline checks for preparation and generation orchestration."""

from contextlib import redirect_stderr, redirect_stdout
from collections import Counter
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from src.activation import ActivationError, ActivationResult
from src.evaluation import load_answer_matcher
from src.generation_cache import build_generation_context
from src.config import TaskConfig
from src.model import ModelLoadError
from src.run import (
    _Tee, _capture_activations, _format_duration, _print_progress, _start_execution_log,
    _stop_execution_log, main, prepare_run,
)
from src.generation import (
    ALLOW_ABSTENTION_INSTRUCTION,
    ANSWER_INSTRUCTION,
    GENERATION_PROTOCOL_VERSION,
    GENERATION_RECORD_SCHEMA_VERSION,
    GenerationError,
    GenerationUnit,
    render_initial_prompt,
)
from src.generation_cache import generation_runtime_versions
from src.run_store import build_run_identity


class ExecutionLogTests(unittest.TestCase):
    def test_warning_color_reaches_terminal_but_not_execution_log(self):
        terminal, log = io.StringIO(), io.StringIO()
        warning = "Model precision: warning"
        colored = f"\x1b[1m\x1b[31m{warning}\x1b[0m\n"
        stream = _Tee(terminal, log)
        self.assertEqual(stream.write(colored), len(colored))
        self.assertEqual(terminal.getvalue(), colored)
        self.assertEqual(log.getvalue(), warning + "\n")

    def test_duration_uses_largest_two_units(self):
        for seconds, expected in (
            (0, "0s"), (0.25, "250ms"), (1, "1s"), (8, "8s"),
            (1.25, "1s 250ms"), (59.9, "59s 900ms"), (60, "1m"), (61, "1m 1s"),
            (1808, "30m 8s"), (3599, "59m 59s"), (3600, "1h"),
            (16200, "4h 30m"), (16208, "4h 30m"), (86400, "1d"),
            (97200, "1d 3h"),
        ):
            with self.subTest(seconds=seconds):
                self.assertEqual(_format_duration(seconds), expected)

    def test_progress_formats_elapsed_and_eta_in_hours_and_minutes(self):
        terminal, log = io.StringIO(), io.StringIO()
        with redirect_stdout(_Tee(terminal, log)), patch(
            "src.run.time.monotonic", return_value=3600
        ):
            _print_progress("Generation", 2, 11, 0, 0)
            sys.stdout.finish_progress()
        self.assertIn("1h 0.00/s ETA 4h 30m", log.getvalue())

    def test_thousands_of_updates_use_one_terminal_line_and_one_log_entry(self):
        terminal, log = io.StringIO(), io.StringIO()
        stream = _Tee(terminal, log)
        with patch.object(terminal, "isatty", return_value=True):
            for done in range(1, 4801):
                stream.progress(f"Generation: {done}/4800", complete=done == 4800)
        self.assertEqual(terminal.getvalue().count("\n"), 1)
        self.assertEqual(terminal.getvalue().count("\r"), 4800)
        self.assertEqual(log.getvalue(), "Generation: 4800/4800\n")

    def test_progress_clears_shorter_lines_and_fits_terminal_width(self):
        terminal, log = io.StringIO(), io.StringIO()
        stream = _Tee(terminal, log)
        with (
            patch.object(terminal, "isatty", return_value=True),
            patch("src.run.shutil.get_terminal_size", return_value=os.terminal_size((12, 30))),
        ):
            stream.progress("long progress message", complete=False)
            stream.progress("short", complete=True)
        self.assertEqual(terminal.getvalue(), "\rlong progre\rshort      \n")
        self.assertEqual(log.getvalue(), "short\n")

    def test_redirected_progress_is_quiet_until_completion(self):
        terminal, log = io.StringIO(), io.StringIO()
        stream = _Tee(terminal, log)
        with redirect_stdout(stream), patch("src.run.time.monotonic", return_value=12):
            _print_progress("Generation", 21, 22, 10, 20)
            self.assertEqual(terminal.getvalue(), "")
            self.assertEqual(log.getvalue(), "")
            _print_progress("Generation", 22, 22, 10, 20)
        self.assertEqual(terminal.getvalue(), log.getvalue())
        self.assertIn("22/22 100.0% | 2s 1.00/s ETA 0s", log.getvalue())
        self.assertEqual(log.getvalue().count("\n"), 1)

    def test_progress_finishes_before_errors_and_on_interruption(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as tmp:
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    log, terminal_out, terminal_err = _start_execution_log(Path(tmp))
                    try:
                        sys.stdout.progress("Generation: 2/20", complete=False)
                        if not interrupted:
                            print("Generation error", file=sys.stderr)
                    finally:
                        _stop_execution_log(log, terminal_out, terminal_err)
                saved = (Path(tmp) / "execution.log").read_text(encoding="utf-8")
                self.assertEqual(saved.count("Generation: 2/20"), 1)
                self.assertIn("Generation: 2/20\n", stdout.getvalue())
                if not interrupted:
                    self.assertIn("Generation: 2/20\nGeneration error\n", saved)

    def test_retained_streams_can_write_and_flush_after_log_closes(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                log, terminal_out, terminal_err = _start_execution_log(directory)
                retained_out, retained_err = sys.stdout, sys.stderr
                try:
                    print("saved output")
                    print("saved error", file=sys.stderr)
                finally:
                    _stop_execution_log(log, terminal_out, terminal_err)
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)
                self.assertTrue(log.closed)
                saved = (directory / "execution.log").read_bytes()
                for stream in (retained_out, retained_err):
                    self.assertEqual(stream.write("shutdown reset"), len("shutdown reset"))
                    stream.flush()
                self.assertEqual((directory / "execution.log").read_bytes(), saved)
            self.assertIn(b"saved output", saved)
            self.assertIn(b"saved error", saved)
            self.assertIn("shutdown reset", stdout.getvalue())
            self.assertIn("shutdown reset", stderr.getvalue())


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
            "output_dir": str(self.root / "outputs"),
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
            patch("src.run._select_probe") as select_probe,
            patch("src.run._evaluate_frozen_test") as evaluate_test,
            patch("src.run._create_report") as create_report,
        ):
            train_probes.return_value = (object(), "probe-hash")
            select_probe.return_value = (object(), "selection-hash")
            evaluate_test.return_value = (object(), "test-hash")
            code = main([str(path or self.config), *options])
        self.capture_mock = capture
        self.probe_mock = train_probes
        self.selection_mock = select_probe
        self.test_evaluation_mock = evaluate_test
        self.report_mock = create_report
        return code, stdout.getvalue(), stderr.getvalue()

    def invoke_registration(self, path=None):
        with (
            patch("src.run._generate"),
            patch("src.run._evaluate", return_value={"overall": {
                "total": 0, "correct": 0, "incorrect": 0,
                "abstained": 0, "invalid": 0,
                "correct_prediction_rate": None, "wrong_prediction_rate": None,
            }}),
            patch("src.run._evaluate_custom_metrics"),
        ):
            return self.invoke_full(path, "--behavior-only")

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
            "generation_batch": {"ids": [example["id"]], "seed": 42},
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

        code, stdout, stderr = self.invoke_registration()

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        for text in (
            "Configuration valid.", "Model: organization/model",
            "Reasoning mode: reasoning", "Valid examples: 700",
            "Excluded examples: 0", "Split source: automatic",
            "train: 490", "validation: 105", "test: 105",
            "Run record: created",
            "Model Behavior complete.",
        ):
            self.assertIn(text, stdout)

        run_directories = self.run_directories()
        self.assertEqual(len(run_directories), 1)
        record = json.loads((run_directories[0] / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(run_directories[0].name, "task")
        self.assertEqual(record["run_id"], record["fingerprint"][:12])
        self.assertEqual(record["completed_stages"], ["preparation"])
        self.assertEqual(record["preparation"]["split_sizes"], {
            "train": 490, "validation": 105, "test": 105,
        })
        self.assertEqual(
            set(run_directories[0].iterdir()),
            {run_directories[0] / "run.json", run_directories[0] / "execution.log"},
        )
        self.assertIn("Configuration valid.", (run_directories[0] / "execution.log").read_text())

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

    def test_prepare_only_allows_a_small_smoke_dataset(self):
        self.write_dataset(self.record(index) for index in range(10))
        self.write_config()

        code, stdout, stderr = self.invoke()

        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Valid examples: 10", stdout)
        self.assertIn("Probe-size minimums: deferred", stdout)
        self.assertIn("Preparation complete. No model was run.", stdout)

    def test_prepare_only_writes_nothing_and_ignores_existing_run_artifacts(self):
        self.write_dataset(self.record(index) for index in range(20))
        self.write_config()
        output = self.root / "outputs"
        for existing_output in (False, True):
            with self.subTest(existing_output=existing_output):
                if existing_output:
                    output.mkdir()
                    (output / "run.json").write_text("existing content", encoding="utf-8")
                before = {
                    path: (path.read_bytes(), path.stat().st_mtime_ns)
                    for path in self.root.rglob("*") if path.is_file()
                }
                paths_before = set(self.root.rglob("*"))
                with (
                    patch("src.run.register_run") as register,
                    patch("src.run._start_execution_log") as log,
                    patch("src.run.load_model") as model,
                    patch("src.run._generate") as generate,
                ):
                    code, stdout, stderr = self.invoke()
                self.assertEqual((code, stderr), (0, ""))
                self.assertIn("No files were written.", stdout)
                self.assertNotIn("Run ID:", stdout)
                self.assertNotIn("Run name:", stdout)
                for mock in (register, log, model, generate):
                    mock.assert_not_called()
                self.assertEqual(set(self.root.rglob("*")), paths_before)
                self.assertEqual({
                    path: (path.read_bytes(), path.stat().st_mtime_ns)
                    for path in self.root.rglob("*") if path.is_file()
                }, before)

    def test_full_run_keeps_probe_dataset_minimums(self):
        self.write_dataset(self.record(index) for index in range(10))
        self.write_config()

        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main([str(self.config)])

        self.assertEqual(code, 1)
        self.assertIn("at least 700", stderr.getvalue())
        self.assertFalse((self.root / "outputs").exists())

    def test_unavailable_probe_position_fails_before_model_loading(self):
        self.write_dataset(self.record(index) for index in range(10))
        self.write_config(probe_positions=["span_1"])

        code, _, stderr = self.invoke()

        self.assertEqual(code, 1)
        self.assertIn("probe_positions", stderr)
        self.assertIn("span_1", stderr)

    def test_equivalent_config_and_dataset_paths_reuse_without_rewriting(self):
        self.write_dataset(self.record(index) for index in range(700))
        dataset_copy = self.root / "dataset-copy.jsonl"
        shutil.copyfile(self.dataset, dataset_copy)
        output = self.root / "shared-output"
        self.write_config(output_dir=str(output))

        first_code, first_stdout, _ = self.invoke_registration()
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
        second_code, second_stdout, _ = self.invoke_registration(second_config)

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

        self.write_config(reasoning_max_new_tokens=999, target_tpr=0.8)
        selection_change = prepare_run(self.config)
        selection_id = build_run_identity(
            selection_change.config,
            selection_change.dataset.content_sha256,
            selection_change.split_source,
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
        self.assertEqual(first_id, selection_id)
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

    def test_system_prompt_content_defines_identity(self):
        self.write_dataset(self.record(index) for index in range(700))
        first_prompt = self.root / "first-system.md"
        second_prompt = self.root / "second-system.md"
        first_prompt.write_text("Fixed instructions", encoding="utf-8")
        second_prompt.write_text("Fixed instructions", encoding="utf-8")

        identities = []
        for prompt in (first_prompt, second_prompt):
            self.write_config(system_prompt_path=str(prompt))
            prepared = prepare_run(self.config)
            identities.append(
                build_run_identity(
                    prepared.config,
                    prepared.dataset.content_sha256,
                    prepared.split_source,
                ).run_id
            )

        second_prompt.write_text("Changed instructions", encoding="utf-8")
        self.write_config(system_prompt_path=str(second_prompt))
        changed = prepare_run(self.config)
        changed_id = build_run_identity(
            changed.config,
            changed.dataset.content_sha256,
            changed.split_source,
        ).run_id

        self.assertEqual(identities[0], identities[1])
        self.assertNotEqual(identities[0], changed_id)

    def test_only_active_generation_settings_define_identity(self):
        self.write_dataset(self.record(index) for index in range(700))
        identities = []
        for changes in (
            {},
            {"generation_seed": 7},
            {"direct_batch_size": 4},
            {"decoding_strategy": "greedy"},
        ):
            self.write_config(**changes)
            prepared = prepare_run(self.config)
            identities.append(build_run_identity(
                prepared.config,
                prepared.dataset.content_sha256,
                prepared.split_source,
            ).run_id)
        self.assertEqual(len(set(identities)), 4)

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
            **generation_runtime_versions(),
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
        self.assertIn("Model Behavior results: created", stdout)
        self.assertIn("Model Behavior:", stdout)
        self.assertIn("Correct prediction", stdout)
        self.assertIn("Token-limit output", stdout)
        load_model.assert_called_once_with(prepared.config, pinned_revision=None)

        with patch("src.run.load_model") as load_model:
            code, stdout, stderr = self.invoke_full()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Generation artifact: reused (model not loaded).", stdout)
        self.assertIn("Model Behavior results: reused", stdout)
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
            **generation_runtime_versions(),
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
            patch("src.run._select_probe") as select_probe,
            patch("src.run._evaluate_frozen_test") as evaluate_test,
            patch("src.run._create_report") as create_report,
        ):
            train_probes.return_value = (object(), "probe-hash")
            select_probe.return_value = (object(), "selection-hash")
            evaluate_test.return_value = (object(), "test-hash")
            code = main([str(self.config)])

        self.assertEqual((code, stderr.getvalue()), (0, ""))
        capture.assert_called_once()
        self.assertIs(capture.call_args.args[2], loaded)
        train_probes.assert_called_once()
        select_probe.assert_called_once()
        evaluate_test.assert_called_once()
        create_report.assert_called_once()

    def test_behavior_only_stops_after_evaluation(self):
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
            **generation_runtime_versions(),
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
            code, stdout, stderr = self.invoke_full(None, "--behavior-only")

        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Model Behavior complete", stdout)
        self.capture_mock.assert_not_called()
        self.probe_mock.assert_not_called()
        self.selection_mock.assert_not_called()
        self.test_evaluation_mock.assert_not_called()
        self.report_mock.assert_not_called()

    def test_runner_resumes_saved_batch_plan_after_partial_save(self):
        self.write_dataset(self.record(index, "train") for index in range(10))
        self.write_config(direct_batch_size=4)
        examples = [self.record(index, "train") for index in range(10)]
        records = [self.generation_record(example, "Positive") for example in examples]
        expected_batches = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model", "tokenizer_class": "Tokenizer",
            "dtype": "float32", "device": "cpu",
            **generation_runtime_versions(), "generation_config": {"do_sample": True},
        }

        def interrupted(*_, completed_ids, batch_ids):
            self.assertEqual(set(completed_ids), set())
            self.assertEqual(batch_ids, expected_batches)
            yield GenerationUnit(tuple(records[:3]))
            raise GenerationError("Interrupted after partial save")

        def resumed(*_, completed_ids, batch_ids):
            self.assertEqual(set(completed_ids), {0, 1, 2})
            self.assertEqual(batch_ids, expected_batches)
            yield GenerationUnit(tuple(records[3:]))

        with (
            patch("src.run.load_model", return_value=object()),
            patch("src.run.describe_model", return_value=metadata),
            patch("src.run.generation_units", side_effect=interrupted),
        ):
            code, _, stderr = self.invoke_full(None, "--behavior-only")
        self.assertEqual(code, 1)
        self.assertIn("Interrupted after partial save", stderr)
        path = self.run_directories()[0] / "run.json"
        saved_plan = json.loads(path.read_text(encoding="utf-8"))["generation_batches"]
        with (
            patch("src.run.load_model", return_value=object()),
            patch("src.run.describe_model", return_value=metadata),
            patch("src.run.generation_units", side_effect=resumed),
        ):
            code, _, stderr = self.invoke_full(None, "--behavior-only")
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["generation_batches"], saved_plan)
        with patch("src.run.load_model") as load_model:
            code, _, stderr = self.invoke_full(None, "--behavior-only")
        self.assertEqual((code, stderr), (0, ""))
        load_model.assert_not_called()

    def test_behavior_only_does_not_require_probe_ready_classes(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config()
        prepared = prepare_run(self.config)
        records = tuple(
            self.generation_record(example, "Positive")
            for example in prepared.dataset.examples
        )
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            **generation_runtime_versions(),
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
            code, stdout, stderr = self.invoke_full(None, "--behavior-only")

        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Model Behavior complete", stdout)
        self.assertNotIn("Probe training cannot begin", stdout)
        self.capture_mock.assert_not_called()

    def test_behavior_only_allows_a_small_smoke_dataset(self):
        self.write_dataset(self.record(index) for index in range(10))
        self.write_config()
        prepared = prepare_run(self.config, enforce_probe_sizes=False)
        records = tuple(
            self.generation_record(example, "Positive")
            for example in prepared.dataset.examples
        )
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            **generation_runtime_versions(),
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
            code, stdout, stderr = self.invoke_full(None, "--behavior-only")

        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Valid examples: 10", stdout)
        self.assertIn("Probe-size minimums: deferred", stdout)
        self.assertIn("Model Behavior complete", stdout)
        self.capture_mock.assert_not_called()

    def test_behavior_only_can_print_individual_prediction_examples(self):
        self.write_dataset(self.record(index) for index in range(3))
        self.write_config()
        prepared = prepare_run(self.config, enforce_probe_sizes=False)
        records = tuple(
            self.generation_record(example, "Positive")
            for example in prepared.dataset.examples
        )
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            **generation_runtime_versions(),
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
            code, stdout, stderr = self.invoke_full(
                None, "--behavior-only", "--show-examples", "1"
            )

        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Prediction examples (first 1):", stdout)
        self.assertIn("Input:\nClassify example 0.", stdout)
        self.assertIn("Prediction: 'Positive'", stdout)
        self.assertIn("Target: 'Positive'", stdout)
        self.assertIn("Outcome: correct", stdout)
        self.assertNotIn("Classify example 1.", stdout)
        recap = "Summary: 3 examples | correct: 3 (100.0%) | incorrect: 0 (0.0%) | abstained: 0 | invalid: 0"
        self.assertIn(recap, stdout)
        self.assertGreater(stdout.index(recap), stdout.index("Outcome: correct"))
        self.assertEqual(stdout.strip().splitlines()[-1], f"Results: {self.run_directories()[0]}")
        with patch("src.run.load_model") as load_model:
            code, reused_stdout, stderr = self.invoke_full(
                None, "--behavior-only", "--show-examples", "1"
            )
        self.assertEqual((code, stderr), (0, ""))
        load_model.assert_not_called()
        self.assertIn(recap, reused_stdout)
        self.assertGreater(reused_stdout.index(recap), reused_stdout.index("Outcome: correct"))

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
            **generation_runtime_versions(),
            "generation_config": {},
        }

        def captured(_loaded, batch):
            return {
                i: ActivationResult({
                    name: torch.zeros(2, len(positions), 3, dtype=torch.float16)
                    for name, positions in plan.positions.items()
                }, ("embedding", "hidden_state_1"), 0.0)
                for i, plan in batch.items()
            }

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
            patch("src.run.capture_hidden_state_batch", side_effect=captured) as capture,
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
            patch("src.run.capture_hidden_state_batch") as capture,
        ):
            second_code = main([str(self.config)])

        self.assertEqual((second_code, stderr.getvalue()), (0, ""))
        self.assertIn("Activation artifact: reused (model not loaded).", stdout.getvalue())
        load_model.assert_not_called()
        capture.assert_not_called()

    def test_capture_resume_keeps_original_batch_and_excludes_ineligible_storage(self):
        examples = [self.record(i, "train") for i in range(1, 5)]
        config = TaskConfig("organization/model", self.dataset, "direct", self.root / "outputs")
        prepared = SimpleNamespace(config=config, dataset=SimpleNamespace(examples=examples), answer_matcher=load_answer_matcher(None))
        registered = SimpleNamespace(directory=self.root / "outputs" / "task")
        generations = {e["id"]: self.generation_record(e) for e in examples}
        for g in generations.values():
            g["generation_batch"] = {"ids": [1, 2, 3, 4], "seed": 42}
        generations[4]["answer"].update(text="", token_ids=[], token_logprobs=[], tokens=0)
        evaluations = {i: {"outcome": outcome} for i, outcome in enumerate(
            ("correct", "abstained", "incorrect", "invalid"), 1)}
        run_record = {"generation": {"artifact_sha256": "g" * 64}, "model": {"resolved_revision": "commit"}}
        for fail in (False, True):
            with (
                self.subTest(fail=fail),
                patch("src.run.load_run_record", return_value=run_record),
                patch("src.run._recorded_generation_context", return_value=build_generation_context(config)),
                patch("src.run.load_evaluation_records", return_value=evaluations),
                patch("src.run.load_generation_records", return_value=(generations, False)),
                patch("src.run.validate_completed_activation_capture", return_value=False),
                patch("src.run.load_activation_records", return_value=({1: object()}, False)),
                patch("src.run.reuse_activation_records", return_value={}),
                patch("src.run.capture_hidden_state_batch", return_value={3: object()}) as capture,
                patch("src.run.append_activation_record") as append,
                patch("src.run.finalize_activation_file", return_value="hash"),
                patch("src.run.complete_activation_capture"),
                patch("src.run._activation_summary", return_value={"max_logprob_difference": 0}),
                redirect_stdout(io.StringIO()),
            ):
                if fail:
                    capture.side_effect = ActivationError("Replay probability mismatch")
                    with self.assertRaisesRegex(ActivationError, "probability mismatch"):
                        _capture_activations(prepared, registered, object(), False)
                    append.assert_not_called()
                else:
                    _capture_activations(prepared, registered, object(), False)
                    append.assert_called_once()
                    self.assertEqual(append.call_args.args[3]["id"], 3)
                capture.assert_called_once()
                self.assertEqual(list(capture.call_args.args[1]), [1, 2, 3, 4])
                self.assertEqual(capture.call_args.args[1][4].answer_token_ids, ())

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
            **generation_runtime_versions(),
            "generation_config": {},
        }

        def captured(_loaded, batch):
            return {
                i: ActivationResult({
                    name: torch.zeros(2, len(positions), 3, dtype=torch.float16)
                    for name, positions in plan.positions.items()
                }, ("embedding", "hidden_state_1"), 0.0)
                for i, plan in batch.items()
            }

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
            patch("src.run.capture_hidden_state_batch", side_effect=captured) as capture,
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
        self.assertIn("probe_selection", run_record["completed_stages"])
        self.assertIn("test_evaluation", run_record["completed_stages"])
        self.assertEqual(len(run_record["probe_trainings"]), 1)
        self.assertEqual(len(run_record["probe_selections"]), 1)
        self.assertEqual(len(run_record["test_evaluations"]), 1)
        selection = next(iter(run_record["probe_selections"].values()))
        self.assertEqual(selection["target_tpr"], 0.9)
        self.assertIn("Selected probe:", stdout.getvalue())
        self.assertIn("Output probability:", stdout.getvalue())
        self.assertEqual(next(iter(run_record["probe_trainings"].values()))["seed"], 42)
        self.assertTrue((run_directory / "probes.h5").is_file())

        reused_stdout = io.StringIO()
        with (
            redirect_stdout(reused_stdout),
            redirect_stderr(io.StringIO()),
            patch("src.run.load_model") as load_model,
            patch("src.run.load_tokenizer") as load_tokenizer,
            patch("src.run.capture_hidden_state_batch") as capture,
        ):
            second_code = main([str(self.config)])
        self.assertEqual(second_code, 0)
        load_model.assert_not_called()
        load_tokenizer.assert_not_called()
        capture.assert_not_called()
        self.assertIn("Probe artifact: reused.", reused_stdout.getvalue())
        self.assertIn("Gate evaluation: reused.", reused_stdout.getvalue())

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
                **generation_runtime_versions(),
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
            **generation_runtime_versions(),
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

    def test_force_recompute_validates_model_before_removing_artifacts(self):
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
            **generation_runtime_versions(),
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
        manifest = run_directory / "generation-manifest.jsonl"
        before = manifest.read_bytes()
        with patch(
            "src.run.load_model",
            side_effect=ModelLoadError("replacement is incompatible"),
        ):
            code, _, stderr = self.invoke_full(None, "--force-recompute")

        self.assertEqual(code, 1)
        self.assertIn("replacement is incompatible", stderr)
        self.assertEqual(manifest.read_bytes(), before)
        run_record = json.loads(
            (run_directory / "run.json").read_text(encoding="utf-8")
        )
        self.assertIn("generation", run_record["completed_stages"])

    def test_completed_generation_reuses_recorded_runtime_after_upgrade(self):
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
            **generation_runtime_versions(),
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

        upgraded = {
            "torch_version": "99.0.0",
            "transformers_version": "99.0.0",
        }
        with (
            patch(
                "src.generation_cache.generation_runtime_versions",
                return_value=upgraded,
            ),
            patch("src.run.load_model") as load_model,
        ):
            code, stdout, stderr = self.invoke_full()

        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Generation artifact: reused", stdout)
        load_model.assert_not_called()

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
            **generation_runtime_versions(),
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

        def remaining_units(*_, completed_ids, batch_ids):
            completed_count.append(len(completed_ids))
            self.assertEqual(batch_ids, [[new_record["id"]]])
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
            **generation_runtime_versions(),
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
        self.assertIn("Model Behavior results: created", stdout)
        self.assertIn("train needs 100 correct predictions", stderr)
        self.capture_mock.assert_not_called()
        run_directory = self.run_directories()[0]
        execution_log = (run_directory / "execution.log").read_text(encoding="utf-8")
        self.assertIn("Model Behavior results: created", execution_log)
        self.assertIn("train needs 100 correct predictions", execution_log)
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
            **generation_runtime_versions(),
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
        self.assertIn("Model Behavior results: reused", stdout)
        load_model.assert_not_called()

        copy.write_text(source + "# revised matcher\n", encoding="utf-8")
        with patch("src.run.load_model") as load_model:
            code, stdout, stderr = self.invoke_full()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Generation artifact: reused", stdout)
        self.assertIn("Model Behavior results: created", stdout)
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

    def test_invalid_custom_metrics_stop_before_model_loading(self):
        self.write_dataset(self.record(index) for index in range(700))
        metrics = self.root / "metrics.py"
        metrics.write_text("value = 1\n", encoding="utf-8")
        self.write_config(custom_metrics_path=str(metrics))

        with patch("src.run.load_model") as load_model:
            code, stdout, stderr = self.invoke_full()

        self.assertEqual((code, stdout), (1, ""))
        self.assertIn("must define compute_metrics", stderr)
        self.assertFalse((self.root / "outputs").exists())
        load_model.assert_not_called()

    def test_custom_metrics_run_reuse_and_do_not_change_generation(self):
        records = [
            self.record(index) | {"metadata": {"subset": "priority"}}
            for index in range(700)
        ]
        self.write_dataset(records)
        metrics = self.root / "metrics.py"
        source = (
            "def compute_metrics(records):\n"
            "    chosen = [r for r in records if r['metadata'].get('subset') == 'priority']\n"
            "    correct = sum(r['is_correct'] is True for r in chosen)\n"
            "    return {'subset_accuracy': {\n"
            "        'numerator': correct, 'denominator': len(chosen)}}\n"
        )
        metrics.write_text(source, encoding="utf-8")
        self.write_config(custom_metrics_path=str(metrics))
        prepared = prepare_run(self.config)
        generations = self.probe_ready_generation_records(prepared.dataset.examples)
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            **generation_runtime_versions(),
            "generation_config": {},
        }
        with (
            patch("src.run.load_model", return_value=object()),
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                return_value=iter((GenerationUnit(generations),)),
            ),
        ):
            code, stdout, stderr = self.invoke_full()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Custom metrics: created", stdout)
        self.assertIn("subset_accuracy", stdout)

        with patch("src.run.load_model") as load_model:
            code, stdout, stderr = self.invoke_full()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Custom metrics: reused", stdout)
        load_model.assert_not_called()

        metrics.write_text(source + "# changed reporting code\n", encoding="utf-8")
        with patch("src.run.load_model") as load_model:
            code, stdout, stderr = self.invoke_full()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Generation artifact: reused", stdout)
        self.assertIn("Model Behavior results: reused", stdout)
        self.assertIn("Custom metrics: created", stdout)
        load_model.assert_not_called()
        run_record = json.loads(
            (self.run_directories()[0] / "run.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(run_record["custom_metrics"]), 2)

    def test_metadata_change_reuses_cached_generations(self):
        records = [self.record(index) for index in range(700)]
        self.write_dataset(records)
        self.write_config()
        prepared = prepare_run(self.config)
        generations = self.probe_ready_generation_records(prepared.dataset.examples)
        metadata = {
            "resolved_revision": "resolved-commit",
            "model_class": "Model",
            "tokenizer_class": "Tokenizer",
            "dtype": "float32",
            "device": "cpu",
            **generation_runtime_versions(),
            "generation_config": {},
        }
        with (
            patch("src.run.load_model", return_value=object()),
            patch("src.run.describe_model", return_value=metadata),
            patch(
                "src.run.generation_units",
                return_value=iter((GenerationUnit(generations),)),
            ),
        ):
            self.assertEqual(self.invoke_full()[0], 0)

        self.write_dataset(
            record | {"metadata": {"example_type": "corrupted"}}
            for record in records
        )
        with patch("src.run.load_model") as load_model:
            code, stdout, stderr = self.invoke_full()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Generation cache: 700 reused", stdout)
        self.assertIn("Generation complete from cached records; model not loaded", stdout)
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
            **generation_runtime_versions(),
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
        self.assertIn("Model Behavior results: created", stdout)
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
        run_directory = prepared.config.output_dir / self.config.stem
        run_directory.mkdir(parents=True)

        code, stdout, stderr = self.invoke_registration()
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("Run record: created", stdout)

        (run_directory / "run.json").write_text("not json", encoding="utf-8")
        code, stdout, stderr = self.invoke_registration()
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Run storage error:", stderr)
        self.assertNotIn("Traceback", stderr)

        shutil.rmtree(run_directory)
        run_directory.mkdir()
        (run_directory / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        code, stdout, stderr = self.invoke_registration()
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
