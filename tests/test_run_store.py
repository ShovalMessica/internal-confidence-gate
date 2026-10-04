"""Offline checks for generation artifact integrity and recovery."""

import json
from pathlib import Path
import tempfile
import unittest

from src.generation import (
    GENERATION_PROTOCOL_VERSION,
    GENERATION_RECORD_SCHEMA_VERSION,
)
from src.config import TaskConfig
from src.generation_cache import (
    build_generation_context,
    load_context_model,
    save_context_model,
)
from src.evaluation import (
    EVALUATION_PROTOCOL_VERSION,
    EVALUATION_RECORD_SCHEMA_VERSION,
    build_evaluation_identity,
    load_answer_matcher,
)
from src.run_store import (
    RunStoreError,
    append_generation_records,
    complete_evaluation,
    evaluation_artifact_path,
    generation_artifact_path,
    load_evaluation_records,
    load_run_record,
    load_generation_records,
    reset_generation,
    reuse_cached_generation_records,
    validate_completed_evaluation,
    write_evaluation_records,
)


def _example(example_id):
    return {"id": example_id, "input": f"Task {example_id}", "split": "train"}


def _record(example_id):
    return {
        "schema_version": GENERATION_RECORD_SCHEMA_VERSION,
        "protocol_version": GENERATION_PROTOCOL_VERSION,
        "id": example_id,
        "split": "train",
        "status": "success",
        "formatted_prompt_token_ids": [1],
        "final_control": {"token_ids": [2]},
        "answer": {"token_ids": [3]},
    }


def _evaluation(example_id, outcome="correct"):
    return {
        "schema_version": EVALUATION_RECORD_SCHEMA_VERSION,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "id": example_id,
        "split": "train",
        "outcome": outcome,
        "normalized_answer": "answer" if outcome != "invalid" else None,
        "is_correct": {"correct": True, "incorrect": False}.get(outcome),
        **({"invalid_reason": "empty_answer"} if outcome == "invalid" else {}),
    }


class RunStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = generation_artifact_path(self.directory)
        self.context = build_generation_context(
            TaskConfig(
                model_name_or_path="organization/model",
                dataset_path=self.directory / "data.jsonl",
                reasoning_mode="direct",
                output_dir=self.directory / "outputs",
            )
        )

    def append(self, examples, records):
        append_generation_records(
            self.directory,
            self.context,
            {example["id"]: example for example in examples},
            records,
        )

    def test_only_truncated_final_line_is_recovered(self):
        first = _record(1)
        examples = [_example(1), _example(2)]
        self.append(examples, [first])
        with self.path.open("a", encoding="utf-8") as output:
            output.write('{"id": 2')
        records, recovered = load_generation_records(
            self.directory, self.context, examples
        )
        self.assertTrue(recovered)
        self.assertEqual(records, {1: first})
        self.assertTrue(self.path.read_bytes().endswith(b"\n"))

        self.path.write_text("not-json\n" + '{"id": 2', encoding="utf-8")
        with self.assertRaisesRegex(RunStoreError, "line 1"):
            load_generation_records(self.directory, self.context, examples)

    def test_duplicate_and_unexpected_ids_are_rejected(self):
        examples = [_example(1), _example(2)]
        self.append(examples, [_record(1), _record(1)])
        with self.assertRaisesRegex(RunStoreError, "Duplicate"):
            load_generation_records(self.directory, self.context, examples)

        self.path.unlink()
        unexpected = _example(9)
        self.append([unexpected], [_record(9)])
        with self.assertRaisesRegex(RunStoreError, "invalid"):
            load_generation_records(self.directory, self.context, examples)

    def test_cache_reuses_unchanged_examples_and_adapts_split(self):
        original = [_example(1)]
        self.append(original, [_record(1)])
        second_run = self.directory / "second"
        second_run.mkdir()
        changed_split = [{**original[0], "split": "test"}]
        reused = reuse_cached_generation_records(
            second_run, self.context, changed_split, set()
        )
        self.assertEqual(reused[1]["split"], "test")
        loaded, _ = load_generation_records(
            second_run, self.context, changed_split
        )
        self.assertEqual(loaded, reused)

        changed_spans = [{
            **changed_split[0],
            "semantic_spans": {
                "span_1": {"start_char": 0, "end_char": 4},
            },
        }]
        span_run = self.directory / "spans"
        span_run.mkdir()
        self.assertEqual(
            set(reuse_cached_generation_records(
                span_run, self.context, changed_spans, set()
            )),
            {1},
        )

        changed_input = [{**changed_split[0], "input": "Changed"}]
        third_run = self.directory / "third"
        third_run.mkdir()
        self.assertEqual(
            reuse_cached_generation_records(
                third_run, self.context, changed_input, set()
            ),
            {},
        )

    def test_cache_context_pins_model_metadata(self):
        metadata = {"resolved_revision": "commit", "model_class": "Model"}
        self.assertIsNone(load_context_model(self.context))
        save_context_model(self.context, metadata)
        self.assertEqual(load_context_model(self.context), metadata)

    def test_evaluation_artifact_validation_and_hash_detection(self):
        examples = [_example(1), _example(2)]
        records = [_evaluation(1), _evaluation(2, "incorrect")]
        matcher = load_answer_matcher(None)
        identity = build_evaluation_identity("generation-hash", matcher)
        artifact_hash = write_evaluation_records(
            self.directory, identity.evaluation_id, records
        )
        loaded = load_evaluation_records(
            self.directory, identity.evaluation_id, examples
        )
        self.assertEqual(list(loaded), [1, 2])

        (self.directory / "run.json").write_text(
            json.dumps({"completed_stages": ["preparation", "generation"]}),
            encoding="utf-8",
        )
        summary = {"probe_ready": False, "shortages": []}
        complete_evaluation(
            self.directory, identity, matcher, artifact_hash, summary
        )
        run_record = load_run_record(self.directory)
        self.assertTrue(
            validate_completed_evaluation(
                self.directory, run_record, identity
            )
        )

        evaluation_artifact_path(self.directory, identity.evaluation_id).write_text(
            json.dumps(records[0]) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(RunStoreError, "missing or has changed"):
            validate_completed_evaluation(
                self.directory, run_record, identity
            )

        write_evaluation_records(
            self.directory, identity.evaluation_id, list(reversed(records))
        )
        with self.assertRaisesRegex(RunStoreError, "dataset order"):
            load_evaluation_records(
                self.directory, identity.evaluation_id, examples
            )

    def test_force_reset_removes_downstream_artifacts(self):
        evaluation_id = "a" * 12
        path = evaluation_artifact_path(self.directory, evaluation_id)
        path.parent.mkdir()
        path.write_text("{}\n", encoding="utf-8")
        (self.directory / "evaluations.jsonl").write_text("{}\n", encoding="utf-8")
        activation_path = self.directory / "activations.h5"
        activation_path.write_bytes(b"saved activations")
        probe_path = self.directory / "probes.h5"
        probe_path.write_bytes(b"saved probes")
        generation_artifact_path(self.directory).write_text("{}\n", encoding="utf-8")
        (self.directory / "run.json").write_text(
            json.dumps(
                {
                    "completed_stages": [
                        "preparation",
                        "generation",
                        "evaluation",
                        "activation_capture",
                        "probe_training",
                        "probe_selection",
                        "test_evaluation",
                    ],
                    "model": {"resolved_revision": "commit"},
                    "generation": {},
                    "evaluation": {},
                    "evaluations": {evaluation_id: {}},
                    "activation_capture": {},
                    "probe_trainings": {"probe": {}},
                    "probe_selections": {"selection": {}},
                    "test_evaluations": {"test": {}},
                }
            ),
            encoding="utf-8",
        )
        pinned = reset_generation(self.directory)
        record = load_run_record(self.directory)
        self.assertEqual(pinned, "commit")
        self.assertEqual(record["completed_stages"], ["preparation"])
        self.assertNotIn("evaluation", record)
        self.assertNotIn("evaluations", record)
        self.assertNotIn("activation_capture", record)
        self.assertNotIn("probe_trainings", record)
        self.assertNotIn("probe_selections", record)
        self.assertNotIn("test_evaluations", record)
        self.assertFalse(path.exists())
        self.assertFalse((self.directory / "evaluations").exists())
        self.assertFalse((self.directory / "evaluations.jsonl").exists())
        self.assertFalse(activation_path.exists())
        self.assertFalse(probe_path.exists())


if __name__ == "__main__":
    unittest.main()
