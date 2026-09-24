"""Offline checks for generation artifact integrity and recovery."""

import json
from pathlib import Path
import tempfile
import unittest

from src.generation import (
    GENERATION_PROTOCOL_VERSION,
    GENERATION_RECORD_SCHEMA_VERSION,
)
from src.evaluation import (
    EVALUATION_PROTOCOL_VERSION,
    EVALUATION_RECORD_SCHEMA_VERSION,
)
from src.run_store import (
    RunStoreError,
    complete_evaluation,
    evaluation_artifact_path,
    generation_artifact_path,
    load_evaluation_records,
    load_run_record,
    load_generation_records,
    reset_generation,
    reset_partial_direct_batches,
    validate_completed_evaluation,
    write_evaluation_records,
)


def _example(example_id):
    return {"id": example_id, "split": "train"}


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

    def test_only_truncated_final_line_is_recovered(self):
        first = _record(1)
        self.path.write_text(
            json.dumps(first) + "\n" + '{"id": 2', encoding="utf-8"
        )
        records, recovered = load_generation_records(
            self.directory, [_example(1), _example(2)]
        )
        self.assertTrue(recovered)
        self.assertEqual(records, {1: first})
        self.assertTrue(self.path.read_bytes().endswith(b"\n"))

        self.path.write_text("not-json\n" + '{"id": 2', encoding="utf-8")
        with self.assertRaisesRegex(RunStoreError, "line 1"):
            load_generation_records(self.directory, [_example(1), _example(2)])

    def test_duplicate_and_unexpected_ids_are_rejected(self):
        for rows, message in (
            ([_record(1), _record(1)], "Duplicate"),
            ([_record(9)], "unexpected"),
        ):
            with self.subTest(message=message):
                self.path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RunStoreError, message):
                    load_generation_records(
                        self.directory, [_example(1), _example(2)]
                    )

    def test_partial_direct_batch_is_removed_before_resume(self):
        examples = [_example(index) for index in range(6)]
        records = {index: _record(index) for index in (0, 1, 2, 4, 5)}
        self.path.write_text(
            "".join(json.dumps(record) + "\n" for record in records.values()),
            encoding="utf-8",
        )
        retained, removed = reset_partial_direct_batches(
            self.directory, records, examples, batch_size=4
        )
        self.assertEqual((set(retained), removed), ({4, 5}, 3))
        loaded, _ = load_generation_records(self.directory, examples)
        self.assertEqual(set(loaded), {4, 5})

    def test_evaluation_artifact_validation_and_hash_detection(self):
        examples = [_example(1), _example(2)]
        records = [_evaluation(1), _evaluation(2, "incorrect")]
        artifact_hash = write_evaluation_records(self.directory, records)
        loaded = load_evaluation_records(self.directory, examples)
        self.assertEqual(list(loaded), [1, 2])

        (self.directory / "run.json").write_text(
            json.dumps({"completed_stages": ["preparation", "generation"]}),
            encoding="utf-8",
        )
        summary = {"probe_ready": False, "shortages": []}
        complete_evaluation(
            self.directory, artifact_hash, "generation-hash", summary
        )
        run_record = load_run_record(self.directory)
        self.assertTrue(
            validate_completed_evaluation(
                self.directory, run_record, "generation-hash"
            )
        )

        evaluation_artifact_path(self.directory).write_text(
            json.dumps(records[0]) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(RunStoreError, "missing or has changed"):
            validate_completed_evaluation(
                self.directory, run_record, "generation-hash"
            )

        write_evaluation_records(self.directory, list(reversed(records)))
        with self.assertRaisesRegex(RunStoreError, "dataset order"):
            load_evaluation_records(self.directory, examples)

    def test_force_reset_removes_downstream_evaluation(self):
        evaluation_artifact_path(self.directory).write_text("{}\n", encoding="utf-8")
        generation_artifact_path(self.directory).write_text("{}\n", encoding="utf-8")
        (self.directory / "run.json").write_text(
            json.dumps(
                {
                    "completed_stages": ["preparation", "generation", "evaluation"],
                    "model": {"resolved_revision": "commit"},
                    "generation": {},
                    "evaluation": {},
                }
            ),
            encoding="utf-8",
        )
        pinned = reset_generation(self.directory)
        record = load_run_record(self.directory)
        self.assertEqual(pinned, "commit")
        self.assertEqual(record["completed_stages"], ["preparation"])
        self.assertNotIn("evaluation", record)
        self.assertFalse(evaluation_artifact_path(self.directory).exists())


if __name__ == "__main__":
    unittest.main()
