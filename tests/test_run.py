"""Offline checks for the validation runner."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from src.run import main, prepare_run


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

    def write_config(self, **changes):
        values = {
            "model_name_or_path": "organization/model",
            "dataset_path": str(self.dataset),
            "reasoning_mode": "direct",
        }
        values.update(changes)
        self.config.write_text(yaml.safe_dump(values), encoding="utf-8")

    def invoke(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main([str(self.config)])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_automatic_splits_print_summary_without_side_effects(self):
        self.write_dataset(self.record(index) for index in range(700))
        self.write_config(reasoning_mode="reasoning")
        before = {path.name for path in self.root.iterdir()}

        code, stdout, stderr = self.invoke()

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        for text in (
            "Configuration valid.", "Model: organization/model",
            "Reasoning mode: reasoning", "Valid examples: 700",
            "Excluded examples: 0", "Split source: automatic",
            "train: 490", "validation: 105", "test: 105",
            "Validation complete. No model was run.",
        ):
            self.assertIn(text, stdout)
        self.assertEqual({path.name for path in self.root.iterdir()}, before)
        self.assertFalse((self.root / "outputs").exists())

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
        self.assertIn("Line 2, ID 900: target_answer cannot be UNKNOWN", stdout)

    def test_configuration_error_returns_one_without_traceback(self):
        self.write_dataset([self.record(1)])
        self.write_config(model_name_or_path=None)

        code, stdout, stderr = self.invoke()

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Configuration errors:", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_dataset_error_returns_one_without_traceback(self):
        self.write_dataset(self.record(index) for index in range(10))
        self.write_config()

        code, stdout, stderr = self.invoke()

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("at least 700", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_usage_error_uses_argparse_exit_code(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main([])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
