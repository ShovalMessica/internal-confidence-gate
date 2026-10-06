"""Offline tests for task-independent answer evaluation."""

import unittest
from pathlib import Path
import tempfile

from src.evaluation import (
    EvaluationError,
    build_custom_metrics_identity,
    build_evaluation_identity,
    evaluate_answers,
    evaluate_custom_metrics,
    load_answer_matcher,
    load_custom_metrics,
    normalize_answer,
    probe_shortages,
)


def _example(example_id, target="answer", split="train"):
    return {
        "id": example_id,
        "split": split,
        "target_answer": target,
    }


def _generation(example, text, stop_reason="eos"):
    return {
        "id": example["id"],
        "split": example["split"],
        "status": "success",
        "answer": {"text": text, "stop_reason": stop_reason},
    }


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_normalization_and_complete_answer_matching(self):
        examples = [
            _example(1, "  STRASSE   NAME "),
            _example(2, "answer"),
        ]
        generations = {
            1: _generation(examples[0], " Straße\tname "),
            2: _generation(examples[1], "answer with explanation"),
        }
        result = evaluate_answers(examples, generations, allow_abstention=True)

        self.assertEqual(normalize_answer(" Straße\tNAME "), "strasse name")
        self.assertEqual(
            [record["outcome"] for record in result.records],
            ["correct", "incorrect"],
        )
        self.assertEqual(result.records[0]["normalized_answer"], "strasse name")
        self.assertNotIn("invalid_reason", result.records[1])

    def test_structural_failures_are_invalid(self):
        examples = [_example(index) for index in range(4)]
        generations = {
            0: {
                "id": 0,
                "split": "train",
                "status": "failed",
                "failure": {"reason": "context_length"},
            },
            1: _generation(examples[1], " \n "),
            2: _generation(examples[2], "FINAL: answer"),
            3: _generation(examples[3], "answer\nexplanation"),
        }
        result = evaluate_answers(examples, generations, allow_abstention=True)
        self.assertEqual(
            [record["invalid_reason"] for record in result.records],
            [
                "generation_failed",
                "empty_answer",
                "repeated_final_marker",
                "multiple_nonempty_lines",
            ],
        )
        self.assertTrue(all(record["is_correct"] is None for record in result.records))

    def test_unknown_depends_on_abstention_setting(self):
        example = _example(1, target="UNKNOWN")
        generation = {1: _generation(example, "  UnKnOwN ")}

        enabled_example = _example(1, target="answer")
        enabled = evaluate_answers(
            [enabled_example],
            {1: _generation(enabled_example, "  UnKnOwN ")},
            allow_abstention=True,
        )
        disabled = evaluate_answers(
            [example], generation, allow_abstention=False
        )

        self.assertEqual(enabled.records[0]["outcome"], "abstained")
        self.assertIsNone(enabled.records[0]["is_correct"])
        self.assertEqual(disabled.records[0]["outcome"], "correct")
        self.assertTrue(disabled.records[0]["is_correct"])

    def test_token_limit_answer_remains_valid_and_is_reported(self):
        example = _example(1, target="answer")
        result = evaluate_answers(
            [example],
            {1: _generation(example, "answer", stop_reason="token_limit")},
            allow_abstention=True,
        )
        self.assertEqual(result.records[0]["outcome"], "correct")
        self.assertEqual(result.summary["overall"]["answer_token_limit"], 1)
        self.assertEqual(result.summary["overall"]["token_limit_rate"], 1.0)

    def test_model_behavior_rates_use_every_example(self):
        examples = [_example(index) for index in range(4)]
        generations = {
            0: _generation(examples[0], "answer"),
            1: _generation(examples[1], "wrong", stop_reason="token_limit"),
            2: _generation(examples[2], "UNKNOWN"),
            3: {
                "id": 3,
                "split": "train",
                "status": "failed",
                "failure": {"reason": "context_length"},
            },
        }
        metrics = evaluate_answers(
            examples, generations, allow_abstention=True
        ).summary["overall"]

        self.assertEqual(
            [
                metrics["correct_prediction_rate"],
                metrics["wrong_prediction_rate"],
                metrics["missed_prediction_rate"],
                metrics["invalid_output_rate"],
                metrics["token_limit_rate"],
            ],
            [0.25, 0.25, 0.25, 0.25, 0.25],
        )

    def test_custom_matcher_controls_correctness(self):
        path = self.root / "matcher.py"
        path.write_text(
            "def answer_match(prediction, target_answer):\n"
            "    return prediction.rstrip('.').casefold() == target_answer.casefold()\n",
            encoding="utf-8",
        )
        matcher = load_answer_matcher(path)
        example = _example(1, target="London")
        result = evaluate_answers(
            [example],
            {1: _generation(example, "LONDON.")},
            allow_abstention=True,
            matcher=matcher,
        )

        self.assertEqual(result.records[0]["outcome"], "correct")
        self.assertEqual(matcher.source_path, path)
        self.assertEqual(
            build_evaluation_identity("generation", matcher).matcher_sha256,
            matcher.sha256,
        )

    def test_invalid_custom_matchers_fail_clearly(self):
        cases = {
            "missing": "value = 1\n",
            "signature": "def answer_match(prediction):\n    return True\n",
            "syntax": "def answer_match(:\n",
        }
        for name, source in cases.items():
            with self.subTest(name=name):
                path = self.root / f"{name}.py"
                path.write_text(source, encoding="utf-8")
                with self.assertRaises(EvaluationError):
                    load_answer_matcher(path)

        example = _example(1)
        for name, body, message in (
            ("return", "return 1", "return true or false"),
            ("raise", "raise RuntimeError('broken')", "RuntimeError"),
        ):
            with self.subTest(name=name):
                path = self.root / f"runtime_{name}.py"
                path.write_text(
                    "def answer_match(prediction, target_answer):\n"
                    f"    {body}\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(EvaluationError, message):
                    evaluate_answers(
                        [example],
                        {1: _generation(example, "answer")},
                        allow_abstention=True,
                        matcher=load_answer_matcher(path),
                    )

    def test_custom_metrics_receive_documented_records_and_calculate_rates(self):
        path = self.root / "metrics.py"
        path.write_text(
            "def compute_metrics(records):\n"
            "    selected = [r for r in records if r['metadata'].get('kind') == 'x']\n"
            "    correct = sum(r['is_correct'] is True for r in selected)\n"
            "    return {'selected_accuracy': {\n"
            "        'numerator': correct, 'denominator': len(selected)}}\n",
            encoding="utf-8",
        )
        custom = load_custom_metrics(path)
        examples = [
            {**_example(1, split="train"), "metadata": {"kind": "x"}},
            {**_example(2, split="validation"), "metadata": {"other": 1}},
            {**_example(3, split="test"), "metadata": {"kind": "x"}},
        ]
        generations = {
            1: _generation(examples[0], "answer"),
            2: _generation(examples[1], "wrong", stop_reason="token_limit"),
            3: _generation(examples[2], "UNKNOWN"),
        }
        evaluated = evaluate_answers(examples, generations, allow_abstention=True)
        evaluations = {record["id"]: record for record in evaluated.records}
        results = evaluate_custom_metrics(
            examples, generations, evaluations, custom
        )

        self.assertEqual(
            results["overall"]["selected_accuracy"],
            {"numerator": 1, "denominator": 2, "rate": 0.5},
        )
        self.assertEqual(
            results["by_split"]["validation"]["selected_accuracy"],
            {"numerator": 0, "denominator": 0, "rate": None},
        )
        identity = build_custom_metrics_identity("data", "generation", "evaluation", custom)
        self.assertEqual(identity.implementation_sha256, custom.sha256)

    def test_custom_metric_failures_are_clear_and_scope_specific(self):
        sources = {
            "missing": "value = 1\n",
            "signature": "def compute_metrics():\n    return {}\n",
            "syntax": "def compute_metrics(:\n",
        }
        for name, source in sources.items():
            with self.subTest(name=name):
                path = self.root / f"{name}_metrics.py"
                path.write_text(source, encoding="utf-8")
                with self.assertRaises(EvaluationError):
                    load_custom_metrics(path)

        invalid_outputs = {
            "empty": "{}",
            "extra": "{'metric': {'numerator': 0, 'denominator': 1, 'extra': 2}}",
            "boolean": "{'metric': {'numerator': False, 'denominator': 1}}",
            "negative": "{'metric': {'numerator': -1, 'denominator': 1}}",
            "oversized": "{'metric': {'numerator': 2, 'denominator': 1}}",
        }
        example = {**_example(1), "metadata": {}}
        generation = {1: _generation(example, "answer")}
        evaluation = evaluate_answers(
            [example], generation, allow_abstention=True
        ).records[0]
        for name, output in invalid_outputs.items():
            with self.subTest(name=name):
                path = self.root / f"{name}.py"
                path.write_text(
                    f"def compute_metrics(records):\n    return {output}\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(EvaluationError, "overall"):
                    evaluate_custom_metrics(
                        [example], generation, {1: evaluation}, load_custom_metrics(path)
                    )

        path = self.root / "raises.py"
        path.write_text(
            "def compute_metrics(records):\n    raise RuntimeError('broken')\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EvaluationError, "overall.*RuntimeError"):
            evaluate_custom_metrics(
                [example], generation, {1: evaluation}, load_custom_metrics(path)
            )

    def test_custom_metrics_require_consistent_names_across_scopes(self):
        path = self.root / "different.py"
        path.write_text(
            "def compute_metrics(records):\n"
            "    name = records[0]['split'] if records else 'empty'\n"
            "    return {name: {'numerator': 0, 'denominator': len(records)}}\n",
            encoding="utf-8",
        )
        examples = [
            {**_example(1, split="train"), "metadata": {}},
            {**_example(2, split="validation"), "metadata": {}},
            {**_example(3, split="test"), "metadata": {}},
        ]
        generations = {item["id"]: _generation(item, "answer") for item in examples}
        evaluations = {
            item["id"]: result
            for item, result in zip(
                examples,
                evaluate_answers(examples, generations, allow_abstention=True).records,
            )
        }
        with self.assertRaisesRegex(EvaluationError, "different metric names"):
            evaluate_custom_metrics(
                examples, generations, evaluations, load_custom_metrics(path)
            )

    def test_exact_probe_minimums_pass_and_shortages_are_explicit(self):
        examples = []
        generations = {}
        example_id = 0
        for split, per_outcome in (
            ("train", 100),
            ("validation", 50),
            ("test", 50),
        ):
            for correct in (True, False):
                for _ in range(per_outcome):
                    example = _example(example_id, split=split)
                    examples.append(example)
                    generations[example_id] = _generation(
                        example, "answer" if correct else "wrong"
                    )
                    example_id += 1

        ready = evaluate_answers(examples, generations, allow_abstention=True)
        self.assertTrue(ready.summary["probe_ready"])
        self.assertEqual(ready.shortages, ())
        self.assertEqual(probe_shortages(ready.records), ())

        generations[0] = _generation(examples[0], "UNKNOWN")
        short = evaluate_answers(examples, generations, allow_abstention=True)
        self.assertFalse(short.summary["probe_ready"])
        self.assertEqual(
            short.shortages,
            ({
                "split": "train",
                "outcome": "correct",
                "required": 100,
                "actual": 99,
                "missing": 1,
            },),
        )

    def test_generation_ids_and_answer_shape_must_be_valid(self):
        example = _example(1)
        with self.assertRaisesRegex(EvaluationError, "do not match"):
            evaluate_answers([example], {}, allow_abstention=True)
        with self.assertRaisesRegex(EvaluationError, "answer text"):
            evaluate_answers(
                [example],
                {1: {"id": 1, "split": "train", "status": "success", "answer": {}}},
                allow_abstention=True,
            )


if __name__ == "__main__":
    unittest.main()
