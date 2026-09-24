"""Offline tests for task-independent answer evaluation."""

import unittest

from src.evaluation import EvaluationError, evaluate_answers, normalize_answer


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
