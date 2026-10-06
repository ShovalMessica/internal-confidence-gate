"""Offline checks for dataset validation and exclusion reports."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from src.config import SplitRatios
from src.dataset import DatasetError, DatasetResult, assign_splits, load_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "dataset.jsonl"

    def record(self, id=1, **values):
        return {"id": id, "input": "Please ask Jhon.", "target_answer": "John", **values}

    def write(self, *records):
        self.source.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
            encoding="utf-8",
        )
        return self.source

    def test_valid_records_preserve_text_order_and_have_no_side_effects(self):
        records = [self.record(4, input="  Review:\nGreat!\t", target_answer=" Positive "),
                   self.record(-2), self.record(0)]
        self.write(*records)
        before = self.source.read_bytes()
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = load_dataset(self.source)
        self.assertEqual(result.examples, records)
        self.assertEqual(result.excluded, [])
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(result.content_sha256, hashlib.sha256(before).hexdigest())
        self.assertEqual(list(self.root.iterdir()), [self.source])

    def test_metadata_is_preserved_and_other_extra_fields_are_ignored(self):
        record = self.record(metadata={"source": "user"}, semantic_spans={
            "span_3": {"start_char": 11, "end_char": 15, "note": "name"}},
            ignored="value")
        result = load_dataset(self.write(record))
        self.assertEqual(result.examples, [self.record(
            metadata={"source": "user"},
            semantic_spans={"span_3": {"start_char": 11, "end_char": 15}},
        )])

    def test_metadata_is_optional_free_form_and_may_differ(self):
        records = [
            self.record(1, metadata={"example_type": "corrupted", "nested": {"x": [1]}}),
            self.record(2, metadata={}),
            self.record(3, metadata={"difficulty": 4}),
            self.record(4),
        ]
        self.assertEqual(load_dataset(self.write(*records)).examples, records)

    def test_metadata_must_be_an_object(self):
        for value in (None, [], "type", 1, True):
            with self.subTest(value=value):
                result = load_dataset(
                    self.write(self.record(1, metadata=value), self.record(2))
                )
                self.assertEqual(result.examples, [self.record(2)])
                self.assertIn("metadata must be an object", result.excluded[0]["reasons"][0])

    def test_invalid_fields_are_reported_and_skipped(self):
        cases = {
            "id": [None, True, False, 1.0, "1", []],
            "input": [None, "", " \n\t", 3, []],
            "target_answer": [None, "", " \t", 3, [], "  UnKnOwN\n", "FINAL: John",
                              "text final: John"],
            "split": [None, "", "dev", True, []],
            "semantic_spans": [None, [], "span_1"],
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    result = load_dataset(self.write(self.record(**{field: value}), self.record(99)))
                    self.assertEqual(result.examples, [self.record(99)])
                    self.assertEqual(result.excluded[0]["line"], 1)
                    self.assertIn(field, " ".join(result.excluded[0]["reasons"]))

    def test_unknown_target_is_allowed_when_abstention_is_disabled(self):
        record = self.record(target_answer="UNKNOWN")
        result = load_dataset(self.write(record), allow_abstention=False)
        self.assertEqual(result.examples, [record])

    def test_missing_fields_report_all_reasons(self):
        result = load_dataset(self.write({}, self.record()))
        exclusion = result.excluded[0]
        self.assertIsNone(exclusion["id"])
        self.assertEqual(len(exclusion["reasons"]), 3)
        for field in ("id", "input", "target_answer"):
            self.assertIn(field, " ".join(exclusion["reasons"]))

    def test_invalid_json_blank_lines_and_nonobjects(self):
        invalid = ['{"id":', "", "  ", "null", "42", "[]",
                   '{"metadata": NaN}', '{"metadata": Infinity}', '{"metadata": -Infinity}']
        self.source.write_text("\n".join(invalid + [json.dumps(self.record())]), encoding="utf-8")
        result = load_dataset(self.source)
        self.assertEqual(result.examples, [self.record()])
        self.assertEqual([item["line"] for item in result.excluded], list(range(1, len(invalid) + 1)))
        self.assertTrue(all(item["reasons"] for item in result.excluded))

    def test_duplicate_json_keys_are_rejected_including_nested_metadata(self):
        for raw in ('{"id":1,"id":2}', '{"metadata":{"key":1,"key":2}}',
                    '{"semantic_spans":{"span_1":{"start_char":0,"start_char":1}}}'):
            with self.subTest(raw=raw):
                self.source.write_text(raw + "\n" + json.dumps(self.record()), encoding="utf-8")
                result = load_dataset(self.source)
                self.assertIn("Duplicate JSON key", result.excluded[0]["reasons"][0])
                self.assertEqual(result.examples, [self.record()])

    def test_first_id_occurrence_reserves_id_even_when_invalid(self):
        result = load_dataset(self.write(
            self.record(7, input=""), self.record(7), self.record(8), self.record(8, input="different")
        ))
        self.assertEqual(result.examples, [self.record(8)])
        self.assertEqual([item["id"] for item in result.excluded], [7, 7, 8])
        self.assertIn("Duplicate id 7", result.excluded[1]["reasons"][0])
        self.assertIn("Duplicate id 8", result.excluded[2]["reasons"][0])

    def test_span_boundaries_unicode_and_nonconsecutive_keys(self):
        # Python code points: A, emoji, e, combining accent, newline, Hebrew letter.
        text = "A\U0001f600e\u0301\n\u05d0"
        spans = {"span_2": {"start_char": 1, "end_char": 2},
                 "span_9": {"start_char": 0, "end_char": len(text)}}
        records = [self.record(1, input=text, semantic_spans=spans),
                   self.record(2, semantic_spans={"span_9": {"start_char": 0, "end_char": 15},
                                                  "span_2": {"start_char": 11, "end_char": 15}})]
        for escape in (True, False):
            with self.subTest(escape=escape):
                self.source.write_text("\n".join(json.dumps(r, ensure_ascii=escape) for r in records),
                                       encoding="utf-8")
                result = load_dataset(self.source)
                self.assertEqual(result.examples, records)
                self.assertEqual(result.examples[0]["input"][1:2], "\U0001f600")

    def test_invalid_span_keys_shapes_and_coordinates(self):
        invalid = [{name: {"start_char": 0, "end_char": 1}}
                   for name in ("span_0", "span_01", "span_-1", "span_1x", "entity", "span_\u0661")]
        invalid += [{"span_1": value} for value in (None, [], 1, {}, {"start_char": 0})]
        invalid += [{"span_1": {"start_char": start, "end_char": end}}
                    for start, end in ((True, 1), (0, True), (0.0, 1), (0, "1"),
                                       (-1, 1), (1, 1), (2, 1), (0, len(self.record()["input"]) + 1))]
        for spans in invalid:
            with self.subTest(spans=spans):
                result = load_dataset(self.write(self.record(semantic_spans=spans), self.record(99)))
                self.assertEqual(result.examples, [self.record(99)])
                self.assertIn("semantic_spans", " ".join(result.excluded[0]["reasons"]))

    def test_omitted_and_empty_spans_are_consistent(self):
        records = [self.record(1), self.record(2, semantic_spans={})]
        self.assertEqual(load_dataset(self.write(*records)).examples, records)

    def test_assigned_splits_are_preserved_without_requiring_each_split(self):
        records = [self.record(1, split="train"), self.record(2, split="test")]
        self.assertEqual(load_dataset(self.write(*records)).examples, records)

    def test_partial_splits_fail_and_preserve_exclusions(self):
        with self.assertRaises(DatasetError) as caught:
            load_dataset(self.write(self.record(1, split="train"), {}, self.record(2)))
        self.assertIn("Supply split", str(caught.exception))
        self.assertEqual(caught.exception.excluded[0]["line"], 2)

    def test_mismatched_span_sets_fail_in_either_order(self):
        span = {"start_char": 0, "end_char": 1}
        for first, second in (({}, {"span_1": span}), ({"span_1": span}, {}),
                              ({"span_1": span}, {"span_2": span})):
            with self.subTest(first=first, second=second):
                with self.assertRaisesRegex(DatasetError, "Semantic-span keys differ at line 2"):
                    load_dataset(self.write(self.record(1, semantic_spans=first),
                                            self.record(2, semantic_spans=second)))

    def test_consistency_ignores_excluded_examples(self):
        valid = self.record(9)
        result = load_dataset(self.write(
            self.record(1, target_answer="UNKNOWN", split="train",
                        semantic_spans={"span_1": {"start_char": 0, "end_char": 1}}),
            valid, self.record(9, split="test")
        ))
        self.assertEqual(result.examples, [valid])
        self.assertEqual(len(result.excluded), 2)

    def test_empty_and_fully_excluded_datasets_fail(self):
        for records in ((), ({},), (self.record(target_answer="UNKNOWN"),)):
            with self.subTest(records=records):
                with self.assertRaisesRegex(DatasetError, "No valid examples") as caught:
                    load_dataset(self.write(*records))
                self.assertEqual(len(caught.exception.excluded), len(records))

    def test_unreadable_file_and_encoding_failure(self):
        for path in (self.root / "missing.jsonl", self.root):
            with self.subTest(path=path):
                with self.assertRaisesRegex(DatasetError, "Cannot read dataset"):
                    load_dataset(path)
        self.source.write_bytes(b'{}\n\xff\n')
        with self.assertRaisesRegex(DatasetError, "UTF-8 at line 2") as caught:
            load_dataset(self.source)
        self.assertEqual(caught.exception.excluded[0]["line"], 1)


class SplitTests(unittest.TestCase):
    @staticmethod
    def dataset(size, *, split_counts=None, excluded=None):
        examples = [
            {"id": index, "input": f"Input {index}", "target_answer": str(index)}
            for index in range(size)
        ]
        if split_counts:
            start = 0
            for name, count in split_counts.items():
                for example in examples[start:start + count]:
                    example["split"] = name
                start += count
        return DatasetResult(examples, excluded or [])

    def test_automatic_splits_are_reproducible_and_preserve_order(self):
        source = self.dataset(700, excluded=[{"line": 1, "id": None, "reasons": ["invalid"]}])
        source.content_sha256 = "abc123"
        first = assign_splits(source, SplitRatios(), 42)
        second = assign_splits(source, SplitRatios(), 42)
        different_seed = assign_splits(source, SplitRatios(), 43)

        self.assertEqual(Counter(row["split"] for row in first.examples),
                         {"train": 490, "validation": 105, "test": 105})
        self.assertEqual([row["id"] for row in first.examples], list(range(700)))
        self.assertEqual(first.content_sha256, source.content_sha256)
        self.assertEqual(first, second)
        self.assertNotEqual([row["split"] for row in first.examples],
                            [row["split"] for row in different_seed.examples])
        self.assertTrue(all("split" not in row for row in source.examples))
        self.assertEqual(first.excluded, source.excluded)

    def test_largest_remainder_rounding(self):
        result = assign_splits(self.dataset(701), SplitRatios(), 42)
        self.assertEqual(Counter(row["split"] for row in result.examples),
                         {"train": 491, "validation": 105, "test": 105})

    def test_user_splits_are_preserved(self):
        counts = {"train": 500, "validation": 100, "test": 100}
        source = self.dataset(700, split_counts=counts)
        result = assign_splits(source, SplitRatios(), 42)
        self.assertEqual(result.examples, source.examples)
        self.assertIsNot(result.examples, source.examples)

    def test_minimum_total_and_split_sizes(self):
        cases = [
            (self.dataset(699), SplitRatios(), ("at least 700",)),
            (self.dataset(700), SplitRatios(0.8, 0.1, 0.1),
             ("validation split has 70", "test split has 70")),
            (self.dataset(700, split_counts={"train": 550, "validation": 75, "test": 75}),
             SplitRatios(), ("validation split has 75", "test split has 75")),
            (self.dataset(700, split_counts={"train": 700}),
             SplitRatios(), ("validation split has 0", "test split has 0")),
        ]
        for dataset, ratios, messages in cases:
            with self.subTest(messages=messages):
                with self.assertRaises(DatasetError) as caught:
                    assign_splits(dataset, ratios, 42)
                for message in messages:
                    self.assertIn(message, str(caught.exception))

    def test_split_error_preserves_exclusions(self):
        excluded = [{"line": 2, "id": 2, "reasons": ["invalid"]}]
        with self.assertRaises(DatasetError) as caught:
            assign_splits(self.dataset(10, excluded=excluded), SplitRatios(), 42)
        self.assertEqual(caught.exception.excluded, excluded)

    def test_small_smoke_split_can_defer_probe_minimums(self):
        result = assign_splits(
            self.dataset(10), SplitRatios(), 42, enforce_minimums=False
        )
        self.assertEqual(
            Counter(row["split"] for row in result.examples),
            {"train": 7, "validation": 2, "test": 1},
        )


if __name__ == "__main__":
    unittest.main()
