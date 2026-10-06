"""Checks for the executable synthetic NER example."""

from collections import defaultdict
import unittest

from examples.ner.prepare import (
    DEV_SEED,
    FULL_SEED,
    build_records,
    content_key,
    validate_partition_isolation,
)


class NerExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.development = build_records(
            20,
            seed=DEV_SEED,
            family_prefix="dev",
            id_offset=1_000_000,
        )
        development_keys = {content_key(record["input"]) for record in cls.development}
        cls.full = build_records(
            700,
            seed=FULL_SEED,
            family_prefix="full",
            forbidden_content=development_keys,
        )

    def test_partitions_are_disjoint_and_families_do_not_cross_splits(self):
        development_keys = {content_key(record["input"]) for record in self.development}
        validate_partition_isolation(self.full, external_content=development_keys)
        all_keys = development_keys | {content_key(record["input"]) for record in self.full}
        self.assertEqual(len(all_keys), len(self.development) + len(self.full))

        families = defaultdict(list)
        for record in self.development + self.full:
            families[record["metadata"]["case_family"]].append(record)
        for records in families.values():
            self.assertEqual(len(records), 2)
            self.assertEqual({record["split"] for record in records}, {records[0]["split"]})
            self.assertEqual(
                {record["metadata"]["example_type"] for record in records},
                {"clean", "corrupted"},
            )

    def test_generation_is_deterministic_and_spans_select_the_mentions(self):
        repeated = build_records(
            20,
            seed=DEV_SEED,
            family_prefix="dev",
            id_offset=1_000_000,
        )
        self.assertEqual(repeated, self.development)

        for record in self.development:
            span = record["semantic_spans"]["span_1"]
            mention = record["input"][span["start_char"] : span["end_char"]]
            self.assertTrue(mention)
            self.assertEqual(record["input"].rindex(mention), span["start_char"])
            self.assertGreater(span["start_char"], record["input"].index("<MEETING_TRANSCRIPT>"))
            self.assertIn("<Speaker ", record["input"])
            if record["metadata"]["example_type"] == "clean":
                self.assertEqual(record["target_answer"], "NONE")
            else:
                self.assertNotEqual(record["target_answer"], "NONE")


if __name__ == "__main__":
    unittest.main()
