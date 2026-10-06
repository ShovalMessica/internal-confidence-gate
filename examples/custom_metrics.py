"""Example task metrics for name-correction predictions.

Dataset records use metadata.example_type = "corrupted" for examples that
require a correction. Valid correction predictions are participant labels A-J.
"""


_PARTICIPANT_LABELS = set("abcdefghij")


def compute_metrics(records):
    corrupted = [
        record
        for record in records
        if record["metadata"].get("example_type") == "corrupted"
    ]
    proposals = [
        record
        for record in records
        if record["normalized_prediction"] in _PARTICIPANT_LABELS
    ]
    correct_corrections = sum(
        record["is_correct"] is True
        and record["normalized_prediction"] in _PARTICIPANT_LABELS
        for record in corrupted
    )
    wrong_corrections = sum(record["is_correct"] is False for record in proposals)

    return {
        "correction_recall": {
            "numerator": correct_corrections,
            "denominator": len(corrupted),
        },
        "correction_fdr": {
            "numerator": wrong_corrections,
            "denominator": len(proposals),
        },
    }
