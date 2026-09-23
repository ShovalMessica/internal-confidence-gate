"""Command-line coordinator for the implemented toolkit stages."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal, Sequence, TextIO

from src.config import ConfigurationError, TaskConfig, load_config
from src.dataset import DatasetError, DatasetResult, assign_splits, load_dataset


@dataclass(frozen=True)
class PreparedRun:
    config: TaskConfig
    dataset: DatasetResult
    split_source: Literal["dataset", "automatic"]


def prepare_run(config_path: str | Path) -> PreparedRun:
    """Load configuration and return validated, split-assigned examples."""
    config = load_config(config_path)
    dataset = load_dataset(
        config.dataset_path, allow_abstention=config.allow_abstention
    )
    split_source = "dataset" if "split" in dataset.examples[0] else "automatic"
    dataset = assign_splits(dataset, config.split_ratios, config.split_seed)
    return PreparedRun(config, dataset, split_source)


def _print_exclusions(excluded: list[dict], stream: TextIO) -> None:
    if not excluded:
        return
    print("Excluded records:", file=stream)
    for record in excluded:
        record_id = record["id"] if record["id"] is not None else "unavailable"
        reasons = "; ".join(record["reasons"])
        print(f"  Line {record['line']}, ID {record_id}: {reasons}", file=stream)


def _print_summary(prepared: PreparedRun, stream: TextIO) -> None:
    config, dataset = prepared.config, prepared.dataset
    counts = Counter(example["split"] for example in dataset.examples)
    print("Configuration valid.", file=stream)
    print(f"Model: {config.model_name_or_path}", file=stream)
    print(f"Reasoning mode: {config.reasoning_mode}", file=stream)
    print(f"Dataset: {config.dataset_path}", file=stream)
    print(f"Valid examples: {len(dataset.examples)}", file=stream)
    print(f"Excluded examples: {len(dataset.excluded)}", file=stream)
    source = "provided by dataset" if prepared.split_source == "dataset" else "automatic"
    print(f"Split source: {source}", file=stream)
    for name in ("train", "validation", "test"):
        print(f"  {name}: {counts[name]}", file=stream)
    _print_exclusions(dataset.excluded, stream)
    print("Validation complete. No model was run.", file=stream)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an Internal Confidence Gate task configuration and dataset."
    )
    parser.add_argument("config", type=Path, help="Path to the task YAML configuration.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prepared = prepare_run(args.config)
    except ConfigurationError as error:
        print(error, file=sys.stderr)
        return 1
    except DatasetError as error:
        print(error, file=sys.stderr)
        _print_exclusions(error.excluded, sys.stderr)
        return 1

    _print_summary(prepared, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
