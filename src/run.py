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
from src.run_store import (
    RegisteredRun,
    RunStoreError,
    build_run_identity,
    register_run,
)


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


def _split_counts(dataset: DatasetResult) -> dict[str, int]:
    counts = Counter(example["split"] for example in dataset.examples)
    return {name: counts[name] for name in ("train", "validation", "test")}


def _preparation_summary(prepared: PreparedRun) -> dict:
    return {
        "valid_examples": len(prepared.dataset.examples),
        "excluded_examples": len(prepared.dataset.excluded),
        "split_source": prepared.split_source,
        "split_sizes": _split_counts(prepared.dataset),
    }


def _print_summary(
    prepared: PreparedRun, registered: RegisteredRun, stream: TextIO
) -> None:
    config, dataset = prepared.config, prepared.dataset
    counts = _split_counts(dataset)
    print("Configuration valid.", file=stream)
    print(f"Model: {config.model_name_or_path}", file=stream)
    print(f"Reasoning mode: {config.reasoning_mode}", file=stream)
    print(f"Dataset: {config.dataset_path}", file=stream)
    print(f"Run ID: {registered.directory.name}", file=stream)
    print(f"Run directory: {registered.directory}", file=stream)
    status = "created" if registered.created else "reused"
    print(f"Run record: {status}", file=stream)
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
        identity = build_run_identity(
            prepared.config,
            prepared.dataset.content_sha256,
            prepared.split_source,
        )
        registered = register_run(
            identity,
            prepared.config,
            args.config,
            _preparation_summary(prepared),
        )
    except ConfigurationError as error:
        print(error, file=sys.stderr)
        return 1
    except DatasetError as error:
        print(error, file=sys.stderr)
        _print_exclusions(error.excluded, sys.stderr)
        return 1
    except RunStoreError as error:
        print(f"Run storage error: {error}", file=sys.stderr)
        return 1

    _print_summary(prepared, registered, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
