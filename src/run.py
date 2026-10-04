"""Command-line coordinator for preparation and generation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal, Sequence, TextIO

from src.config import ConfigurationError, TaskConfig, load_config
from src.dataset import DatasetError, DatasetResult, assign_splits, load_dataset
from src.evaluation import (
    AnswerMatcher,
    EvaluationError,
    build_evaluation_identity,
    evaluate_answers,
    load_answer_matcher,
)
from src.generation import GenerationError, generation_units
from src.model import ModelLoadError, describe_model, load_model
from src.run_store import (
    RegisteredRun,
    RunStoreError,
    append_generation_records,
    build_run_identity,
    complete_generation,
    complete_evaluation,
    evaluation_artifact_path,
    finalize_generation_records,
    load_evaluation_records,
    load_generation_records,
    load_run_record,
    register_run,
    reset_generation,
    reset_partial_direct_batches,
    save_model_metadata,
    validate_completed_generation,
    validate_completed_evaluation,
    write_evaluation_records,
)


@dataclass(frozen=True)
class PreparedRun:
    config: TaskConfig
    dataset: DatasetResult
    split_source: Literal["dataset", "automatic"]
    answer_matcher: AnswerMatcher


def prepare_run(config_path: str | Path) -> PreparedRun:
    """Load configuration and return validated, split-assigned examples."""
    config = load_config(config_path)
    matcher = load_answer_matcher(config.answer_matcher_path)
    dataset = load_dataset(
        config.dataset_path, allow_abstention=config.allow_abstention
    )
    split_source = "dataset" if "split" in dataset.examples[0] else "automatic"
    dataset = assign_splits(dataset, config.split_ratios, config.split_seed)
    return PreparedRun(config, dataset, split_source, matcher)


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


def _print_preparation(
    prepared: PreparedRun, registered: RegisteredRun, stream: TextIO
) -> None:
    config, dataset = prepared.config, prepared.dataset
    print("Configuration valid.", file=stream)
    print(f"Model: {config.model_name_or_path}", file=stream)
    print(f"Reasoning mode: {config.reasoning_mode}", file=stream)
    matcher = prepared.answer_matcher
    matcher_name = str(matcher.source_path) if matcher.source_path else "built-in"
    print(f"Answer matcher: {matcher_name}", file=stream)
    print(f"Dataset: {config.dataset_path}", file=stream)
    print(f"Run ID: {registered.directory.name}", file=stream)
    print(f"Run directory: {registered.directory}", file=stream)
    print(
        f"Run record: {'created' if registered.created else 'reused'}", file=stream
    )
    print(f"Valid examples: {len(dataset.examples)}", file=stream)
    print(f"Excluded examples: {len(dataset.excluded)}", file=stream)
    source = "provided by dataset" if prepared.split_source == "dataset" else "automatic"
    print(f"Split source: {source}", file=stream)
    for name, count in _split_counts(dataset).items():
        print(f"  {name}: {count}", file=stream)
    _print_exclusions(dataset.excluded, stream)


def _generation_settings(config: TaskConfig) -> dict:
    settings = {
        "mode": config.reasoning_mode,
        "answer_max_new_tokens": config.answer_max_new_tokens,
        "allow_abstention": config.allow_abstention,
        "generation_seed": config.generation_seed,
    }
    if config.reasoning_mode == "reasoning":
        settings["reasoning_max_new_tokens"] = config.reasoning_max_new_tokens
    else:
        settings["direct_batch_size"] = config.direct_batch_size
    return settings


def _git_state() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": dirty}


def _provenance(config_path: Path, force_recompute: bool) -> dict:
    invocation = [sys.executable, "-m", "src.run", str(config_path)]
    if force_recompute:
        invocation.append("--force-recompute")
    return {"invocation": invocation, "git": _git_state()}


def _progress(
    done: int,
    total: int,
    records: dict[int, dict],
    started: float,
    starting_done: int,
) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = (done - starting_done) / elapsed
    remaining = total - done
    eta = f"{remaining / rate:.1f}s" if rate else "unknown"
    successes = sum(record["status"] == "success" for record in records.values())
    failures = len(records) - successes
    percent = 100 * done / total
    print(
        f"Generation: {done}/{total} ({percent:.1f}%) | "
        f"elapsed {elapsed:.1f}s | {rate:.2f} examples/s | ETA {eta} | "
        f"success {successes} | failed {failures}",
        flush=True,
    )


def _generate(
    prepared: PreparedRun,
    registered: RegisteredRun,
    config_path: Path,
    force_recompute: bool,
) -> None:
    directory = registered.directory
    examples = prepared.dataset.examples
    record = load_run_record(directory)

    pinned_revision = None
    if force_recompute:
        pinned_revision = reset_generation(directory)
        record = load_run_record(directory)
    elif validate_completed_generation(directory, record):
        records, _ = load_generation_records(directory, examples)
        if set(records) != {example["id"] for example in examples}:
            raise RunStoreError("Completed generation does not contain every example.")
        print("Generation artifact: reused (model not loaded).")
        return

    records, recovered = load_generation_records(directory, examples)
    if recovered:
        print("Recovered a truncated final generation record.")
    if prepared.config.reasoning_mode == "direct":
        records, removed = reset_partial_direct_batches(
            directory, records, examples, prepared.config.direct_batch_size
        )
        if removed:
            print(f"Reset {removed} record(s) from an incomplete direct batch.")

    stored_model = record.get("model")
    if records and not isinstance(stored_model, dict):
        raise RunStoreError("Generation records exist without recorded model metadata.")
    expected_ids = {example["id"] for example in examples}
    if set(records) == expected_ids:
        artifact_hash, counts = finalize_generation_records(directory, records, examples)
        complete_generation(directory, artifact_hash, counts)
        print("Recovered and finalized the complete generation artifact.")
        return
    if pinned_revision is None and isinstance(stored_model, dict):
        pinned_revision = stored_model.get("resolved_revision")

    loaded = load_model(prepared.config, pinned_revision=pinned_revision)
    metadata = {
        "identifier": prepared.config.model_name_or_path,
        "requested_revision": prepared.config.model_revision,
        **describe_model(loaded),
    }
    save_model_metadata(
        directory,
        metadata,
        _provenance(config_path, force_recompute),
        _generation_settings(prepared.config),
    )

    total = len(examples)
    started = time.monotonic()
    starting_done = len(records)
    if records:
        print(f"Resuming generation with {len(records)}/{total} records complete.")
    for unit in generation_units(
        loaded, examples, prepared.config, completed_ids=records
    ):
        append_generation_records(directory, list(unit.records))
        records.update((item["id"], item) for item in unit.records)
        _progress(len(records), total, records, started, starting_done)

    artifact_hash, counts = finalize_generation_records(directory, records, examples)
    complete_generation(directory, artifact_hash, counts)
    print(
        f"Generation complete: {counts['successful']} succeeded, "
        f"{counts['failed']} failed."
    )


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _print_evaluation(summary: dict, *, reused: bool) -> None:
    print(f"Evaluation artifact: {'reused' if reused else 'created'}")
    columns = (
        ("Correct prediction", "correct", "correct_prediction_rate"),
        ("Wrong prediction", "incorrect", "wrong_prediction_rate"),
        ("Missed prediction", "abstained", "missed_prediction_rate"),
        ("Invalid output", "invalid", "invalid_output_rate"),
        ("Token-limit output", "answer_token_limit", "token_limit_rate"),
    )
    print("Model Behavior:")
    print(f"  {'Scope':<12}{'Outcome':<22}{'Count':>8}{'Rate':>10}")
    scopes = [("overall", summary["overall"]), *summary["by_split"].items()]
    for scope, metrics in scopes:
        for index, (label, count_key, rate_key) in enumerate(columns):
            scope_label = scope if index == 0 else ""
            print(
                f"  {scope_label:<12}{label:<22}{metrics[count_key]:>8}"
                f"{_format_rate(metrics[rate_key]):>10}"
            )


def _print_shortages(shortages: list[dict], stream: TextIO) -> None:
    print("Probe-readiness requirements are not met:", file=stream)
    for shortage in shortages:
        print(
            f"  {shortage['split']} needs {shortage['required']} "
            f"{shortage['outcome']} predictions; found {shortage['actual']} "
            f"(missing {shortage['missing']}).",
            file=stream,
        )


def _evaluate(prepared: PreparedRun, registered: RegisteredRun) -> dict:
    directory = registered.directory
    examples = prepared.dataset.examples
    run_record = load_run_record(directory)
    if not validate_completed_generation(directory, run_record):
        raise RunStoreError("Evaluation requires completed generation.")
    generation_hash = run_record["generation"]["artifact_sha256"]
    identity = build_evaluation_identity(
        generation_hash, prepared.answer_matcher
    )

    if validate_completed_evaluation(directory, run_record, identity):
        load_evaluation_records(directory, identity.evaluation_id, examples)
        summary = run_record["evaluations"][identity.evaluation_id]["summary"]
        print(f"Evaluation ID: {identity.evaluation_id}")
        _print_evaluation(summary, reused=True)
        return summary

    generations, _ = load_generation_records(directory, examples)
    result = evaluate_answers(
        examples,
        generations,
        allow_abstention=prepared.config.allow_abstention,
        matcher=prepared.answer_matcher,
    )
    path = evaluation_artifact_path(directory, identity.evaluation_id)
    if path.exists():
        existing = load_evaluation_records(directory, identity.evaluation_id, examples)
        ordered = [existing[example["id"]] for example in examples]
        if ordered != list(result.records):
            raise RunStoreError(
                "Unregistered evaluation artifact conflicts with computed results."
            )
    artifact_hash = write_evaluation_records(
        directory, identity.evaluation_id, list(result.records)
    )
    complete_evaluation(
        directory,
        identity,
        prepared.answer_matcher,
        artifact_hash,
        result.summary,
    )
    print(f"Evaluation ID: {identity.evaluation_id}")
    _print_evaluation(result.summary, reused=False)
    return result.summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and run an Internal Confidence Gate task."
    )
    parser.add_argument("config", type=Path, help="Path to the task YAML configuration.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and register the run without loading a model.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Discard saved generations and rerun them with the pinned model revision.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prepare_only and args.force_recompute:
        print("--force-recompute cannot be used with --prepare-only.", file=sys.stderr)
        return 2
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
        _print_preparation(prepared, registered, sys.stdout)
        if args.prepare_only:
            print("Preparation complete. No model was run.")
        else:
            _generate(prepared, registered, args.config.resolve(), args.force_recompute)
            summary = _evaluate(prepared, registered)
            if not summary["probe_ready"]:
                _print_shortages(summary["shortages"], sys.stderr)
                return 1
    except ConfigurationError as error:
        print(error, file=sys.stderr)
        return 1
    except DatasetError as error:
        print(error, file=sys.stderr)
        _print_exclusions(error.excluded, sys.stderr)
        return 1
    except ModelLoadError as error:
        print(f"Model loading error: {error}", file=sys.stderr)
        return 1
    except GenerationError as error:
        print(f"Generation error: {error}", file=sys.stderr)
        return 1
    except EvaluationError as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 1
    except RunStoreError as error:
        print(f"Run storage error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
