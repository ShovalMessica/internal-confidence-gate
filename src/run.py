"""Command-line coordinator for the confidence-gate pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal, Sequence, TextIO

from src.activation import (
    ActivationError,
    ReplayPlan,
    build_replay_plan,
    capture_hidden_states,
    map_semantic_spans,
)
from src.activation_store import (
    ActivationStoreError,
    StoredActivation,
    append_activation_record,
    build_activation_context,
    build_activation_identity,
    build_capture_request_fingerprint,
    finalize_activation_file,
    load_activation_records,
    reuse_activation_records,
)
from src.config import ConfigurationError, TaskConfig, load_config
from src.dataset import DatasetError, DatasetResult, assign_splits, load_dataset
from src.evaluation import (
    AnswerMatcher,
    EvaluationError,
    build_evaluation_identity,
    evaluate_answers,
    load_answer_matcher,
    probe_shortages,
)
from src.generation import GenerationError, generation_units
from src.generation_cache import (
    GenerationCacheError,
    build_generation_context,
    generation_record_sha256,
    load_context_model,
    save_context_model,
)
from src.model import (
    LoadedModel,
    ModelLoadError,
    describe_model,
    load_model,
    load_tokenizer,
)
from src.probe import (
    ProbeError,
    ProbeIdentity,
    SelectionIdentity,
    TestEvaluationIdentity,
    build_probe_identity,
    build_selection_identity,
    build_test_evaluation_identity,
    evaluate_frozen_test,
    select_probe,
    train_probes,
)
from src.reporting import (
    ReportingError,
    build_report_identity,
    create_report,
)
from src.run_store import (
    RegisteredRun,
    RunStoreError,
    append_generation_records,
    build_run_identity,
    complete_generation,
    complete_evaluation,
    complete_activation_capture,
    complete_probe_selection,
    complete_probe_training,
    complete_test_evaluation,
    complete_report,
    evaluation_artifact_path,
    finalize_generation_records,
    load_evaluation_records,
    load_generation_records,
    load_run_record,
    register_run,
    reset_generation,
    reuse_cached_generation_records,
    save_model_metadata,
    validate_completed_generation,
    validate_completed_evaluation,
    validate_completed_activation_capture,
    validate_completed_probe_selection,
    validate_completed_probe_training,
    validate_completed_test_evaluation,
    validate_completed_report,
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
    if config.system_prompt_path is not None:
        print(f"System prompt: {config.system_prompt_path}", file=stream)
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
    if config.system_prompt_sha256 is not None:
        settings["system_prompt_sha256"] = config.system_prompt_sha256
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


def _model_metadata(config: TaskConfig, loaded: LoadedModel) -> dict:
    return {
        "identifier": config.model_name_or_path,
        "requested_revision": config.model_revision,
        **describe_model(loaded),
    }


def _generate(
    prepared: PreparedRun,
    registered: RegisteredRun,
    config_path: Path,
    force_recompute: bool,
) -> LoadedModel | None:
    directory = registered.directory
    examples = prepared.dataset.examples
    examples_by_id = {example["id"]: example for example in examples}
    context = build_generation_context(prepared.config)
    record = load_run_record(directory)

    pinned_revision = None
    if force_recompute:
        pinned_revision = reset_generation(directory)
        record = load_run_record(directory)
    elif validate_completed_generation(directory, record, context):
        records, _ = load_generation_records(directory, context, examples)
        if set(records) != {example["id"] for example in examples}:
            raise RunStoreError("Completed generation does not contain every example.")
        print("Generation artifact: reused (model not loaded).")
        return None

    records, recovered = load_generation_records(directory, context, examples)
    if recovered:
        print("Recovered a truncated final generation record.")

    stored_model = record.get("model")
    cached_model = load_context_model(context)
    if isinstance(stored_model, dict) and cached_model not in (None, stored_model):
        raise RunStoreError("Run model metadata conflicts with the generation cache.")
    if cached_model is not None and stored_model is None:
        save_model_metadata(
            directory,
            cached_model,
            _provenance(config_path, force_recompute),
            _generation_settings(prepared.config),
        )
        stored_model = cached_model
    if records and not isinstance(stored_model, dict):
        raise RunStoreError("Generation records exist without recorded model metadata.")

    if not force_recompute and cached_model is not None:
        reused = reuse_cached_generation_records(
            directory, context, examples, set(records)
        )
        records.update(reused)
        if reused:
            print(f"Generation cache: {len(reused)} reused.")

    expected_ids = {example["id"] for example in examples}
    if set(records) == expected_ids:
        artifact_hash, counts = finalize_generation_records(
            directory, context, records, examples
        )
        complete_generation(directory, context, artifact_hash, counts)
        print("Generation complete from cached records; model not loaded.")
        return None
    if pinned_revision is None and isinstance(stored_model, dict):
        pinned_revision = stored_model.get("resolved_revision")

    loaded = load_model(prepared.config, pinned_revision=pinned_revision)
    metadata = _model_metadata(prepared.config, loaded)
    save_context_model(context, metadata)
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
        append_generation_records(
            directory, context, examples_by_id, list(unit.records)
        )
        records.update((item["id"], item) for item in unit.records)
        _progress(len(records), total, records, started, starting_done)

    artifact_hash, counts = finalize_generation_records(
        directory, context, records, examples
    )
    complete_generation(directory, context, artifact_hash, counts)
    print(
        f"Generation complete: {starting_done} reused, "
        f"{total - starting_done} generated; {counts['successful']} succeeded, "
        f"{counts['failed']} failed."
    )
    return loaded


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
    context = build_generation_context(prepared.config)
    run_record = load_run_record(directory)
    if not validate_completed_generation(directory, run_record, context):
        raise RunStoreError("Evaluation requires completed generation.")
    generation_hash = run_record["generation"]["artifact_sha256"]
    identity = build_evaluation_identity(
        generation_hash, prepared.answer_matcher
    )

    if validate_completed_evaluation(directory, run_record, identity):
        records_by_id = load_evaluation_records(
            directory, identity.evaluation_id, examples
        )
        summary = dict(run_record["evaluations"][identity.evaluation_id]["summary"])
        shortages = probe_shortages(
            list(records_by_id.values()), prepared.config.probe_excluded_answers
        )
        summary["probe_ready"] = not shortages
        summary["shortages"] = list(shortages)
        print(f"Evaluation ID: {identity.evaluation_id}")
        _print_evaluation(summary, reused=True)
        return summary

    generations, _ = load_generation_records(directory, context, examples)
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
    summary = dict(result.summary)
    shortages = probe_shortages(
        list(result.records), prepared.config.probe_excluded_answers
    )
    summary["probe_ready"] = not shortages
    summary["shortages"] = list(shortages)
    _print_evaluation(summary, reused=False)
    return summary


def _activation_progress(done: int, total: int, started: float, starting_done: int) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = (done - starting_done) / elapsed
    remaining = total - done
    eta = f"{remaining / rate:.1f}s" if rate else "unknown"
    print(
        f"Activation capture: {done}/{total} ({100 * done / total:.1f}%) | "
        f"elapsed {elapsed:.1f}s | {rate:.2f} examples/s | ETA {eta}",
        flush=True,
    )


def _activation_summary(
    records: dict[int, StoredActivation],
    expected: dict[int, tuple[dict, str, ReplayPlan]],
) -> dict:
    state_labels = None
    hidden_size = None
    position_names = None
    max_difference = 0.0
    for example_id, cached in records.items():
        metadata = cached.metadata
        labels = json.loads(metadata["state_labels"])
        size = int(metadata["hidden_size"])
        names = list(expected[example_id][2].positions)
        if state_labels is None:
            state_labels, hidden_size, position_names = labels, size, names
        elif (
            labels != state_labels
            or size != hidden_size
            or names != position_names
        ):
            raise ActivationStoreError(
                "Stored activations use inconsistent positions or hidden-state dimensions."
            )
        max_difference = max(
            max_difference, float(metadata["max_logprob_difference"])
        )
    return {
        "examples": len(records),
        "positions": position_names,
        "state_labels": state_labels,
        "hidden_size": hidden_size,
        "storage_dtype": "float16",
        "max_logprob_difference": max_difference,
    }


def _capture_activations(
    prepared: PreparedRun,
    registered: RegisteredRun,
    loaded: LoadedModel | None,
    force_recompute: bool,
) -> None:
    directory = registered.directory
    examples = prepared.dataset.examples
    generation_context = build_generation_context(prepared.config)
    run_record = load_run_record(directory)
    generation = run_record.get("generation")
    model_metadata = run_record.get("model")
    if not isinstance(generation, dict) or not isinstance(model_metadata, dict):
        raise RunStoreError("Activation capture requires generation model provenance.")

    evaluation_identity = build_evaluation_identity(
        generation["artifact_sha256"], prepared.answer_matcher
    )
    evaluations = load_evaluation_records(
        directory, evaluation_identity.evaluation_id, examples
    )
    excluded_answers = set(prepared.config.probe_excluded_answers)
    eligible = [
        example
        for example in examples
        if evaluations[example["id"]]["outcome"] in ("correct", "incorrect")
        and evaluations[example["id"]].get("normalized_answer")
        not in excluded_answers
    ]
    if not eligible:
        raise ActivationError("No correct or incorrect predictions are available to capture.")

    context = build_activation_context(
        prepared.config, generation_context, model_metadata
    )
    request_fingerprint = build_capture_request_fingerprint(eligible)
    identity = build_activation_identity(
        context,
        generation["artifact_sha256"],
        request_fingerprint,
    )

    if validate_completed_activation_capture(directory, run_record, identity):
        print(f"Activation capture ID: {identity.capture_id}")
        print("Activation artifact: reused (model not loaded).")
        return

    generations, _ = load_generation_records(directory, generation_context, examples)
    has_semantic_spans = bool(eligible[0].get("semantic_spans", {}))
    tokenizer = None
    if has_semantic_spans:
        tokenizer = (
            loaded.tokenizer
            if loaded is not None
            else load_tokenizer(
                prepared.config,
                pinned_revision=model_metadata.get("resolved_revision"),
            )
        )
    expected = {}
    for example in eligible:
        example_id = example["id"]
        record = generations[example_id]
        semantic_positions = (
            map_semantic_spans(
                tokenizer,
                example,
                record,
                prepared.config.system_prompt,
            )
            if tokenizer is not None
            else None
        )
        plan = build_replay_plan(record, semantic_positions)
        expected[example_id] = (
            example,
            generation_record_sha256(record),
            plan,
        )

    records, recovered = load_activation_records(
        directory, identity, context, expected
    )
    if recovered:
        print("Recovered an incomplete activation write.")

    if not force_recompute:
        reused = reuse_activation_records(
            prepared.config.output_dir,
            directory,
            identity,
            context,
            expected,
            set(records),
        )
        records.update(reused)
        if reused:
            print(f"Prior-run activations: {len(reused)} reused.")

    if set(records) == set(expected):
        artifact_hash = finalize_activation_file(
            directory, identity, context, expected, records
        )
        complete_activation_capture(
            directory,
            identity,
            artifact_hash,
            _activation_summary(records, expected),
        )
        print(f"Activation capture ID: {identity.capture_id}")
        print("Activation capture complete from cached records; model not loaded.")
        return

    if loaded is None:
        pinned_revision = model_metadata.get("resolved_revision")
        loaded = load_model(
            prepared.config,
            pinned_revision=pinned_revision,
            tokenizer=tokenizer,
        )
        if _model_metadata(prepared.config, loaded) != model_metadata:
            raise RunStoreError(
                "Loaded model metadata differs from the generation model."
            )

    total = len(expected)
    starting_done = len(records)
    started = time.monotonic()
    if records:
        print(f"Resuming activation capture with {len(records)}/{total} complete.")
    for example_id, (example, generation_sha256, plan) in expected.items():
        if example_id in records:
            continue
        result = capture_hidden_states(loaded, plan)
        stored = append_activation_record(
            directory,
            identity,
            context,
            example,
            generation_sha256,
            plan,
            result,
        )
        records[example_id] = stored
        new_count = len(records) - starting_done
        progress_interval = max(1, total // 100)
        if new_count == 1 or len(records) == total or new_count % progress_interval == 0:
            _activation_progress(len(records), total, started, starting_done)

    artifact_hash = finalize_activation_file(
        directory, identity, context, expected, records
    )
    complete_activation_capture(
        directory,
        identity,
        artifact_hash,
        _activation_summary(records, expected),
    )
    print(f"Activation capture ID: {identity.capture_id}")
    print(
        f"Activation capture complete: {starting_done} reused, "
        f"{total - starting_done} captured."
    )


def _probe_progress(done: int, total: int, started: float, starting_done: int) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = (done - starting_done) / elapsed
    remaining = total - done
    eta = f"{remaining / rate:.1f}s" if rate else "unknown"
    print(
        f"Probe training: {done}/{total} ({100 * done / total:.1f}%) | "
        f"elapsed {elapsed:.1f}s | {rate:.2f} probes/s | ETA {eta}",
        flush=True,
    )


def _train_probes(
    prepared: PreparedRun, registered: RegisteredRun
) -> tuple[ProbeIdentity, str]:
    directory = registered.directory
    run_record = load_run_record(directory)
    capture = run_record.get("activation_capture")
    generation = run_record.get("generation")
    if not isinstance(capture, dict) or not isinstance(generation, dict):
        raise RunStoreError("Probe training requires completed activations.")

    evaluation_identity = build_evaluation_identity(
        generation["artifact_sha256"], prepared.answer_matcher
    )
    evaluations = run_record.get("evaluations")
    evaluation = (
        evaluations.get(evaluation_identity.evaluation_id)
        if isinstance(evaluations, dict)
        else None
    )
    if not isinstance(evaluation, dict):
        raise RunStoreError("Probe training requires the current evaluation.")

    identity = build_probe_identity(
        capture["artifact_sha256"],
        evaluation["artifact_sha256"],
        prepared.config.probe_seed,
        positions=prepared.config.probe_positions,
        layers=prepared.config.probe_layers,
        excluded_answers=prepared.config.probe_excluded_answers,
        regularization_c=prepared.config.probe_regularization_c,
        class_weight=prepared.config.probe_class_weight,
    )
    if validate_completed_probe_training(directory, run_record, identity):
        print(f"Probe training ID: {identity.probe_id}")
        print("Probe artifact: reused.")
        return identity, run_record["probe_trainings"][identity.probe_id][
            "content_sha256"
        ]

    records = load_evaluation_records(
        directory, evaluation_identity.evaluation_id, prepared.dataset.examples
    )
    result = train_probes(directory, identity, records, _probe_progress)
    complete_probe_training(
        directory,
        identity,
        result.content_sha256,
        result.summary,
    )
    print(f"Probe training ID: {identity.probe_id}")
    if result.trained_candidates:
        print(
            f"Probe training complete: {result.starting_candidates} resumed, "
            f"{result.trained_candidates} trained."
        )
    else:
        print("Probe training complete from stored candidates.")
    return identity, result.content_sha256


def _print_probe_selection(selection_id: str, summary: dict, reused: bool) -> None:
    balanced_accuracy = (summary["tpr"] + 1 - summary["fpr"]) / 2
    print(f"Probe selection ID: {selection_id}")
    print(f"Probe selection: {'reused' if reused else 'created'}.")
    print(
        "Selected probe: "
        f"{summary['position']} / {summary['state']} | "
        f"threshold {summary['threshold']:.6f} | "
        f"validation TPR {summary['tpr']:.4f} | "
        f"validation FPR {summary['fpr']:.4f} | "
        f"balanced accuracy {balanced_accuracy:.4f} | "
        f"AUROC {summary['auroc']:.4f}"
    )


def _select_probe(
    prepared: PreparedRun,
    registered: RegisteredRun,
    probe_identity: ProbeIdentity,
    probe_sha256: str,
) -> tuple[SelectionIdentity, str]:
    identity = build_selection_identity(
        probe_identity.probe_id,
        probe_sha256,
        prepared.config.target_tpr,
    )
    run_record = load_run_record(registered.directory)
    if validate_completed_probe_selection(
        registered.directory, run_record, probe_identity, identity
    ):
        summary = run_record["probe_selections"][identity.selection_id]["selected"]
        _print_probe_selection(identity.selection_id, summary, True)
        return identity, run_record["probe_selections"][identity.selection_id][
            "content_sha256"
        ]

    result = select_probe(registered.directory, probe_identity, identity)
    complete_probe_selection(
        registered.directory,
        identity,
        result.content_sha256,
        result.summary,
    )
    _print_probe_selection(identity.selection_id, result.summary, not result.created)
    return identity, result.content_sha256


def _print_test_evaluation(test_id: str, summary: dict, reused: bool) -> None:
    print(f"Test evaluation ID: {test_id}")
    print(f"Test evaluation: {'reused' if reused else 'created'}.")
    for label, key in (
        ("Probe", "probe"),
        ("Output probability", "output_probability"),
    ):
        metrics = summary[key]
        balanced_accuracy = (metrics["tpr"] + 1 - metrics["fpr"]) / 2
        print(
            f"{label}: threshold {metrics['threshold']:.6f} | "
            f"TPR {metrics['tpr']:.4f} | FPR {metrics['fpr']:.4f} | "
            f"balanced accuracy {balanced_accuracy:.4f} | "
            f"AUROC {metrics['auroc']:.4f}"
        )


def _evaluate_frozen_test(
    prepared: PreparedRun,
    registered: RegisteredRun,
    probe_identity: ProbeIdentity,
    selection_identity: SelectionIdentity,
    selection_sha256: str,
) -> tuple[TestEvaluationIdentity, str]:
    directory = registered.directory
    run_record = load_run_record(directory)
    generation = run_record.get("generation")
    if not isinstance(generation, dict):
        raise RunStoreError("Frozen test evaluation requires completed generation.")
    identity = build_test_evaluation_identity(
        selection_identity.selection_id,
        selection_sha256,
        generation["artifact_sha256"],
    )
    if validate_completed_test_evaluation(
        directory,
        run_record,
        probe_identity,
        selection_identity,
        identity,
    ):
        summary = run_record["test_evaluations"][identity.test_id]["summary"]
        _print_test_evaluation(identity.test_id, summary, True)
        return identity, run_record["test_evaluations"][identity.test_id][
            "content_sha256"
        ]

    evaluation_identity = build_evaluation_identity(
        generation["artifact_sha256"], prepared.answer_matcher
    )
    evaluations = load_evaluation_records(
        directory, evaluation_identity.evaluation_id, prepared.dataset.examples
    )
    generation_context = build_generation_context(prepared.config)
    generations, _ = load_generation_records(
        directory, generation_context, prepared.dataset.examples
    )
    result = evaluate_frozen_test(
        directory,
        probe_identity,
        selection_identity,
        identity,
        evaluations,
        generations,
    )
    complete_test_evaluation(
        directory,
        identity,
        result.content_sha256,
        result.summary,
    )
    _print_test_evaluation(identity.test_id, result.summary, not result.created)
    return identity, result.content_sha256


def _create_report(
    prepared: PreparedRun,
    registered: RegisteredRun,
    probe_identity: ProbeIdentity,
    selection_identity: SelectionIdentity,
    selection_sha256: str,
    test_identity: TestEvaluationIdentity,
    test_sha256: str,
) -> None:
    identity = build_report_identity(
        selection_identity.selection_id,
        selection_sha256,
        test_identity.test_id,
        test_sha256,
    )
    run_record = load_run_record(registered.directory)
    if validate_completed_report(registered.directory, run_record, identity):
        print(f"Report ID: {identity.report_id}")
        print(f"Report: reused ({registered.directory / 'reports' / identity.report_id}).")
        return

    context = build_generation_context(prepared.config)
    generations, _ = load_generation_records(
        registered.directory, context, prepared.dataset.examples
    )
    result = create_report(
        registered.directory,
        probe_identity,
        selection_identity,
        test_identity,
        identity,
        generations,
    )
    complete_report(
        registered.directory,
        identity,
        result.artifact_sha256,
        result.summary,
        result.files,
    )
    print(f"Report ID: {identity.report_id}")
    print(f"Report: created ({registered.directory / 'reports' / identity.report_id}).")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and run an Internal Confidence Gate task."
    )
    parser.add_argument("config", type=Path, help="Path to the task YAML configuration.")
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and register the run without loading a model.",
    )
    stop.add_argument(
        "--behavior-only",
        action="store_true",
        help="Run generation and answer evaluation, then stop before activations.",
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
            loaded = _generate(
                prepared, registered, args.config.resolve(), args.force_recompute
            )
            summary = _evaluate(prepared, registered)
            if not summary["probe_ready"]:
                _print_shortages(summary["shortages"], sys.stderr)
                return 1
            if args.behavior_only:
                print(
                    "Model Behavior complete. Activation capture and probe stages "
                    "were not run."
                )
            else:
                _capture_activations(
                    prepared, registered, loaded, args.force_recompute
                )
                probe_identity, probe_sha256 = _train_probes(prepared, registered)
                selection_identity, selection_sha256 = _select_probe(
                    prepared, registered, probe_identity, probe_sha256
                )
                test_identity, test_sha256 = _evaluate_frozen_test(
                    prepared,
                    registered,
                    probe_identity,
                    selection_identity,
                    selection_sha256,
                )
                _create_report(
                    prepared,
                    registered,
                    probe_identity,
                    selection_identity,
                    selection_sha256,
                    test_identity,
                    test_sha256,
                )
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
    except GenerationCacheError as error:
        print(f"Generation cache error: {error}", file=sys.stderr)
        return 1
    except EvaluationError as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 1
    except ActivationError as error:
        print(f"Activation capture error: {error}", file=sys.stderr)
        return 1
    except ActivationStoreError as error:
        print(f"Activation storage error: {error}", file=sys.stderr)
        return 1
    except ProbeError as error:
        print(f"Probe error: {error}", file=sys.stderr)
        return 1
    except ReportingError as error:
        print(f"Reporting error: {error}", file=sys.stderr)
        return 1
    except RunStoreError as error:
        print(f"Run storage error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
