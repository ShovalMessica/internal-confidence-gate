"""Create machine-readable metrics and plots from completed probe artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Mapping
import warnings

import numpy as np

from src.probe import (
    ProbeIdentity,
    SelectionIdentity,
    TestEvaluationIdentity,
    probe_file_path,
    validate_selection_group,
    validate_test_evaluation_group,
)
from src.probe_math import answer_probability


REPORT_PROTOCOL_VERSION = 3
REPORT_SCHEMA_VERSION = 1
REPORTS_DIR = "reports"


class ReportingError(ValueError):
    """Saved scores or reporting artifacts are invalid."""


@dataclass(frozen=True)
class ReportIdentity:
    report_id: str
    fingerprint: str
    selection_id: str
    selection_sha256: str
    test_id: str
    test_sha256: str


@dataclass(frozen=True)
class ReportResult:
    artifact_sha256: str
    summary: dict
    files: dict[str, str]
    created: bool


def _encoded(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReportingError(f"Cannot hash report artifact: {path}") from exc
    return digest.hexdigest()


def build_report_identity(
    selection_id: str,
    selection_sha256: str,
    test_id: str,
    test_sha256: str,
) -> ReportIdentity:
    """Identify a report by its frozen validation and test inputs."""
    payload = {
        "protocol_version": REPORT_PROTOCOL_VERSION,
        "selection_id": selection_id,
        "selection_sha256": selection_sha256,
        "test_id": test_id,
        "test_sha256": test_sha256,
    }
    fingerprint = hashlib.sha256(_encoded(payload)).hexdigest()
    return ReportIdentity(
        fingerprint[:12],
        fingerprint,
        selection_id,
        selection_sha256,
        test_id,
        test_sha256,
    )


def report_directory(run_directory: Path, report_id: str) -> Path:
    return run_directory / REPORTS_DIR / report_id


def _balanced_accuracy(tpr: float, fpr: float) -> float:
    return (float(tpr) + 1.0 - float(fpr)) / 2.0


def _layer_number(state: str) -> int:
    """Return the model layer represented by a saved hidden-state label."""
    if state == "embedding":
        return 0
    match = re.fullmatch(r"hidden_state_([1-9]\d*)", state)
    if match is None:
        raise ReportingError(f"Unsupported hidden-state label: {state}")
    return int(match.group(1))


def _roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, ...]:
    try:
        from sklearn.metrics import roc_curve
    except ImportError as exc:
        raise ReportingError(
            "Report generation is unavailable; install requirements.txt."
        ) from exc
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    except ValueError as exc:
        raise ReportingError(f"Cannot calculate a TPR-FPR curve: {exc}") from exc
    return tpr, fpr, thresholds


def _matplotlib():
    try:
        # Some Matplotlib releases import older pyparsing APIs and emit
        # third-party deprecation warnings. Keep CLI output focused on toolkit
        # diagnostics while preserving all runtime warnings from our code.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator, PercentFormatter
    except ImportError as exc:
        raise ReportingError(
            "Plot generation is unavailable; install requirements.txt."
        ) from exc
    return plt, PercentFormatter, MaxNLocator


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return result or "position"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _curve_rows(
    series: str,
    split: str,
    labels: np.ndarray,
    scores: np.ndarray,
) -> list[dict]:
    tpr, fpr, thresholds = _roc(labels, scores)
    return [
        {
            "split": split,
            "series": series,
            "threshold": float(threshold),
            "tpr": float(true_positive),
            "fpr": float(false_positive),
        }
        for true_positive, false_positive, threshold in zip(tpr, fpr, thresholds)
    ]


def _plot_layers(path: Path, position: str, rows: list[dict]) -> None:
    plt, PercentFormatter, MaxNLocator = _matplotlib()
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        [row["layer"] for row in rows],
        [100 * row["balanced_accuracy"] for row in rows],
        marker="o",
        linewidth=2,
    )
    winner = next(row for row in rows if row["position_winner"])
    axis.scatter(
        [winner["layer"]],
        [100 * winner["balanced_accuracy"]],
        color="black",
        s=55,
        zorder=3,
        label=f"Selected: layer {winner['layer']}",
    )
    axis.set_title(f"Validation balanced accuracy by layer: {position}")
    axis.set_xlabel("Layer (0 = token embedding)")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0, 100)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_formatter(PercentFormatter(100))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_curves(
    path: Path,
    rows: list[dict],
    operating_points: Mapping[str, tuple[float, float]],
    title: str,
    highlighted: str,
) -> None:
    plt, PercentFormatter, _ = _matplotlib()
    figure, axis = plt.subplots(figsize=(9, 6))
    series_names = tuple(dict.fromkeys(row["series"] for row in rows))
    for series in series_names:
        values = [row for row in rows if row["series"] == series]
        emphasized = series == highlighted
        axis.step(
            [100 * row["tpr"] for row in values],
            [100 * row["fpr"] for row in values],
            where="post",
            linewidth=3 if emphasized else 1.8,
            linestyle="--" if series == "output_probability" else "-",
            label=series,
        )
        if series in operating_points:
            tpr, fpr = operating_points[series]
            axis.scatter([100 * tpr], [100 * fpr], s=50, zorder=3)
    axis.set_title(title)
    axis.set_xlabel("Correct predictions accepted (TPR)")
    axis.set_ylabel("Incorrect predictions accepted (FPR)")
    axis.xaxis.set_major_formatter(PercentFormatter(100))
    axis.yaxis.set_major_formatter(PercentFormatter(100))
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _manifest(directory: Path, identity: ReportIdentity, summary: dict) -> ReportResult:
    files = {
        path.name: _sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_version": REPORT_PROTOCOL_VERSION,
        "report_id": identity.report_id,
        "fingerprint": identity.fingerprint,
        "selection_id": identity.selection_id,
        "selection_sha256": identity.selection_sha256,
        "test_id": identity.test_id,
        "test_sha256": identity.test_sha256,
        "files": files,
        "summary": summary,
    }
    path = directory / "manifest.json"
    _write_json(path, manifest)
    return ReportResult(_sha256(path), summary, files, True)


def validate_report(
    run_directory: Path,
    identity: ReportIdentity,
    expected_sha256: str | None = None,
) -> ReportResult:
    """Validate a completed report directory and every listed artifact."""
    directory = report_directory(run_directory, identity.report_id)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportingError(
            f"Cannot read valid report manifest: {manifest_path}"
        ) from exc
    expected = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_version": REPORT_PROTOCOL_VERSION,
        "report_id": identity.report_id,
        "fingerprint": identity.fingerprint,
        "selection_id": identity.selection_id,
        "selection_sha256": identity.selection_sha256,
        "test_id": identity.test_id,
        "test_sha256": identity.test_sha256,
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(name) != value for name, value in expected.items()
    ):
        raise ReportingError(f"Report manifest is incompatible: {manifest_path}")
    files = manifest.get("files")
    summary = manifest.get("summary")
    if not isinstance(files, dict) or not files or not isinstance(summary, dict):
        raise ReportingError(f"Report manifest is incomplete: {manifest_path}")
    actual_names = {
        path.name for path in directory.iterdir() if path.is_file()
    }
    if actual_names != set(files) | {"manifest.json"}:
        raise ReportingError(f"Report files differ from their manifest: {directory}")
    for name, digest in files.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ReportingError(
                f"Report manifest has invalid file entries: {manifest_path}"
            )
        if _sha256(directory / name) != digest:
            raise ReportingError(f"Report artifact has changed: {directory / name}")
    manifest_hash = _sha256(manifest_path)
    if expected_sha256 is not None and manifest_hash != expected_sha256:
        raise ReportingError(f"Report manifest has changed: {manifest_path}")
    return ReportResult(manifest_hash, summary, files, False)


def create_report(
    run_directory: Path,
    probe_identity: ProbeIdentity,
    selection_identity: SelectionIdentity,
    test_identity: TestEvaluationIdentity,
    identity: ReportIdentity,
    generations: Mapping[int, Mapping[str, object]],
) -> ReportResult:
    """Create validation comparisons and frozen test reports from saved scores."""
    validate_selection_group(
        run_directory,
        probe_identity,
        selection_identity,
        identity.selection_sha256,
    )
    _, test_summary = validate_test_evaluation_group(
        run_directory,
        probe_identity,
        selection_identity,
        test_identity,
        identity.test_sha256,
    )
    directory = report_directory(run_directory, identity.report_id)
    if directory.exists():
        return validate_report(run_directory, identity)

    try:
        import h5py
    except ImportError as exc:
        raise ReportingError(
            "Report generation is unavailable; install requirements.txt."
        ) from exc

    temporary = directory.parent / f".{identity.report_id}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with h5py.File(probe_file_path(run_directory), "r") as source:
            training = source[f"trainings/{probe_identity.probe_id}"]
            selection = source[f"selections/{selection_identity.selection_id}"]
            test = source[f"test_evaluations/{test_identity.test_id}"]
            positions = tuple(json.loads(selection.attrs["positions"]))
            states = tuple(json.loads(selection.attrs["state_labels"]))
            validation_labels = np.asarray(training["validation_labels"][...])
            validation_ids = training["validation_ids"][...].tolist()
            validation_probabilities = np.asarray(
                [
                    answer_probability(generations[int(item)], int(item))
                    for item in validation_ids
                ]
            )

            layer_rows = []
            representatives = {}
            overall_position = str(selection.attrs["selected_position"])
            overall_state = int(selection.attrs["selected_state_index"])
            for position_index, position in enumerate(positions):
                fprs = np.asarray(selection["fpr"][position_index, :])
                state_index = int(np.argmin(fprs))
                representatives[position] = state_index
                rows = []
                for index, state in enumerate(states):
                    row = {
                        "position": position,
                        "layer": _layer_number(state),
                        "state": state,
                        "threshold": float(
                            selection["thresholds"][position_index, index]
                        ),
                        "tpr": float(selection["tpr"][position_index, index]),
                        "fpr": float(selection["fpr"][position_index, index]),
                        "balanced_accuracy": _balanced_accuracy(
                            selection["tpr"][position_index, index],
                            selection["fpr"][position_index, index],
                        ),
                        "auroc": float(selection["auroc"][position_index, index]),
                        "position_winner": index == state_index,
                        "overall_winner": (
                            position == overall_position and index == overall_state
                        ),
                    }
                    rows.append(row)
                    layer_rows.append(row)
                _plot_layers(
                    temporary / f"validation_layers_{_slug(position)}.png",
                    position,
                    rows,
                )

            validation_curve_rows = []
            validation_points = {}
            representative_metrics = {}
            representative_series = {}
            for position_index, position in enumerate(positions):
                state_index = representatives[position]
                layer = _layer_number(states[state_index])
                series = f"{position} (layer {layer})"
                representative_series[position] = series
                scores = np.asarray(
                    training["position_models"][position]["validation_scores"][
                        state_index, :
                    ]
                )
                validation_curve_rows.extend(
                    _curve_rows(series, "validation", validation_labels, scores)
                )
                tpr = float(selection["tpr"][position_index, state_index])
                fpr = float(selection["fpr"][position_index, state_index])
                validation_points[series] = (tpr, fpr)
                representative_metrics[position] = {
                    "layer": layer,
                    "state": states[state_index],
                    "threshold": float(
                        selection["thresholds"][position_index, state_index]
                    ),
                    "tpr": tpr,
                    "fpr": fpr,
                    "balanced_accuracy": _balanced_accuracy(tpr, fpr),
                    "auroc": float(selection["auroc"][position_index, state_index]),
                }

            probability_validation = json.loads(test.attrs["probability_validation"])
            validation_curve_rows.extend(
                _curve_rows(
                    "output_probability",
                    "validation",
                    validation_labels,
                    validation_probabilities,
                )
            )
            validation_points["output_probability"] = (
                float(probability_validation["tpr"]),
                float(probability_validation["fpr"]),
            )
            _plot_curves(
                temporary / "validation_tpr_fpr.png",
                validation_curve_rows,
                validation_points,
                "Validation TPR-FPR comparison",
                representative_series[overall_position],
            )

            test_labels = np.asarray(test["labels"][...])
            test_curve_rows = _curve_rows(
                representative_series[overall_position],
                "test",
                test_labels,
                np.asarray(test["probe_scores"][...]),
            )
            test_curve_rows.extend(
                _curve_rows(
                    "output_probability",
                    "test",
                    test_labels,
                    np.asarray(test["probability_scores"][...]),
                )
            )
            test_points = {
                representative_series[overall_position]: (
                    float(test_summary["probe"]["tpr"]),
                    float(test_summary["probe"]["fpr"]),
                ),
                "output_probability": (
                    float(test_summary["output_probability"]["tpr"]),
                    float(test_summary["output_probability"]["fpr"]),
                ),
            }
            _plot_curves(
                temporary / "test_tpr_fpr.png",
                test_curve_rows,
                test_points,
                "Frozen test TPR-FPR comparison",
                representative_series[overall_position],
            )

        for values in test_summary.values():
            if isinstance(values, dict) and "tpr" in values:
                values["balanced_accuracy"] = _balanced_accuracy(
                    values["tpr"], values["fpr"]
                )
        summary = {
            "metrics": ["tpr", "fpr", "balanced_accuracy", "auroc"],
            "operational_metrics": ["coverage", "accepted_error_rate"],
            "validation": {
                "representatives": representative_metrics,
                "overall_winner": {
                    "position": overall_position,
                    **representative_metrics[overall_position],
                },
                "output_probability": {
                    "threshold": float(probability_validation["thresholds"]),
                    "tpr": float(probability_validation["tpr"]),
                    "fpr": float(probability_validation["fpr"]),
                    "balanced_accuracy": _balanced_accuracy(
                        probability_validation["tpr"],
                        probability_validation["fpr"],
                    ),
                    "auroc": float(probability_validation["auroc"]),
                },
            },
            "test": {
                "probe": {
                    key: test_summary["probe"][key]
                    for key in (
                        "threshold", "tpr", "fpr", "balanced_accuracy", "auroc",
                        "coverage", "accepted_error_rate",
                    )
                },
                "output_probability": {
                    key: test_summary["output_probability"][key]
                    for key in (
                        "threshold", "tpr", "fpr", "balanced_accuracy", "auroc",
                        "coverage", "accepted_error_rate",
                    )
                },
            },
        }
        _write_json(temporary / "metrics.json", summary)
        _write_csv(
            temporary / "validation_layers.csv",
            (
                "position",
                "layer",
                "state",
                "threshold",
                "tpr",
                "fpr",
                "balanced_accuracy",
                "auroc",
                "position_winner",
                "overall_winner",
            ),
            layer_rows,
        )
        curve_fields = ("split", "series", "threshold", "tpr", "fpr")
        _write_csv(
            temporary / "validation_tpr_fpr.csv", curve_fields, validation_curve_rows
        )
        _write_csv(temporary / "test_tpr_fpr.csv", curve_fields, test_curve_rows)
        result = _manifest(temporary, identity, summary)
        os.replace(temporary, directory)
        return result
    except ReportingError:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise ReportingError(f"Cannot create report: {directory}") from exc
