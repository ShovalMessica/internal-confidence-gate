"""Train and store linear correctness probes over saved activations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np

from src.activation_store import activation_file_path


PROBE_PROTOCOL_VERSION = 1
PROBE_STORE_SCHEMA_VERSION = 1
SELECTION_PROTOCOL_VERSION = 1
TEST_EVALUATION_PROTOCOL_VERSION = 1
PROBE_FILE = "probes.h5"
PROBE_SCORE = "probability_correct"
PROBABILITY_AGGREGATION = "geometric_mean_token_probability"
TOKEN_POOLING = "mean"
REGULARIZATION_C = 1.0
MAX_ITERATIONS = 5_000
SOLVER = "liblinear"
SELECTION_METRICS = (
    "thresholds",
    "accepted_correct",
    "accepted_incorrect",
    "tpr",
    "fpr",
    "auroc",
)


class ProbeError(ValueError):
    """Probe inputs, fitting, or saved artifacts are invalid."""


@dataclass(frozen=True)
class ProbeIdentity:
    probe_id: str
    fingerprint: str
    activation_sha256: str
    evaluation_sha256: str
    seed: int


@dataclass(frozen=True)
class ProbeTrainingResult:
    content_sha256: str
    summary: dict
    starting_candidates: int
    trained_candidates: int


@dataclass(frozen=True)
class SelectionIdentity:
    selection_id: str
    fingerprint: str
    probe_id: str
    probe_sha256: str
    target_tpr: float


@dataclass(frozen=True)
class ProbeSelectionResult:
    content_sha256: str
    summary: dict
    created: bool


@dataclass(frozen=True)
class TestEvaluationIdentity:
    test_id: str
    fingerprint: str
    selection_id: str
    selection_sha256: str
    generation_sha256: str


@dataclass(frozen=True)
class TestEvaluationResult:
    content_sha256: str
    summary: dict
    created: bool


@dataclass(frozen=True)
class _ProbeData:
    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    train_labels: np.ndarray
    validation_labels: np.ndarray
    positions: tuple[str, ...]
    state_labels: tuple[str, ...]
    hidden_size: int


@dataclass(frozen=True)
class _FittedProbe:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    iterations: int
    train_scores: np.ndarray
    validation_scores: np.ndarray


ProgressCallback = Callable[[int, int, float, int], None]


def _encoded(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _settings(seed: int) -> dict[str, object]:
    return {
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "score": PROBE_SCORE,
        "token_pooling": TOKEN_POOLING,
        "standardization": "training_split_only",
        "model": "logistic_regression",
        "penalty": "l2",
        "regularization_C": REGULARIZATION_C,
        "class_weight": "balanced",
        "solver": SOLVER,
        "max_iterations": MAX_ITERATIONS,
        "seed": seed,
    }


def build_probe_identity(
    activation_sha256: str, evaluation_sha256: str, seed: int
) -> ProbeIdentity:
    """Identify one downstream probe training without changing upstream identity."""
    payload = {
        "schema_version": PROBE_STORE_SCHEMA_VERSION,
        "activation_sha256": activation_sha256,
        "evaluation_sha256": evaluation_sha256,
        "settings": _settings(seed),
    }
    fingerprint = hashlib.sha256(_encoded(payload)).hexdigest()
    return ProbeIdentity(
        fingerprint[:12], fingerprint, activation_sha256, evaluation_sha256, seed
    )


def build_selection_identity(
    probe_id: str, probe_sha256: str, target_tpr: float
) -> SelectionIdentity:
    """Identify validation selection independently from probe training."""
    if (
        type(target_tpr) not in (int, float)
        or not math.isfinite(target_tpr)
        or not 0 < target_tpr <= 1
    ):
        raise ProbeError("target_tpr must be greater than 0 and at most 1.")
    target_tpr = float(target_tpr)
    payload = {
        "protocol_version": SELECTION_PROTOCOL_VERSION,
        "probe_id": probe_id,
        "probe_sha256": probe_sha256,
        "target_tpr": target_tpr,
    }
    fingerprint = hashlib.sha256(_encoded(payload)).hexdigest()
    return SelectionIdentity(
        fingerprint[:12], fingerprint, probe_id, probe_sha256, target_tpr
    )


def build_test_evaluation_identity(
    selection_id: str, selection_sha256: str, generation_sha256: str
) -> TestEvaluationIdentity:
    """Identify frozen test scoring independently from upstream computation."""
    payload = {
        "protocol_version": TEST_EVALUATION_PROTOCOL_VERSION,
        "selection_id": selection_id,
        "selection_sha256": selection_sha256,
        "generation_sha256": generation_sha256,
        "probability_aggregation": PROBABILITY_AGGREGATION,
    }
    fingerprint = hashlib.sha256(_encoded(payload)).hexdigest()
    return TestEvaluationIdentity(
        fingerprint[:12],
        fingerprint,
        selection_id,
        selection_sha256,
        generation_sha256,
    )


def probe_file_path(run_directory: Path) -> Path:
    return run_directory / PROBE_FILE


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ProbeError(
            "Probe storage is unavailable; install requirements.txt."
        ) from exc
    return h5py


def _open(path: Path, mode: str):
    try:
        return _h5py().File(path, mode)
    except (OSError, ValueError) as exc:
        raise ProbeError(f"Cannot open probe artifact: {path}") from exc


def _json_attr(container, name: str) -> object:
    try:
        return json.loads(container.attrs[name])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"Probe artifact has invalid '{name}' metadata.") from exc


def _labels(records: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray(
        [1 if record["outcome"] == "correct" else 0 for record in records],
        dtype=np.int8,
    )


def _split_records(
    evaluations: Mapping[int, Mapping[str, object]], split: str
) -> tuple[tuple[int, ...], np.ndarray]:
    records = [
        record
        for record in evaluations.values()
        if record.get("split") == split
        and record.get("outcome") in ("correct", "incorrect")
    ]
    ids = tuple(int(record["id"]) for record in records)
    labels = _labels(records)
    if not ids or set(labels.tolist()) != {0, 1}:
        raise ProbeError(
            f"Probe training requires correct and incorrect examples in the {split} split."
        )
    return ids, labels


def _activation_metadata(
    activation_path: Path,
    evaluations: Mapping[int, Mapping[str, object]],
) -> _ProbeData:
    train_ids, train_labels = _split_records(evaluations, "train")
    validation_ids, validation_labels = _split_records(evaluations, "validation")
    selected_ids = train_ids + validation_ids
    try:
        with _h5py().File(activation_path, "r") as source:
            if not bool(source.attrs.get("completed", False)):
                raise ProbeError("Activation capture is not complete.")
            state_labels = tuple(_json_attr(source, "state_labels"))
            hidden_size = int(source.attrs.get("hidden_size", 0))
            if (
                not state_labels
                or int(source.attrs.get("state_count", 0)) != len(state_labels)
                or hidden_size <= 0
                or "examples" not in source
            ):
                raise ProbeError("Activation metadata is invalid.")

            first_key = str(selected_ids[0])
            if first_key not in source["examples"]:
                raise ProbeError(
                    f"Activations are missing for example ID {selected_ids[0]}."
                )
            positions = tuple(source["examples"][first_key].keys())
            if not positions or any("/" in name for name in positions):
                raise ProbeError("Activation position names are invalid.")

            for example_id in selected_ids:
                key = str(example_id)
                if key not in source["examples"]:
                    raise ProbeError(
                        f"Activations are missing for example ID {example_id}."
                    )
                group = source["examples"][key]
                expected_split = str(evaluations[example_id]["split"])
                if group.attrs.get("split") != expected_split:
                    raise ProbeError(
                        f"Activation split is invalid for example ID {example_id}."
                    )
                if tuple(group.keys()) != positions:
                    raise ProbeError(
                        f"Activation positions differ for example ID {example_id}."
                    )
                for position in positions:
                    dataset = group[position]
                    if (
                        len(dataset.shape) != 3
                        or dataset.shape[0] != len(state_labels)
                        or dataset.shape[1] < 1
                        or dataset.shape[2] != hidden_size
                    ):
                        raise ProbeError(
                            f"Activation tensor '{position}' is invalid for "
                            f"example ID {example_id}."
                        )
    except ProbeError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ProbeError(f"Activation artifact is invalid: {activation_path}") from exc

    return _ProbeData(
        train_ids,
        validation_ids,
        train_labels,
        validation_labels,
        positions,
        state_labels,
        hidden_size,
    )


def _root(source) -> object:
    if not source.attrs:
        source.attrs["schema_version"] = PROBE_STORE_SCHEMA_VERSION
        source.create_group("trainings")
        source.flush()
    if (
        source.attrs.get("schema_version") != PROBE_STORE_SCHEMA_VERSION
        or "trainings" not in source
    ):
        raise ProbeError(f"Probe artifact has an invalid structure: {source.filename}")
    return source["trainings"]


def _selections(source, *, create: bool = False):
    if "selections" in source:
        return source["selections"]
    if create:
        return source.create_group("selections")
    return None


def _test_evaluations(source, *, create: bool = False):
    if "test_evaluations" in source:
        return source["test_evaluations"]
    if create:
        return source.create_group("test_evaluations")
    return None


def _create_training(group, identity: ProbeIdentity, data: _ProbeData) -> None:
    group.attrs["fingerprint"] = identity.fingerprint
    group.attrs["activation_sha256"] = identity.activation_sha256
    group.attrs["evaluation_sha256"] = identity.evaluation_sha256
    group.attrs["settings"] = json.dumps(_settings(identity.seed), sort_keys=True)
    group.attrs["positions"] = json.dumps(data.positions)
    group.attrs["state_labels"] = json.dumps(data.state_labels)
    group.attrs["hidden_size"] = data.hidden_size
    group.attrs["completed"] = False
    group.create_dataset("train_ids", data=data.train_ids, dtype="int64")
    group.create_dataset("validation_ids", data=data.validation_ids, dtype="int64")
    group.create_dataset("train_labels", data=data.train_labels, dtype="int8")
    group.create_dataset(
        "validation_labels", data=data.validation_labels, dtype="int8"
    )
    group.create_dataset(
        "completed_candidates",
        shape=(len(data.positions), len(data.state_labels)),
        dtype="bool",
        fillvalue=False,
    )
    positions = group.create_group("position_models")
    for name in data.positions:
        model = positions.create_group(name)
        state_shape = (len(data.state_labels), data.hidden_size)
        model.create_dataset("scaler_mean", shape=state_shape, dtype="float64")
        model.create_dataset("scaler_scale", shape=state_shape, dtype="float64")
        model.create_dataset("coefficients", shape=state_shape, dtype="float64")
        model.create_dataset(
            "intercepts", shape=(len(data.state_labels),), dtype="float64"
        )
        model.create_dataset(
            "iterations", shape=(len(data.state_labels),), dtype="int32"
        )
        model.create_dataset(
            "train_scores",
            shape=(len(data.state_labels), len(data.train_ids)),
            dtype="float64",
        )
        model.create_dataset(
            "validation_scores",
            shape=(len(data.state_labels), len(data.validation_ids)),
            dtype="float64",
        )


def _expected_metadata(identity: ProbeIdentity, data: _ProbeData) -> dict[str, object]:
    return {
        "fingerprint": identity.fingerprint,
        "activation_sha256": identity.activation_sha256,
        "evaluation_sha256": identity.evaluation_sha256,
        "settings": json.dumps(_settings(identity.seed), sort_keys=True),
        "positions": json.dumps(data.positions),
        "state_labels": json.dumps(data.state_labels),
        "hidden_size": data.hidden_size,
    }


def _validate_training(group, identity: ProbeIdentity, data: _ProbeData) -> None:
    for name, expected in _expected_metadata(identity, data).items():
        if group.attrs.get(name) != expected:
            raise ProbeError(
                f"Probe group '{identity.probe_id}' has incompatible {name}."
            )
    expected_arrays = {
        "train_ids": (data.train_ids, "int64"),
        "validation_ids": (data.validation_ids, "int64"),
        "train_labels": (data.train_labels, "int8"),
        "validation_labels": (data.validation_labels, "int8"),
    }
    for name, (expected, dtype) in expected_arrays.items():
        if (
            name not in group
            or group[name].dtype.name != dtype
            or not np.array_equal(group[name][...], np.asarray(expected))
        ):
            raise ProbeError(
                f"Probe group '{identity.probe_id}' has invalid {name}."
            )
    shape = (len(data.positions), len(data.state_labels))
    if (
        "completed_candidates" not in group
        or group["completed_candidates"].dtype.name != "bool"
        or group["completed_candidates"].shape != shape
        or "position_models" not in group
        or set(group["position_models"].keys()) != set(data.positions)
    ):
        raise ProbeError(f"Probe group '{identity.probe_id}' is incomplete.")
    for position in data.positions:
        model = group["position_models"][position]
        shapes = {
            "scaler_mean": (shape[1], data.hidden_size),
            "scaler_scale": (shape[1], data.hidden_size),
            "coefficients": (shape[1], data.hidden_size),
            "intercepts": (shape[1],),
            "iterations": (shape[1],),
            "train_scores": (shape[1], len(data.train_ids)),
            "validation_scores": (shape[1], len(data.validation_ids)),
        }
        if set(model.keys()) != set(shapes):
            raise ProbeError(
                f"Probe position '{position}' has an invalid structure."
            )
        for name, expected_shape in shapes.items():
            if model[name].shape != expected_shape:
                raise ProbeError(
                    f"Probe position '{position}' has invalid {name}."
                )


def _matrix(source, ids: Sequence[int], position: str, state: int) -> np.ndarray:
    rows = []
    for example_id in ids:
        values = np.asarray(
            source["examples"][str(example_id)][position][state, :, :],
            dtype=np.float32,
        )
        rows.append(values.mean(axis=0, dtype=np.float32))
    return np.stack(rows)


def _fit_probe(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    seed: int,
    position: str,
    state_label: str,
) -> _FittedProbe:
    if not np.isfinite(train_values).all() or not np.isfinite(validation_values).all():
        raise ProbeError(
            f"Probe features contain nonfinite values at '{position}', {state_label}."
        )
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ProbeError(
            "Probe training is unavailable; install requirements.txt."
        ) from exc

    scaler = StandardScaler().fit(train_values)
    train_scaled = scaler.transform(train_values)
    validation_scaled = scaler.transform(validation_values)
    model = LogisticRegression(
        penalty="l2",
        C=REGULARIZATION_C,
        class_weight="balanced",
        solver=SOLVER,
        max_iter=MAX_ITERATIONS,
        random_state=seed,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train_scaled, train_labels)
    except ConvergenceWarning as exc:
        raise ProbeError(
            f"Probe did not converge at '{position}', {state_label}."
        ) from exc
    except ValueError as exc:
        raise ProbeError(
            f"Probe fitting failed at '{position}', {state_label}: {exc}"
        ) from exc
    classes = model.classes_.tolist()
    if classes != [0, 1] or model.coef_.shape != (1, train_values.shape[1]):
        raise ProbeError(
            f"Probe produced invalid classes at '{position}', {state_label}."
        )
    correct_index = classes.index(1)
    return _FittedProbe(
        np.asarray(scaler.mean_, dtype=np.float64),
        np.asarray(scaler.scale_, dtype=np.float64),
        np.asarray(model.coef_[0], dtype=np.float64),
        float(model.intercept_[0]),
        int(model.n_iter_[0]),
        np.asarray(model.predict_proba(train_scaled)[:, correct_index]),
        np.asarray(model.predict_proba(validation_scaled)[:, correct_index]),
    )


def _write_candidate(model, state: int, fitted: _FittedProbe) -> None:
    model["scaler_mean"][state, :] = fitted.scaler_mean
    model["scaler_scale"][state, :] = fitted.scaler_scale
    model["coefficients"][state, :] = fitted.coefficients
    model["intercepts"][state] = fitted.intercept
    model["iterations"][state] = fitted.iterations
    model["train_scores"][state, :] = fitted.train_scores
    model["validation_scores"][state, :] = fitted.validation_scores


def _summary(group) -> dict[str, object]:
    positions = tuple(_json_attr(group, "positions"))
    states = tuple(_json_attr(group, "state_labels"))
    return {
        "positions": list(positions),
        "state_labels": list(states),
        "candidates": len(positions) * len(states),
        "train_examples": int(group["train_ids"].shape[0]),
        "validation_examples": int(group["validation_ids"].shape[0]),
        "score": PROBE_SCORE,
        "token_pooling": TOKEN_POOLING,
    }


def _validate_completed_layout(group) -> None:
    positions = tuple(_json_attr(group, "positions"))
    states = tuple(_json_attr(group, "state_labels"))
    hidden_size = int(group.attrs.get("hidden_size", 0))
    if not positions or not states or hidden_size <= 0:
        raise ProbeError("Completed probe metadata is invalid.")
    core = {
        "train_ids": ("int64", 1),
        "validation_ids": ("int64", 1),
        "train_labels": ("int8", 1),
        "validation_labels": ("int8", 1),
        "completed_candidates": ("bool", 2),
    }
    for name, (dtype, dimensions) in core.items():
        if (
            name not in group
            or group[name].dtype.name != dtype
            or len(group[name].shape) != dimensions
        ):
            raise ProbeError(f"Completed probe data '{name}' is invalid.")
    if (
        group["train_ids"].shape != group["train_labels"].shape
        or group["validation_ids"].shape != group["validation_labels"].shape
        or group["completed_candidates"].shape != (len(positions), len(states))
        or "position_models" not in group
        or set(group["position_models"].keys()) != set(positions)
    ):
        raise ProbeError("Completed probe data has inconsistent dimensions.")
    shapes = {
        "scaler_mean": ((len(states), hidden_size), "float64"),
        "scaler_scale": ((len(states), hidden_size), "float64"),
        "coefficients": ((len(states), hidden_size), "float64"),
        "intercepts": ((len(states),), "float64"),
        "iterations": ((len(states),), "int32"),
        "train_scores": ((len(states), group["train_ids"].shape[0]), "float64"),
        "validation_scores": (
            (len(states), group["validation_ids"].shape[0]),
            "float64",
        ),
    }
    for position in positions:
        model = group["position_models"][position]
        if set(model.keys()) != set(shapes):
            raise ProbeError(f"Completed probe position '{position}' is invalid.")
        for name, (shape, dtype) in shapes.items():
            if model[name].shape != shape or model[name].dtype.name != dtype:
                raise ProbeError(
                    f"Completed probe position '{position}' has invalid {name}."
                )
        if (
            not np.isfinite(model["scaler_mean"][...]).all()
            or not np.isfinite(model["scaler_scale"][...]).all()
            or not np.isfinite(model["coefficients"][...]).all()
            or not np.isfinite(model["intercepts"][...]).all()
            or np.any(model["scaler_scale"][...] <= 0)
            or np.any(model["iterations"][...] < 0)
        ):
            raise ProbeError(f"Completed probe position '{position}' is nonfinite.")
        for score_name in ("train_scores", "validation_scores"):
            scores = model[score_name][...]
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ProbeError(
                    f"Completed probe position '{position}' has invalid scores."
                )


def _content_sha256(group) -> str:
    if not bool(group.attrs.get("completed", False)):
        raise ProbeError("Cannot hash incomplete probe training.")
    digest = hashlib.sha256()
    metadata = {
        name: group.attrs[name]
        for name in (
            "fingerprint",
            "activation_sha256",
            "evaluation_sha256",
            "settings",
            "positions",
            "state_labels",
        )
    }
    metadata["hidden_size"] = int(group.attrs["hidden_size"])
    digest.update(_encoded(metadata))
    for name in (
        "train_ids",
        "validation_ids",
        "train_labels",
        "validation_labels",
        "completed_candidates",
    ):
        values = np.asarray(group[name][...])
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(_encoded(values.shape))
        digest.update(values.tobytes(order="C"))
    for position in tuple(_json_attr(group, "positions")):
        model = group["position_models"][position]
        for name in sorted(model.keys()):
            values = np.asarray(model[name][...])
            digest.update(f"{position}/{name}".encode("utf-8"))
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(_encoded(values.shape))
            digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def validate_probe_group(
    run_directory: Path,
    identity: ProbeIdentity,
    expected_sha256: str | None = None,
) -> tuple[str, dict]:
    """Validate one immutable completed group inside the shared probe file."""
    path = probe_file_path(run_directory)
    if not path.is_file():
        raise ProbeError(f"Probe artifact is missing: {path}")
    try:
        with _open(path, "r") as source:
            trainings = _root(source)
            if identity.probe_id not in trainings:
                raise ProbeError(f"Probe group is missing: {identity.probe_id}")
            group = trainings[identity.probe_id]
            if (
                group.attrs.get("fingerprint") != identity.fingerprint
                or group.attrs.get("activation_sha256") != identity.activation_sha256
                or group.attrs.get("evaluation_sha256") != identity.evaluation_sha256
                or group.attrs.get("settings")
                != json.dumps(_settings(identity.seed), sort_keys=True)
                or not bool(group.attrs.get("completed", False))
                or not np.asarray(group["completed_candidates"][...]).all()
            ):
                raise ProbeError(
                    f"Completed probe group is invalid: {identity.probe_id}"
                )
            _validate_completed_layout(group)
            content_hash = _content_sha256(group)
            if expected_sha256 is not None and content_hash != expected_sha256:
                raise ProbeError(
                    f"Completed probe group has changed: {identity.probe_id}"
                )
            return content_hash, _summary(group)
    except ProbeError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ProbeError(f"Probe artifact is invalid: {path}") from exc


def _candidate_metrics(
    scores: np.ndarray, labels: np.ndarray, target_tpr: float
) -> tuple[float, int, int, float, float, float]:
    if (
        scores.ndim != 1
        or scores.shape != labels.shape
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
        or set(labels.tolist()) != {0, 1}
    ):
        raise ProbeError("Validation probe scores or labels are invalid.")
    correct = scores[labels == 1]
    incorrect = scores[labels == 0]
    required_correct = math.ceil(target_tpr * correct.size)
    threshold = np.sort(correct)[-required_correct]
    accepted_correct = int(np.count_nonzero(correct >= threshold))
    accepted_incorrect = int(np.count_nonzero(incorrect >= threshold))
    return (
        float(threshold),
        accepted_correct,
        accepted_incorrect,
        accepted_correct / correct.size,
        accepted_incorrect / incorrect.size,
        _auroc(labels, scores),
    )


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ProbeError(
            "Probe evaluation is unavailable; install requirements.txt."
        ) from exc
    return float(roc_auc_score(labels, scores))


def _frozen_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, object]:
    accepted = scores >= threshold
    correct = labels == 1
    incorrect = labels == 0
    accepted_correct = int(np.count_nonzero(accepted & correct))
    accepted_incorrect = int(np.count_nonzero(accepted & incorrect))
    total_correct = int(np.count_nonzero(correct))
    total_incorrect = int(np.count_nonzero(incorrect))
    return {
        "threshold": threshold,
        "accepted_correct": accepted_correct,
        "total_correct": total_correct,
        "tpr": accepted_correct / total_correct,
        "accepted_incorrect": accepted_incorrect,
        "total_incorrect": total_incorrect,
        "fpr": accepted_incorrect / total_incorrect,
        "accepted": int(np.count_nonzero(accepted)),
        "total": int(labels.size),
        "acceptance_rate": float(np.mean(accepted)),
        "auroc": _auroc(labels, scores),
    }


def answer_probability(record: Mapping[str, object], example_id: int) -> float:
    """Aggregate saved answer-token probabilities into one confidence score."""
    answer = record.get("answer")
    logprobs = answer.get("token_logprobs") if isinstance(answer, Mapping) else None
    if (
        not isinstance(logprobs, (list, tuple))
        or not logprobs
        or any(type(value) not in (int, float) for value in logprobs)
    ):
        raise ProbeError(
            f"Answer token probabilities are invalid for example ID {example_id}."
        )
    values = np.asarray(logprobs, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values > 1e-6):
        raise ProbeError(
            f"Answer token probabilities are invalid for example ID {example_id}."
        )
    return float(np.exp(values.mean()))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return result


def _selection_summary(group) -> dict[str, object]:
    return {
        "target_tpr": float(group.attrs["target_tpr"]),
        "position": str(group.attrs["selected_position"]),
        "position_index": int(group.attrs["selected_position_index"]),
        "state": str(group.attrs["selected_state"]),
        "state_index": int(group.attrs["selected_state_index"]),
        "threshold": float(group.attrs["selected_threshold"]),
        "tpr": float(group.attrs["selected_tpr"]),
        "fpr": float(group.attrs["selected_fpr"]),
        "auroc": float(group.attrs["selected_auroc"]),
        "accepted_correct": int(group.attrs["selected_accepted_correct"]),
        "total_correct": int(group.attrs["total_correct"]),
        "accepted_incorrect": int(group.attrs["selected_accepted_incorrect"]),
        "total_incorrect": int(group.attrs["total_incorrect"]),
    }


def _selection_sha256(group) -> str:
    if not bool(group.attrs.get("completed", False)):
        raise ProbeError("Cannot hash incomplete probe selection.")
    digest = hashlib.sha256()
    metadata_names = (
        "fingerprint",
        "probe_id",
        "probe_sha256",
        "protocol_version",
        "target_tpr",
        "positions",
        "state_labels",
        "selected_position",
        "selected_position_index",
        "selected_state",
        "selected_state_index",
        "selected_threshold",
        "selected_tpr",
        "selected_fpr",
        "selected_auroc",
        "selected_accepted_correct",
        "selected_accepted_incorrect",
        "total_correct",
        "total_incorrect",
    )
    metadata = {}
    for name in metadata_names:
        value = group.attrs[name]
        metadata[name] = value.item() if isinstance(value, np.generic) else value
    digest.update(_encoded(metadata))
    for name in sorted(group.keys()):
        values = np.asarray(group[name][...])
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(_encoded(values.shape))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validate_selection_layout(group, identity: SelectionIdentity) -> None:
    expected = {
        "fingerprint": identity.fingerprint,
        "probe_id": identity.probe_id,
        "probe_sha256": identity.probe_sha256,
        "protocol_version": SELECTION_PROTOCOL_VERSION,
        "target_tpr": identity.target_tpr,
    }
    if any(group.attrs.get(name) != value for name, value in expected.items()):
        raise ProbeError(f"Probe selection is incompatible: {identity.selection_id}")
    positions = tuple(_json_attr(group, "positions"))
    states = tuple(_json_attr(group, "state_labels"))
    shape = (len(positions), len(states))
    arrays = {
        "thresholds": "float64",
        "accepted_correct": "int64",
        "accepted_incorrect": "int64",
        "tpr": "float64",
        "fpr": "float64",
        "auroc": "float64",
    }
    if not positions or not states or set(group.keys()) != set(arrays):
        raise ProbeError(f"Probe selection is incomplete: {identity.selection_id}")
    for name, dtype in arrays.items():
        if group[name].shape != shape or group[name].dtype.name != dtype:
            raise ProbeError(
                f"Probe selection has invalid '{name}': {identity.selection_id}"
            )
    for name in ("thresholds", "tpr", "fpr", "auroc"):
        values = group[name][...]
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ProbeError(
                f"Probe selection has invalid '{name}': {identity.selection_id}"
            )
    position_index = int(group.attrs.get("selected_position_index", -1))
    state_index = int(group.attrs.get("selected_state_index", -1))
    if (
        not bool(group.attrs.get("completed", False))
        or not 0 <= position_index < len(positions)
        or not 0 <= state_index < len(states)
        or group.attrs.get("selected_position") != positions[position_index]
        or group.attrs.get("selected_state") != states[state_index]
    ):
        raise ProbeError(f"Probe selection winner is invalid: {identity.selection_id}")
    selected = (position_index, state_index)
    expected_values = {
        "selected_threshold": float(group["thresholds"][selected]),
        "selected_accepted_correct": int(group["accepted_correct"][selected]),
        "selected_accepted_incorrect": int(group["accepted_incorrect"][selected]),
        "selected_tpr": float(group["tpr"][selected]),
        "selected_fpr": float(group["fpr"][selected]),
        "selected_auroc": float(group["auroc"][selected]),
    }
    if any(group.attrs.get(name) != value for name, value in expected_values.items()):
        raise ProbeError(f"Probe selection summary is invalid: {identity.selection_id}")


def validate_selection_group(
    run_directory: Path,
    probe_identity: ProbeIdentity,
    identity: SelectionIdentity,
    expected_sha256: str | None = None,
) -> tuple[str, dict]:
    """Validate a completed validation selection and its source probes."""
    validate_probe_group(run_directory, probe_identity, identity.probe_sha256)
    path = probe_file_path(run_directory)
    try:
        with _open(path, "r") as source:
            selections = _selections(source)
            if selections is None or identity.selection_id not in selections:
                raise ProbeError(f"Probe selection is missing: {identity.selection_id}")
            group = selections[identity.selection_id]
            _validate_selection_layout(group, identity)
            content_hash = _selection_sha256(group)
            if expected_sha256 is not None and content_hash != expected_sha256:
                raise ProbeError(
                    f"Completed probe selection has changed: {identity.selection_id}"
                )
            return content_hash, _selection_summary(group)
    except ProbeError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ProbeError(f"Probe artifact is invalid: {path}") from exc


def select_probe(
    run_directory: Path,
    probe_identity: ProbeIdentity,
    identity: SelectionIdentity,
) -> ProbeSelectionResult:
    """Select one candidate and threshold using validation scores only."""
    validate_probe_group(run_directory, probe_identity, identity.probe_sha256)
    path = probe_file_path(run_directory)
    try:
        with _open(path, "a") as output:
            trainings = _root(output)
            training = trainings[probe_identity.probe_id]
            selections = _selections(output, create=True)
            if identity.selection_id in selections:
                group = selections[identity.selection_id]
                _validate_selection_layout(group, identity)
                return ProbeSelectionResult(
                    _selection_sha256(group), _selection_summary(group), False
                )

            positions = tuple(_json_attr(training, "positions"))
            states = tuple(_json_attr(training, "state_labels"))
            labels = np.asarray(training["validation_labels"][...], dtype=np.int8)
            shape = (len(positions), len(states))
            metrics = {
                "thresholds": np.empty(shape, dtype=np.float64),
                "accepted_correct": np.empty(shape, dtype=np.int64),
                "accepted_incorrect": np.empty(shape, dtype=np.int64),
                "tpr": np.empty(shape, dtype=np.float64),
                "fpr": np.empty(shape, dtype=np.float64),
                "auroc": np.empty(shape, dtype=np.float64),
            }
            for position_index, position in enumerate(positions):
                scores_by_state = training["position_models"][position][
                    "validation_scores"
                ]
                for state_index in range(len(states)):
                    values = _candidate_metrics(
                        np.asarray(scores_by_state[state_index, :]),
                        labels,
                        identity.target_tpr,
                    )
                    for name, value in zip(SELECTION_METRICS, values):
                        metrics[name][position_index, state_index] = value

            winner = min(
                (
                    float(metrics["fpr"][position_index, state_index]),
                    position,
                    state_index,
                    position_index,
                )
                for position_index, position in enumerate(positions)
                for state_index in range(len(states))
            )
            _, position, state_index, position_index = winner
            pending_name = f"_pending_{identity.selection_id}"
            if pending_name in selections:
                del selections[pending_name]
            group = selections.create_group(pending_name)
            group.attrs["fingerprint"] = identity.fingerprint
            group.attrs["probe_id"] = identity.probe_id
            group.attrs["probe_sha256"] = identity.probe_sha256
            group.attrs["protocol_version"] = SELECTION_PROTOCOL_VERSION
            group.attrs["target_tpr"] = identity.target_tpr
            group.attrs["positions"] = json.dumps(positions)
            group.attrs["state_labels"] = json.dumps(states)
            group.attrs["selected_position"] = position
            group.attrs["selected_position_index"] = position_index
            group.attrs["selected_state"] = states[state_index]
            group.attrs["selected_state_index"] = state_index
            for name, values in metrics.items():
                group.create_dataset(name, data=values)
            group.attrs["selected_threshold"] = metrics["thresholds"][
                position_index, state_index
            ]
            group.attrs["selected_tpr"] = metrics["tpr"][position_index, state_index]
            group.attrs["selected_fpr"] = metrics["fpr"][position_index, state_index]
            group.attrs["selected_auroc"] = metrics["auroc"][
                position_index, state_index
            ]
            group.attrs["selected_accepted_correct"] = metrics[
                "accepted_correct"
            ][position_index, state_index]
            group.attrs["selected_accepted_incorrect"] = metrics[
                "accepted_incorrect"
            ][position_index, state_index]
            group.attrs["total_correct"] = int(np.count_nonzero(labels == 1))
            group.attrs["total_incorrect"] = int(np.count_nonzero(labels == 0))
            group.attrs["completed"] = True
            output.flush()
            selections.move(pending_name, identity.selection_id)
            output.flush()
            completed = selections[identity.selection_id]
            return ProbeSelectionResult(
                _selection_sha256(completed), _selection_summary(completed), True
            )
    except ProbeError:
        raise
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ProbeError(f"Cannot select or save probe: {path}") from exc


def _test_summary(group) -> dict[str, object]:
    labels = np.asarray(group["labels"][...], dtype=np.int8)
    probe = _frozen_metrics(
        labels,
        np.asarray(group["probe_scores"][...]),
        float(group.attrs["probe_threshold"]),
    )
    probability = _frozen_metrics(
        labels,
        np.asarray(group["probability_scores"][...]),
        float(group.attrs["probability_threshold"]),
    )
    probability["aggregation"] = PROBABILITY_AGGREGATION
    return {
        "examples": int(labels.size),
        "correct": int(np.count_nonzero(labels == 1)),
        "incorrect": int(np.count_nonzero(labels == 0)),
        "probe": probe,
        "output_probability": probability,
    }


def _test_sha256(group) -> str:
    if not bool(group.attrs.get("completed", False)):
        raise ProbeError("Cannot hash incomplete frozen test evaluation.")
    metadata_names = (
        "fingerprint",
        "selection_id",
        "selection_sha256",
        "generation_sha256",
        "protocol_version",
        "probability_aggregation",
        "probe_threshold",
        "probability_threshold",
        "probability_validation",
    )
    metadata = {}
    for name in metadata_names:
        value = group.attrs[name]
        metadata[name] = value.item() if isinstance(value, np.generic) else value
    digest = hashlib.sha256(_encoded(metadata))
    for name in sorted(group.keys()):
        values = np.asarray(group[name][...])
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(_encoded(values.shape))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validate_test_layout(group, identity: TestEvaluationIdentity) -> None:
    expected = {
        "fingerprint": identity.fingerprint,
        "selection_id": identity.selection_id,
        "selection_sha256": identity.selection_sha256,
        "generation_sha256": identity.generation_sha256,
        "protocol_version": TEST_EVALUATION_PROTOCOL_VERSION,
        "probability_aggregation": PROBABILITY_AGGREGATION,
    }
    if any(group.attrs.get(name) != value for name, value in expected.items()):
        raise ProbeError(f"Frozen test evaluation is incompatible: {identity.test_id}")
    arrays = {
        "ids": "int64",
        "labels": "int8",
        "probe_scores": "float64",
        "probability_scores": "float64",
        "probe_accepted": "bool",
        "probability_accepted": "bool",
    }
    if set(group.keys()) != set(arrays):
        raise ProbeError(f"Frozen test evaluation is incomplete: {identity.test_id}")
    size = group["ids"].shape
    if len(size) != 1 or size[0] == 0:
        raise ProbeError(f"Frozen test evaluation has no examples: {identity.test_id}")
    for name, dtype in arrays.items():
        if group[name].shape != size or group[name].dtype.name != dtype:
            raise ProbeError(
                f"Frozen test evaluation has invalid '{name}': {identity.test_id}"
            )
    labels = group["labels"][...]
    probe_scores = group["probe_scores"][...]
    probability_scores = group["probability_scores"][...]
    probe_threshold = float(group.attrs.get("probe_threshold", float("nan")))
    probability_threshold = float(
        group.attrs.get("probability_threshold", float("nan"))
    )
    if (
        not bool(group.attrs.get("completed", False))
        or set(labels.tolist()) != {0, 1}
        or len(set(group["ids"][...].tolist())) != size[0]
        or not np.isfinite(probe_scores).all()
        or not np.isfinite(probability_scores).all()
        or np.any((probe_scores < 0) | (probe_scores > 1))
        or np.any((probability_scores < 0) | (probability_scores > 1))
        or not 0 <= probe_threshold <= 1
        or not 0 <= probability_threshold <= 1
        or not np.array_equal(group["probe_accepted"][...], probe_scores >= probe_threshold)
        or not np.array_equal(
            group["probability_accepted"][...],
            probability_scores >= probability_threshold,
        )
        or not isinstance(_json_attr(group, "probability_validation"), dict)
    ):
        raise ProbeError(f"Frozen test evaluation is invalid: {identity.test_id}")


def validate_test_evaluation_group(
    run_directory: Path,
    probe_identity: ProbeIdentity,
    selection_identity: SelectionIdentity,
    identity: TestEvaluationIdentity,
    expected_sha256: str | None = None,
) -> tuple[str, dict]:
    """Validate frozen test scores and their selected-probe provenance."""
    validate_selection_group(
        run_directory,
        probe_identity,
        selection_identity,
        identity.selection_sha256,
    )
    path = probe_file_path(run_directory)
    try:
        with _open(path, "r") as source:
            tests = _test_evaluations(source)
            if tests is None or identity.test_id not in tests:
                raise ProbeError(f"Frozen test evaluation is missing: {identity.test_id}")
            group = tests[identity.test_id]
            _validate_test_layout(group, identity)
            content_hash = _test_sha256(group)
            if expected_sha256 is not None and content_hash != expected_sha256:
                raise ProbeError(
                    f"Frozen test evaluation has changed: {identity.test_id}"
                )
            return content_hash, _test_summary(group)
    except ProbeError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ProbeError(f"Probe artifact is invalid: {path}") from exc


def evaluate_frozen_test(
    run_directory: Path,
    probe_identity: ProbeIdentity,
    selection_identity: SelectionIdentity,
    identity: TestEvaluationIdentity,
    evaluations: Mapping[int, Mapping[str, object]],
    generations: Mapping[int, Mapping[str, object]],
) -> TestEvaluationResult:
    """Apply frozen probe and probability thresholds to concrete test predictions."""
    validate_selection_group(
        run_directory,
        probe_identity,
        selection_identity,
        identity.selection_sha256,
    )
    validation_ids, validation_labels = _split_records(evaluations, "validation")
    test_ids, test_labels = _split_records(evaluations, "test")
    validation_probabilities = np.asarray(
        [answer_probability(generations[item], item) for item in validation_ids]
    )
    test_probabilities = np.asarray(
        [answer_probability(generations[item], item) for item in test_ids]
    )
    probability_values = _candidate_metrics(
        validation_probabilities,
        validation_labels,
        selection_identity.target_tpr,
    )
    probability_validation = dict(zip(SELECTION_METRICS, probability_values))
    probability_threshold = float(probability_validation["thresholds"])

    path = probe_file_path(run_directory)
    activation_path = activation_file_path(run_directory)
    try:
        with (
            _h5py().File(activation_path, "r") as activations,
            _open(path, "a") as output,
        ):
            training = _root(output)[probe_identity.probe_id]
            selection = _selections(output)[selection_identity.selection_id]
            tests = _test_evaluations(output, create=True)
            if identity.test_id in tests:
                group = tests[identity.test_id]
                _validate_test_layout(group, identity)
                return TestEvaluationResult(
                    _test_sha256(group), _test_summary(group), False
                )

            position = str(selection.attrs["selected_position"])
            state = int(selection.attrs["selected_state_index"])
            probe_threshold = float(selection.attrs["selected_threshold"])
            model = training["position_models"][position]
            test_values = _matrix(activations, test_ids, position, state)
            standardized = (
                test_values - np.asarray(model["scaler_mean"][state])
            ) / np.asarray(model["scaler_scale"][state])
            logits = (
                standardized @ np.asarray(model["coefficients"][state])
                + float(model["intercepts"][state])
            )
            probe_scores = _sigmoid(np.asarray(logits, dtype=np.float64))
            if not np.isfinite(probe_scores).all():
                raise ProbeError("Frozen probe produced nonfinite test scores.")

            pending_name = f"_pending_{identity.test_id}"
            if pending_name in tests:
                del tests[pending_name]
            group = tests.create_group(pending_name)
            group.attrs["fingerprint"] = identity.fingerprint
            group.attrs["selection_id"] = identity.selection_id
            group.attrs["selection_sha256"] = identity.selection_sha256
            group.attrs["generation_sha256"] = identity.generation_sha256
            group.attrs["protocol_version"] = TEST_EVALUATION_PROTOCOL_VERSION
            group.attrs["probability_aggregation"] = PROBABILITY_AGGREGATION
            group.attrs["probe_threshold"] = probe_threshold
            group.attrs["probability_threshold"] = probability_threshold
            group.attrs["probability_validation"] = json.dumps(
                probability_validation, sort_keys=True
            )
            group.create_dataset("ids", data=test_ids, dtype="int64")
            group.create_dataset("labels", data=test_labels, dtype="int8")
            group.create_dataset("probe_scores", data=probe_scores, dtype="float64")
            group.create_dataset(
                "probability_scores", data=test_probabilities, dtype="float64"
            )
            group.create_dataset(
                "probe_accepted", data=probe_scores >= probe_threshold, dtype="bool"
            )
            group.create_dataset(
                "probability_accepted",
                data=test_probabilities >= probability_threshold,
                dtype="bool",
            )
            group.attrs["completed"] = True
            output.flush()
            tests.move(pending_name, identity.test_id)
            output.flush()
            completed = tests[identity.test_id]
            return TestEvaluationResult(
                _test_sha256(completed), _test_summary(completed), True
            )
    except ProbeError:
        raise
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ProbeError(f"Cannot evaluate or save frozen test results: {path}") from exc


def train_probes(
    run_directory: Path,
    identity: ProbeIdentity,
    evaluations: Mapping[int, Mapping[str, object]],
    progress: ProgressCallback | None = None,
) -> ProbeTrainingResult:
    """Fit or resume every position-by-state probe without reading test features."""
    activation_path = activation_file_path(run_directory)
    data = _activation_metadata(activation_path, evaluations)
    path = probe_file_path(run_directory)
    if path.exists() and path.stat().st_size == 0:
        path.unlink()

    try:
        with _h5py().File(activation_path, "r") as activations, _open(path, "a") as output:
            trainings = _root(output)
            if identity.probe_id not in trainings:
                group = trainings.create_group(identity.probe_id)
                _create_training(group, identity, data)
                output.flush()
            group = trainings[identity.probe_id]
            _validate_training(group, identity, data)
            completed = group["completed_candidates"]
            starting = int(np.count_nonzero(completed[...]))
            total = int(completed.size)
            if bool(group.attrs.get("completed", False)):
                if starting != total:
                    raise ProbeError("Completed probe training has missing candidates.")
                content_hash = _content_sha256(group)
                return ProbeTrainingResult(
                    content_hash, _summary(group), starting, 0
                )

            started = time.monotonic()
            trained = 0
            for position_index, position in enumerate(data.positions):
                model = group["position_models"][position]
                for state, state_label in enumerate(data.state_labels):
                    if bool(completed[position_index, state]):
                        continue
                    train_values = _matrix(
                        activations, data.train_ids, position, state
                    )
                    validation_values = _matrix(
                        activations, data.validation_ids, position, state
                    )
                    fitted = _fit_probe(
                        train_values,
                        data.train_labels,
                        validation_values,
                        identity.seed,
                        position,
                        state_label,
                    )
                    _write_candidate(model, state, fitted)
                    output.flush()
                    completed[position_index, state] = True
                    output.flush()
                    trained += 1
                    interval = max(1, total // 20)
                    if progress is not None and (
                        trained == 1
                        or starting + trained == total
                        or trained % interval == 0
                    ):
                        progress(starting + trained, total, started, starting)

            if not np.asarray(completed[...]).all():
                raise ProbeError("Probe training ended with missing candidates.")
            group.attrs["completed"] = True
            output.flush()
            content_hash = _content_sha256(group)
            return ProbeTrainingResult(
                content_hash, _summary(group), starting, trained
            )
    except ProbeError:
        raise
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ProbeError(f"Cannot train or save probes: {path}") from exc
