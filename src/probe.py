"""Train and store linear correctness probes over saved activations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np

from src.activation_store import activation_file_path


PROBE_PROTOCOL_VERSION = 1
PROBE_STORE_SCHEMA_VERSION = 1
PROBE_FILE = "probes.h5"
PROBE_SCORE = "probability_correct"
TOKEN_POOLING = "mean"
REGULARIZATION_C = 1.0
MAX_ITERATIONS = 5_000
SOLVER = "liblinear"


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
