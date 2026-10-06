"""Pure fitting and metric operations for linear reliability probes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping
import warnings

import numpy as np


MAX_ITERATIONS = 5_000
SOLVER = "liblinear"


class ProbeError(ValueError):
    """Probe inputs, fitting, or saved artifacts are invalid."""


@dataclass(frozen=True)
class FittedProbe:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    iterations: int
    train_scores: np.ndarray
    validation_scores: np.ndarray


def fit_probe(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    seed: int,
    position: str,
    state_label: str,
    regularization_c: float,
    class_weight: str,
) -> FittedProbe:
    """Fit one standardized logistic reliability probe."""
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
        C=regularization_c,
        class_weight=None if class_weight == "none" else class_weight,
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
    return FittedProbe(
        np.asarray(scaler.mean_, dtype=np.float64),
        np.asarray(scaler.scale_, dtype=np.float64),
        np.asarray(model.coef_[0], dtype=np.float64),
        float(model.intercept_[0]),
        int(model.n_iter_[0]),
        np.asarray(model.predict_proba(train_scaled)[:, correct_index]),
        np.asarray(model.predict_proba(validation_scaled)[:, correct_index]),
    )


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ProbeError(
            "Probe evaluation is unavailable; install requirements.txt."
        ) from exc
    return float(roc_auc_score(labels, scores))


def candidate_metrics(
    scores: np.ndarray, labels: np.ndarray, target_tpr: float
) -> tuple[float, int, int, float, float, float]:
    """Select a validation threshold and return its gate metrics."""
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
        auroc(labels, scores),
    )


def frozen_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, object]:
    """Evaluate one frozen gate, including coverage and accepted-error rate."""
    accepted = scores >= threshold
    correct = labels == 1
    incorrect = labels == 0
    accepted_correct = int(np.count_nonzero(accepted & correct))
    accepted_incorrect = int(np.count_nonzero(accepted & incorrect))
    total_correct = int(np.count_nonzero(correct))
    total_incorrect = int(np.count_nonzero(incorrect))
    accepted_count = accepted_correct + accepted_incorrect
    return {
        "threshold": threshold,
        "accepted_correct": accepted_correct,
        "total_correct": total_correct,
        "tpr": accepted_correct / total_correct,
        "accepted_incorrect": accepted_incorrect,
        "total_incorrect": total_incorrect,
        "fpr": accepted_incorrect / total_incorrect,
        "accepted": accepted_count,
        "total": int(labels.size),
        "coverage": accepted_count / labels.size,
        "accepted_error_rate": (
            accepted_incorrect / accepted_count if accepted_count else None
        ),
        "auroc": auroc(labels, scores),
    }


def answer_probability(record: Mapping[str, object], example_id: int) -> float:
    """Aggregate raw answer-token probabilities into one confidence score."""
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


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return result
