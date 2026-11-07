"""Ranking metrics and patient-level-pilot bootstrap intervals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .labels import Endpoint, PRIMARY_ENDPOINTS, binary_target


def endpoint_observations(
    predictions: Iterable[Mapping[str, Any]], endpoint: Endpoint, positive_threshold: int = 2
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract non-questionable reference labels and matching model scores."""
    targets: list[int] = []
    scores: list[float] = []
    excluded = 0
    for prediction in predictions:
        target = binary_target(prediction["labels"], endpoint, positive_threshold)
        if target is None:
            excluded += 1
            continue
        targets.append(target)
        scores.append(float(prediction["scores"][endpoint.score_key]))
    return np.asarray(targets, dtype=int), np.asarray(scores, dtype=float), excluded


def _point_metrics(targets: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None]:
    if len(targets) == 0 or len(np.unique(targets)) < 2:
        return None, None
    return float(roc_auc_score(targets, scores)), float(average_precision_score(targets, scores))


def _positive_definition(positive_threshold: int) -> str:
    if positive_threshold == 1:
        return "TAIX grade >=1; grade 0 negative"
    if positive_threshold == 2:
        return "TAIX grade >=2; grade 0 negative; grade 1 excluded"
    return (
        f"TAIX grade >={positive_threshold}; grade 0 negative; "
        f"grades 1--{positive_threshold - 1} excluded"
    )


def bootstrap_interval(
    targets: np.ndarray,
    scores: np.ndarray,
    metric: str,
    repetitions: int,
    seed: int,
) -> list[float] | None:
    """Percentile 95% CI over one-image-per-patient pilot observations."""
    if len(targets) == 0 or len(np.unique(targets)) < 2:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        index = rng.integers(0, len(targets), size=len(targets))
        sampled_targets, sampled_scores = targets[index], scores[index]
        if len(np.unique(sampled_targets)) < 2:
            continue
        if metric == "auroc":
            values.append(float(roc_auc_score(sampled_targets, sampled_scores)))
        elif metric == "average_precision":
            values.append(float(average_precision_score(sampled_targets, sampled_scores)))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    if not values:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def evaluate_predictions(
    predictions: list[Mapping[str, Any]],
    repetitions: int = 2_000,
    seed: int = 20261107,
    positive_threshold: int = 2,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Produce pre-specified primary endpoint metrics and plotting inputs."""
    endpoint_results: dict[str, Any] = {}
    plotting_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for offset, endpoint in enumerate(PRIMARY_ENDPOINTS):
        targets, scores, excluded = endpoint_observations(
            predictions, endpoint, positive_threshold
        )
        auroc, average_precision = _point_metrics(targets, scores)
        endpoint_results[endpoint.name] = {
            "score_key": endpoint.score_key,
            "relation_note": endpoint.relation_note,
            "positive_definition": _positive_definition(positive_threshold),
            "n_evaluable": int(len(targets)),
            "n_excluded_questionable": excluded,
            "n_positive": int(targets.sum()),
            "prevalence": float(targets.mean()) if len(targets) else None,
            "auroc": auroc,
            "auroc_95_ci": bootstrap_interval(
                targets, scores, "auroc", repetitions, seed + offset
            ),
            "average_precision": average_precision,
            "average_precision_95_ci": bootstrap_interval(
                targets, scores, "average_precision", repetitions, seed + 100 + offset
            ),
        }
        plotting_inputs[endpoint.name] = (targets, scores)
    return endpoint_results, plotting_inputs


def roc_coordinates(targets: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ROC coordinates only when both reference classes are present."""
    if len(targets) == 0 or len(np.unique(targets)) < 2:
        return None
    fpr, tpr, _ = roc_curve(targets, scores)
    return fpr, tpr
