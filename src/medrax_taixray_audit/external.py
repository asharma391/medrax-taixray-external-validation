"""Generic, aggregate-only evaluation utilities for non-TAIX CXR datasets.

The TAIX-Ray pipeline intentionally has dataset-specific ordinal-label logic.
This module is deliberately separate: it evaluates datasets with binary,
image-level reference labels and never pools them with TAIX-Ray results.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


@dataclass(frozen=True)
class BinaryEndpoint:
    """A direct binary reference-label to model-score mapping."""

    name: str
    score_key: str
    source_label: str
    relation_note: str


def labels_from_mask_values(
    mask_values: np.ndarray, source_label_ids: Mapping[str, int]
) -> dict[str, int]:
    """Derive image-level binary labels from a semantic mask's class values.

    ``source_label_ids`` maps a source finding name to its declared integer
    mask value. The function makes no inference from pixels beyond whether the
    source annotation marks at least one pixel as that class.
    """
    observed = set(np.asarray(mask_values).reshape(-1).tolist())
    return {
        source_label: int(class_id in observed)
        for source_label, class_id in source_label_ids.items()
    }


def _bootstrap_interval(
    targets: np.ndarray, scores: np.ndarray, metric: str, repetitions: int, seed: int
) -> list[float] | None:
    if len(targets) == 0 or len(np.unique(targets)) < 2:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        sample_idx = rng.integers(0, len(targets), size=len(targets))
        sample_targets, sample_scores = targets[sample_idx], scores[sample_idx]
        if len(np.unique(sample_targets)) < 2:
            continue
        if metric == "auroc":
            values.append(float(roc_auc_score(sample_targets, sample_scores)))
        elif metric == "average_precision":
            values.append(float(average_precision_score(sample_targets, sample_scores)))
        else:  # pragma: no cover - protected by callers.
            raise ValueError(f"Unsupported metric: {metric}")
    if not values:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def evaluate_binary_predictions(
    predictions: Iterable[Mapping[str, Any]],
    endpoints: Iterable[BinaryEndpoint],
    repetitions: int = 2_000,
    seed: int = 20261107,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Calculate ranking metrics for direct binary-label endpoints.

    Each prediction is an in-memory mapping containing ``labels`` keyed by the
    source-label name and ``scores`` keyed by the model pathology name. The
    caller controls persistence, so this function cannot write row-level data.
    """
    prediction_list = list(predictions)
    results: dict[str, Any] = {}
    roc_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for offset, endpoint in enumerate(endpoints):
        targets = np.asarray(
            [int(item["labels"][endpoint.source_label]) for item in prediction_list],
            dtype=int,
        )
        scores = np.asarray(
            [float(item["scores"][endpoint.score_key]) for item in prediction_list],
            dtype=float,
        )
        if not np.isin(targets, (0, 1)).all():
            raise ValueError(
                f"{endpoint.name}: binary external evaluation requires only 0/1 "
                "reference labels."
            )
        if not np.isfinite(scores).all():
            raise ValueError(
                f"{endpoint.name}: model scores must be finite before metrics are computed."
            )
        evaluable = len(targets) > 0 and len(np.unique(targets)) == 2
        results[endpoint.name] = {
            "score_key": endpoint.score_key,
            "source_label": endpoint.source_label,
            "relation_note": endpoint.relation_note,
            "positive_definition": f"Source annotation positive for {endpoint.source_label}",
            "n_evaluable": int(len(targets)),
            "n_positive": int(targets.sum()),
            "prevalence": float(targets.mean()) if len(targets) else None,
            "auroc": float(roc_auc_score(targets, scores)) if evaluable else None,
            "auroc_95_ci": _bootstrap_interval(
                targets, scores, "auroc", repetitions, seed + offset
            ),
            "average_precision": (
                float(average_precision_score(targets, scores)) if evaluable else None
            ),
            "average_precision_95_ci": _bootstrap_interval(
                targets, scores, "average_precision", repetitions, seed + 100 + offset
            ),
        }
        roc_inputs[endpoint.name] = (targets, scores)
    return results, roc_inputs


def _format_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _format_interval(value: list[float] | None) -> str:
    return "—" if value is None else f"[{value[0]:.3f}, {value[1]:.3f}]"


def _write_roc_figure(
    roc_inputs: Mapping[str, tuple[np.ndarray, np.ndarray]], title: str, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 5.6), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="0.65", linestyle="--", linewidth=1, label="Chance")
    plotted = 0
    for endpoint, (targets, scores) in roc_inputs.items():
        if len(targets) == 0 or len(np.unique(targets)) < 2:
            continue
        fpr, tpr, _ = roc_curve(targets, scores)
        axis.plot(fpr, tpr, linewidth=2, label=endpoint)
        plotted += 1
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    if plotted:
        axis.legend(loc="lower right", frameon=False)
    figure.savefig(destination, dpi=220)
    plt.close(figure)


def write_external_artifacts(
    *,
    dataset_name: str,
    title: str,
    endpoint_results: Mapping[str, Any],
    roc_inputs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    provenance: Mapping[str, Any],
    cohort: Mapping[str, Any],
    score_semantics: str,
    output_dir: Path,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    limitations: Iterable[str],
) -> None:
    """Write a dataset-scoped aggregate report and ROC figure.

    No image identifier, source URL, per-image score, or raw label is accepted
    by this writer. Callers may only publish the returned aggregate summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "dataset_name": dataset_name,
        "scope": "tool-level external dataset evaluation; not a full MedRAX-agent or clinical evaluation",
        "score_semantics": score_semantics,
        "provenance": dict(provenance),
        "cohort": dict(cohort),
        "bootstrap": {
            "repetitions": bootstrap_repetitions,
            "base_seed": bootstrap_seed,
            "endpoint_seed_scheme": (
                "AUROC=base_seed+endpoint_index; "
                "average_precision=base_seed+100+endpoint_index"
            ),
        },
        "primary_endpoint_results": dict(endpoint_results),
        "limitations": list(limitations),
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_roc_figure(roc_inputs, title, output_dir / "figures" / "primary_roc_curves.png")

    rows = []
    for endpoint, result in endpoint_results.items():
        rows.append(
            "| {endpoint} | {n} | {positive} | {prevalence} | {auroc} {auroc_ci} | "
            "{ap} {ap_ci} |".format(
                endpoint=endpoint,
                n=result["n_evaluable"],
                positive=result["n_positive"],
                prevalence=_format_score(result["prevalence"]),
                auroc=_format_score(result["auroc"]),
                auroc_ci=_format_interval(result["auroc_95_ci"]),
                ap=_format_score(result["average_precision"]),
                ap_ci=_format_interval(result["average_precision_95_ci"]),
            )
        )
    limitation_text = "\n".join(f"- {item}" for item in limitations)
    report = f"""# {dataset_name} results

Generated: {created_at}

This report evaluates the `ChestXRayClassifierTool` configuration used in
MedRAX, not the GPT-driven MedRAX agent or a clinical workflow. It contains
aggregate metrics only; no source images, identifiers, image URLs, row-level
labels, or row-level scores are committed.

## Direct endpoint results

| Endpoint | Evaluable n | Positive n | Prevalence | AUROC (95% bootstrap CI) | AP (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Intervals are {bootstrap_repetitions:,}-replicate percentile bootstrap intervals
over image-level observations, generated from base seed `{bootstrap_seed}`
(endpoint-specific offsets are deterministic). No decision threshold was
selected or tuned on this test set. Scores are {score_semantics.lower()}

## Interpretation boundaries

{limitation_text}

The machine-readable aggregate result is in
[aggregate_metrics.json](aggregate_metrics.json).
"""
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
