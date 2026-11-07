"""Aggregate-only research artifacts: JSON, Markdown, and a ROC figure."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import roc_coordinates
from .provenance import (
    MEDRAX_COMMIT,
    MEDRAX_REPOSITORY,
    PATCH_FORK,
    SCORE_SEMANTICS,
    TAIXRAY_DATASET,
    TAIXRAY_DOI,
    TAIXRAY_HF_REVISION,
)


def _format_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _format_interval(value: list[float] | None) -> str:
    return "—" if value is None else f"[{value[0]:.3f}, {value[1]:.3f}]"


def write_roc_figure(
    plotting_inputs: dict[str, tuple[Any, Any]], destination: Path
) -> None:
    """Write a publication-style aggregate ROC plot with no image-level data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 5.6), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="0.65", linestyle="--", linewidth=1, label="Chance")
    plotted = 0
    for endpoint, (targets, scores) in plotting_inputs.items():
        coordinates = roc_coordinates(targets, scores)
        if coordinates is None:
            continue
        fpr, tpr = coordinates
        axis.plot(fpr, tpr, linewidth=2, label=endpoint)
        plotted += 1
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="MedRAX classifier: TAIX-Ray external pilot (strict labels)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    if plotted:
        axis.legend(loc="lower right", frameon=False)
    figure.savefig(destination, dpi=220)
    plt.close(figure)


def write_aggregate_artifacts(
    endpoint_results: dict[str, Any],
    plotting_inputs: dict[str, tuple[Any, Any]],
    run_metadata: dict[str, Any],
    output_dir: Path,
    sensitivity_analyses: dict[str, dict[str, Any]] | None = None,
    encoding_stratified: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Persist only aggregate output and a human-readable, bounded report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "scope": "tool-level external-validation pilot; not a full MedRAX-agent or clinical evaluation",
        "score_semantics": SCORE_SEMANTICS,
        "provenance": {
            "medrax_repository": MEDRAX_REPOSITORY,
            "medrax_commit": MEDRAX_COMMIT,
            "compatibility_patch": PATCH_FORK,
            "taixray_dataset": TAIXRAY_DATASET,
            "taixray_doi": TAIXRAY_DOI,
            "taixray_hf_revision": TAIXRAY_HF_REVISION,
        },
        "run_metadata": run_metadata,
        "primary_endpoint_results": endpoint_results,
        "sensitivity_analyses": sensitivity_analyses or {},
        "encoding_stratified_results": encoding_stratified or {},
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_roc_figure(plotting_inputs, output_dir / "figures" / "primary_roc_curves.png")

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

    sensitivity_rows = []
    for endpoint in endpoint_results:
        sensitivity_rows.append(
            "| {endpoint} | {strict} | {inclusive} | {moderate} |".format(
                endpoint=endpoint,
                strict=_format_score(endpoint_results[endpoint]["auroc"]),
                inclusive=_format_score(
                    (sensitivity_analyses or {})
                    .get("grade_at_least_1", {})
                    .get(endpoint, {})
                    .get("auroc")
                ),
                moderate=_format_score(
                    (sensitivity_analyses or {})
                    .get("grade_at_least_3", {})
                    .get(endpoint, {})
                    .get("auroc")
                ),
            )
        )

    encoding_rows = []
    for endpoint in endpoint_results:
        low_range = (encoding_stratified or {}).get("4095.0", {}).get(endpoint, {})
        wide_range = (encoding_stratified or {}).get("65535.0", {}).get(endpoint, {})
        encoding_rows.append(
            "| {endpoint} | {low} {low_ci} | {wide} {wide_ci} |".format(
                endpoint=endpoint,
                low=_format_score(low_range.get("auroc")),
                low_ci=_format_interval(low_range.get("auroc_95_ci")),
                wide=_format_score(wide_range.get("auroc")),
                wide_ci=_format_interval(wide_range.get("auroc_95_ci")),
            )
        )

    report = f"""# TAIX-Ray external-validation pilot results

Generated: {created_at}

This report evaluates only the `ChestXRayClassifierTool` configuration used in
MedRAX, not the GPT-driven MedRAX agent or a clinical workflow. The test cohort
is a deterministic, one-image-per-patient sample from TAIX-Ray's public test
split; no images, patient identifiers, image URLs, or row-level labels are
committed to this repository.

## Pre-specified primary endpoints

| Endpoint | Evaluable n | Positive n | Prevalence | AUROC (95% bootstrap CI) | AP (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Primary labels treat TAIX's grade 0 as absent and grades 2--4 as definite;
grade 1 (questionable) is excluded. Intervals are 2,000-replicate percentile
bootstrap intervals over the one-image-per-patient pilot observations.

## Pre-specified label-threshold sensitivity

| Endpoint | Definite (>=2) AUROC | Include questionable (>=1) AUROC | Moderate/severe (>=3) AUROC |
| --- | ---: | ---: | ---: |
{chr(10).join(sensitivity_rows)}

The score threshold is never tuned on this test pilot. These label-threshold
analyses test whether the rank-ordering result is sensitive to how the ordinal
radiologist grades are dichotomized.

## Image-encoding diagnostic

The public TAIX-Ray PNG export contained two observed uint16 range encodings;
the predeclared adapter used 4095 for the 0--4095 range and 65535 for the
wider range. This post-run stratification is descriptive only: the two groups
may also differ in acquisition, patients, or other unobserved factors, so it
does **not** establish that intensity encoding caused a performance difference.

| Endpoint | 0--4095 stratum AUROC (95% CI) | wider uint16 stratum AUROC (95% CI) |
| --- | ---: | ---: |
{chr(10).join(encoding_rows)}

## Interpretation boundaries

- Scores are {SCORE_SEMANTICS.lower()}
- TAIX labels are structured routine radiologist-report grades, not a consensus
  image re-read reference standard.
- This is an external-site pilot, not a full 42,928-image benchmark and not a
  clinical validation. It does not establish absence of all possible pretraining
  overlap.
- The published 16-bit compatibility patch is linked at {PATCH_FORK}; it makes
  the required acquisition scale explicit rather than performing unsafe
  per-image contrast scaling.

See [the protocol](../docs/protocol.md) for frozen revisions, label mapping,
data policy, and the full set of limitations. The aggregate machine-readable
results are in [aggregate_metrics.json](aggregate_metrics.json).
"""
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
