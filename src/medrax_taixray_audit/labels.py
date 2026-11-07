"""Pre-specified TAIX-Ray-to-XRV endpoint mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Endpoint:
    """One deliberately narrow external endpoint."""

    name: str
    score_key: str
    label_columns: tuple[str, ...]
    relation_note: str


PRIMARY_ENDPOINTS = (
    Endpoint(
        name="Atelectasis",
        score_key="Atelectasis",
        label_columns=("Atelectasis_Left", "Atelectasis_Right"),
        relation_note="Direct finding-name match; bilateral TAIX grades are aggregated by max.",
    ),
    Endpoint(
        name="Effusion",
        score_key="Effusion",
        label_columns=("PleuralEffusion_Left", "PleuralEffusion_Right"),
        relation_note="Direct finding-name match; bilateral TAIX grades are aggregated by max.",
    ),
    Endpoint(
        name="Lung Opacity",
        score_key="Lung Opacity",
        label_columns=("PulmonaryOpacities_Left", "PulmonaryOpacities_Right"),
        relation_note="Near-direct terminology match; bilateral TAIX grades are aggregated by max.",
    ),
    Endpoint(
        name="Cardiomegaly",
        score_key="Cardiomegaly",
        label_columns=("HeartSize",),
        relation_note="Related phenotype: TAIX HeartSize >=2 means enlarged or massively enlarged.",
    ),
)

EXPLORATORY_ENDPOINT = Endpoint(
    name="Edema proxy: pulmonary congestion",
    score_key="Edema",
    label_columns=("PulmonaryCongestion",),
    relation_note="Semantic proxy only; not pooled with the primary direct/near-direct endpoints.",
)


def aggregate_grade(labels: Mapping[str, int], endpoint: Endpoint) -> int:
    """Return the largest relevant ordinal grade for an endpoint."""
    return max(int(labels[column]) for column in endpoint.label_columns)


def binary_target(
    labels: Mapping[str, int], endpoint: Endpoint, positive_threshold: int = 2
) -> Optional[int]:
    """Return a conservative binary target or None for an uncertain report grade.

    TAIX grades use 0 for absent, 1 for questionable, and 2--4 for increasingly
    definite findings. The primary definition therefore treats 0 as negative,
    2--4 as positive, and excludes the questionable grade rather than labeling
    it negative. Sensitivity definitions can change ``positive_threshold``.
    """
    grade = aggregate_grade(labels, endpoint)
    if grade >= positive_threshold:
        return 1
    if grade == 0:
        return 0
    return None
