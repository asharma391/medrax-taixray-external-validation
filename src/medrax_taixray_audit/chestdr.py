"""Parsing and validation helpers for the public ChestDR evaluation manifest."""

from __future__ import annotations

import csv
from io import StringIO
from typing import Mapping


CHESTDR_LABEL_COLUMNS = (
    "pleural_effusion",
    "nodule",
    "pneumonia",
    "cardiomegaly",
    "hilar_enlargement",
    "fracture_old",
    "fibrosis",
    "aortic_calcification",
    "tortuous_aorta",
    "thickened_pleura",
    "TB",
    "pneumothorax",
    "emphysema",
    "atelectasis",
    "calcification",
    "pulmonary_edema",
    "increased_lung_markings",
    "elevated_diaphragm",
    "consolidation",
)


def _binary_labels(values: list[str], source: str) -> dict[str, int]:
    if len(values) != len(CHESTDR_LABEL_COLUMNS):
        raise ValueError(
            f"{source}: expected {len(CHESTDR_LABEL_COLUMNS)} labels, got {len(values)}."
        )
    labels = {column: int(value) for column, value in zip(CHESTDR_LABEL_COLUMNS, values)}
    if not set(labels.values()).issubset({0, 1}):
        raise ValueError(f"{source}: source labels must be binary.")
    return labels


def parse_test_manifest(text: str) -> list[tuple[str, dict[str, int]]]:
    """Parse MedFMC's fixed ChestDR test list without persisting image IDs."""
    records: list[tuple[str, dict[str, int]]] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            image_id, comma_values = stripped.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Manifest line {line_no} is malformed.") from exc
        if image_id in seen_ids:
            raise ValueError(f"Manifest has duplicate image ID at line {line_no}.")
        seen_ids.add(image_id)
        records.append(
            (image_id, _binary_labels(comma_values.split(","), f"manifest line {line_no}"))
        )
    return records


def parse_published_csv(text: str) -> dict[str, dict[str, int]]:
    """Parse the official ChestDR CSV packaged in the immutable Figshare ZIP."""
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None or "img_id" not in reader.fieldnames:
        raise ValueError("ChestDR CSV does not contain an img_id column.")
    missing = set(CHESTDR_LABEL_COLUMNS).difference(reader.fieldnames)
    if missing:
        raise ValueError(f"ChestDR CSV missing label columns: {sorted(missing)}")
    records: dict[str, dict[str, int]] = {}
    for row_no, row in enumerate(reader, start=2):
        image_id = str(row["img_id"])
        # The immutable CSV uses ``<id>.png`` while MedFMC's fixed split uses
        # the same identifier without the suffix. Normalize only that known
        # export convention before cross-validating labels.
        if image_id.endswith(".png"):
            image_id = image_id.removesuffix(".png")
        if not image_id or image_id in records:
            raise ValueError(f"ChestDR CSV duplicate/empty img_id at row {row_no}.")
        records[image_id] = _binary_labels(
            [str(row[column]) for column in CHESTDR_LABEL_COLUMNS], f"CSV row {row_no}"
        )
    return records


def validate_manifest_labels(
    manifest: list[tuple[str, Mapping[str, int]]],
    published: Mapping[str, Mapping[str, int]],
) -> None:
    """Require labels in the fixed split manifest to equal official CSV labels."""
    for image_id, labels in manifest:
        source_labels = published.get(image_id)
        if source_labels is None:
            raise ValueError(f"Manifest image ID missing from official ChestDR CSV: {image_id}")
        if dict(labels) != dict(source_labels):
            raise ValueError(f"Manifest/CSV label disagreement for ChestDR image {image_id}.")
