"""Deterministic, one-image-per-patient pilot sampling without saved IDs."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable

from .dataset import TAIXRecord


def candidate_indices(total_rows: int, seed: int) -> Iterable[int]:
    """Yield a deterministic permutation of public test-row indices."""
    indices = list(range(total_rows))
    random.Random(seed).shuffle(indices)
    return indices


def select_unique_patient_records(
    fetch_row: Callable[[int], TAIXRecord],
    total_rows: int,
    sample_size: int,
    seed: int,
) -> list[TAIXRecord]:
    """Select a fixed-size row sample while retaining one image per patient.

    Patient pseudonyms exist only in memory. Callers should persist only the
    selected row indices and aggregate results, never the identifiers.
    """
    if not 1 <= sample_size <= total_rows:
        raise ValueError("sample_size must be between 1 and total_rows.")

    seen_patients: set[str] = set()
    selected: list[TAIXRecord] = []
    for row_idx in candidate_indices(total_rows, seed):
        record = fetch_row(row_idx)
        if record.patient_id in seen_patients:
            continue
        seen_patients.add(record.patient_id)
        selected.append(record)
        if len(selected) == sample_size:
            return selected
    raise RuntimeError("Not enough unique patients were available for the requested sample.")


def select_two_stage_patient_pilot(
    fetch_page: Callable[[int], list[TAIXRecord]],
    total_rows: int,
    sample_size: int,
    seed: int,
    page_size: int = 100,
    page_budget: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[TAIXRecord], dict[str, int]]:
    """Select a deterministic two-stage, one-image-per-patient pilot.

    Stage one samples pages uniformly from the official split; stage two
    shuffles the resulting one-image-per-patient pool before selecting the
    fixed-size pilot. It reduces requests to the public data viewer while
    retaining spread across randomly chosen parts of the predefined test split.
    The design is explicitly reported as a pilot, not a full test-set estimate.
    """
    if not 1 <= sample_size <= total_rows:
        raise ValueError("sample_size must be between 1 and total_rows.")
    if page_size < 1 or page_budget < 1:
        raise ValueError("page_size and page_budget must be positive.")

    rng = random.Random(seed)
    page_offsets = list(range(0, total_rows, page_size))
    rng.shuffle(page_offsets)
    selected_by_patient: dict[str, TAIXRecord] = {}
    pages_examined = 0
    candidate_rows = 0

    for offset in page_offsets:
        records = fetch_page(offset)
        pages_examined += 1
        candidate_rows += len(records)
        rng.shuffle(records)
        for record in records:
            selected_by_patient.setdefault(record.patient_id, record)
        if progress is not None:
            progress(pages_examined, len(selected_by_patient))
        if pages_examined >= page_budget and len(selected_by_patient) >= sample_size:
            break

    pool = list(selected_by_patient.values())
    if len(pool) < sample_size:
        raise RuntimeError("Not enough unique patients were available for the requested sample.")
    rng.shuffle(pool)
    return pool[:sample_size], {
        "pages_examined": pages_examined,
        "candidate_rows": candidate_rows,
        "unique_patients_available": len(pool),
    }
