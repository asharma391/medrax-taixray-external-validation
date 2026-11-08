"""Pinned, temporary transport for the pediatric pneumonia test set.

The Hugging Face dataset-server row endpoint is convenient for exploratory
use, but a long external evaluation should not depend on hundreds of
short-lived per-image URLs.  This module instead fetches the single test
Parquet object from a specific dataset commit, verifies its published LFS
SHA-256, and exposes each embedded image only while it is being inferred.
"""

from __future__ import annotations

from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
import time
from typing import Any

import requests


PEDIATRIC_DATASET = "hf-vision/chest-xray-pneumonia"
PEDIATRIC_REVISION = "c1a67c18df1a52ec332ffc00418f4eaa61a9f2bc"
PEDIATRIC_PARQUET_PATH = "data/test-00000-of-00001.parquet"
PEDIATRIC_PARQUET_URL = (
    "https://huggingface.co/datasets/"
    f"{PEDIATRIC_DATASET}/resolve/{PEDIATRIC_REVISION}/{PEDIATRIC_PARQUET_PATH}"
    "?download=true"
)
PEDIATRIC_PARQUET_BYTES = 78_220_681
PEDIATRIC_PARQUET_SHA256 = "56c173b527abda58ca851a3ecfc094174b5ac9a0181204bcab4f66e999c65cfc"
PEDIATRIC_EXPECTED_ROWS = 624
PEDIATRIC_EXPECTED_LABEL_COUNTS = {0: 234, 1: 390}

USER_AGENT = "medrax-taixray-external-validation/0.1"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_pinned_test_parquet(
    destination: Path,
    *,
    attempts: int = 6,
    timeout: tuple[int, int] = (15, 60),
    session: requests.Session | None = None,
) -> Path:
    """Download the immutable test file with bounded retries and range resume.

    ``destination`` is expected to be inside a caller-owned temporary
    directory.  Only a completed object with the pinned byte count and LFS
    SHA-256 is returned.  Interrupted ``.part`` data is resumed when the CDN
    honors Range and restarted safely when it does not.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    requester: Any = session or requests
    last_error: Exception | None = None

    for attempt in range(attempts):
        existing = partial.stat().st_size if partial.exists() else 0
        if existing > PEDIATRIC_PARQUET_BYTES:
            partial.unlink()
            existing = 0
        headers = {
            "User-Agent": USER_AGENT,
            "Range": f"bytes={existing}-",
        }
        try:
            with requester.get(
                PEDIATRIC_PARQUET_URL,
                stream=True,
                timeout=timeout,
                headers=headers,
            ) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                resumed = existing > 0
                if resumed and (
                    response.status_code != 206
                    or not content_range.startswith(f"bytes {existing}-")
                ):
                    # A proxy that ignored Range would otherwise append the
                    # entire object and corrupt the temporary Parquet file.
                    partial.unlink(missing_ok=True)
                    raise RuntimeError("CDN did not honor the requested resume range.")
                mode = "ab" if existing else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            size = partial.stat().st_size
            if size != PEDIATRIC_PARQUET_BYTES:
                raise RuntimeError(
                    "Pinned pediatric Parquet transfer ended at "
                    f"{size} bytes, expected {PEDIATRIC_PARQUET_BYTES}."
                )
            observed_sha256 = _sha256_file(partial)
            if observed_sha256 != PEDIATRIC_PARQUET_SHA256:
                partial.unlink(missing_ok=True)
                raise RuntimeError(
                    "Pinned pediatric Parquet checksum mismatch: "
                    f"expected {PEDIATRIC_PARQUET_SHA256}, got {observed_sha256}."
                )
            partial.replace(destination)
            return destination
        except (requests.RequestException, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"Pinned pediatric test Parquet failed after {attempts} bounded attempts: {last_error}"
    ) from last_error


def _open_validated_parquet(parquet_path: Path) -> Any:
    """Open and validate the pinned test schema and aggregate label totals."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    if parquet.metadata.num_rows != PEDIATRIC_EXPECTED_ROWS:
        raise RuntimeError(
            "Pinned pediatric test Parquet has "
            f"{parquet.metadata.num_rows} rows, expected {PEDIATRIC_EXPECTED_ROWS}."
        )
    expected_fields = {"image", "label"}
    if not expected_fields.issubset(parquet.schema_arrow.names):
        raise RuntimeError("Pinned pediatric test Parquet does not expose image and label columns.")

    label_counts = {0: 0, 1: 0}
    for batch in parquet.iter_batches(batch_size=256, columns=["label"]):
        for source_label in batch.column(0).to_pylist():
            source_label = int(source_label)
            if source_label not in label_counts:
                raise RuntimeError(
                    f"Unexpected source label {source_label} in pinned pediatric test Parquet."
                )
            label_counts[source_label] += 1
    if label_counts != PEDIATRIC_EXPECTED_LABEL_COUNTS:
        raise RuntimeError(
            "Pinned pediatric test label totals changed: "
            f"labels={label_counts}; expected {PEDIATRIC_EXPECTED_LABEL_COUNTS}."
        )
    return parquet


def iter_pediatric_test_records(parquet_path: Path) -> Iterator[tuple[int, bytes]]:
    """Yield ordered ``(0/1 label, image bytes)`` records from verified Parquet.

    File paths embedded in the source image feature are intentionally ignored,
    so the runner never persists or publishes them.  The expected row and label
    totals turn an accidental train split, changed schema, or label inversion
    into a hard failure rather than a silent benchmark change.
    """
    parquet = _open_validated_parquet(parquet_path)

    observed_rows = 0
    for batch in parquet.iter_batches(batch_size=16, columns=["image", "label"]):
        for row in batch.to_pylist():
            source_label = int(row["label"])
            if source_label not in PEDIATRIC_EXPECTED_LABEL_COUNTS:
                raise RuntimeError(
                    f"Unexpected source label {source_label} in pinned pediatric test Parquet."
                )
            image = row["image"]
            image_bytes = image.get("bytes") if isinstance(image, dict) else None
            if not isinstance(image_bytes, bytes) or not image_bytes:
                raise RuntimeError("Pinned pediatric test Parquet contains a missing image byte payload.")
            observed_rows += 1
            yield source_label, image_bytes

    if observed_rows != PEDIATRIC_EXPECTED_ROWS:
        raise RuntimeError(
            "Pinned pediatric test row total changed while iterating image payloads: "
            f"{observed_rows}, expected {PEDIATRIC_EXPECTED_ROWS}."
        )
