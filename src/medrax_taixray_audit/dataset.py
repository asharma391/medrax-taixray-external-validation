"""Minimal, data-policy-aware access to the public TAIX-Ray dataset server."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import os
import time
from typing import Any, Iterable

import requests

from .provenance import TAIXRAY_HF_REVISION, TAIXRAY_SPLIT


ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
LABEL_COLUMNS = (
    "HeartSize",
    "PulmonaryCongestion",
    "PleuralEffusion_Right",
    "PleuralEffusion_Left",
    "PulmonaryOpacities_Right",
    "PulmonaryOpacities_Left",
    "Atelectasis_Right",
    "Atelectasis_Left",
)


@dataclass(frozen=True)
class TAIXRecord:
    """One public TAIX-Ray row; identifier fields remain in-memory only."""

    row_idx: int
    patient_id: str
    image_url: str
    labels: dict[str, int]

    def manifest_entry(self, image_sha256: str) -> dict[str, Any]:
        """Return a non-identifying, reproducibility-oriented record."""
        return {
            "row_idx": self.row_idx,
            "labels": self.labels,
            "image_sha256": image_sha256,
        }


class TAIXRayClient:
    """Read public metadata/images without persisting signed URLs or IDs."""

    def __init__(
        self,
        revision: str = TAIXRAY_HF_REVISION,
        timeout_seconds: int = 60,
        retries: int = 8,
        min_request_interval_seconds: float = 1.1,
    ) -> None:
        self.revision = revision
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.min_request_interval_seconds = min_request_interval_seconds
        self._last_dataset_request_at = 0.0
        self._page_cache: dict[int, dict[int, TAIXRecord]] = {}

    def _request_json(self, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                elapsed = time.monotonic() - self._last_dataset_request_at
                if elapsed < self.min_request_interval_seconds:
                    time.sleep(self.min_request_interval_seconds - elapsed)
                response = requests.get(
                    ROWS_ENDPOINT,
                    params=params,
                    timeout=self.timeout_seconds,
                )
                self._last_dataset_request_at = time.monotonic()
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after is not None else 5.0 * (attempt + 1)
                    except ValueError:
                        delay = 5.0 * (attempt + 1)
                    time.sleep(min(delay, 120.0))
                    last_error = requests.HTTPError("429 Too Many Requests")
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"TAIX-Ray dataset-server request failed: {last_error}")

    def _parse_row(self, item: dict[str, Any]) -> TAIXRecord:
        row = item["row"]
        labels = {column: int(row[column]) for column in LABEL_COLUMNS}
        return TAIXRecord(
            row_idx=int(item["row_idx"]),
            patient_id=str(row["PatientID"]),
            image_url=str(row["Image"]["src"]),
            labels=labels,
        )

    def _fetch_page(self, offset: int, length: int = 100) -> dict[int, TAIXRecord]:
        """Fetch and cache up to 100 public test-split rows at the pinned revision."""
        if offset in self._page_cache:
            return self._page_cache[offset]
        payload = self._request_json(
            {
                "dataset": "TLAIM/TAIX-Ray",
                "config": "default",
                "split": TAIXRAY_SPLIT,
                "offset": offset,
                "length": length,
                "revision": self.revision,
            }
        )
        page = {int(item["row_idx"]): self._parse_row(item) for item in payload.get("rows", [])}
        self._page_cache[offset] = page
        return page

    def fetch_row(self, row_idx: int) -> TAIXRecord:
        """Fetch a public test-split row, caching its 100-row server page."""
        page_offset = (row_idx // 100) * 100
        page = self._fetch_page(page_offset)
        if row_idx not in page:
            raise RuntimeError(f"Dataset server did not return expected row {row_idx}.")
        return page[row_idx]

    def fetch_page(self, offset: int, length: int = 100) -> list[TAIXRecord]:
        """Return one cached public server page in its original row order."""
        return list(self._fetch_page(offset, length).values())

    def iter_split(self, page_size: int = 100) -> Iterable[TAIXRecord]:
        """Yield the pinned public test split in pages for a remote full run."""
        if not 1 <= page_size <= 100:
            raise ValueError("The dataset server supports page sizes from 1 through 100.")
        offset = 0
        while True:
            page = self._fetch_page(offset, page_size)
            if not page:
                return
            yield from page.values()
            offset += len(page)
            if len(page) < page_size:
                return

    def download_image(self, record: TAIXRecord, destination: Path) -> str:
        """Download one image atomically and return its SHA-256 digest."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        digest = sha256()
        try:
            with requests.get(
                record.image_url,
                stream=True,
                timeout=self.timeout_seconds,
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            digest.update(chunk)
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return digest.hexdigest()
