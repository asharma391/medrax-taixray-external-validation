#!/usr/bin/env python3
"""Run two aggregate-only external evaluations of MedRAX's CXR classifier.

The datasets are purposefully separate from the TAIX-Ray study:

* ChestDR: a 3,869-image testing partition published with the MedFMC
  few-shot benchmark and pinned to an immutable commit;
* a 624-image held-out pediatric pneumonia test set.

Inputs are streamed one at a time and immediately discarded. The script writes
only aggregate metrics and figures, never images, IDs, URLs, or row-level
predictions, to the repository output tree.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
import inspect
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Callable
from zipfile import BadZipFile

from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from remotezip import RemoteZip, RemoteZipError
from urllib3.exceptions import HTTPError as Urllib3HTTPError
from urllib3.util.retry import Retry

from medrax_taixray_audit.adapter import MedRAXClassifierAdapter
from medrax_taixray_audit.chestdr import (
    parse_published_csv,
    parse_test_manifest,
    validate_manifest_labels,
)
from medrax_taixray_audit.external import (
    BinaryEndpoint,
    evaluate_binary_predictions,
    write_external_artifacts,
)
from medrax_taixray_audit.pediatric import (
    PEDIATRIC_DATASET,
    PEDIATRIC_EXPECTED_LABEL_COUNTS,
    PEDIATRIC_EXPECTED_ROWS,
    PEDIATRIC_PARQUET_BYTES,
    PEDIATRIC_PARQUET_PATH,
    PEDIATRIC_PARQUET_SHA256,
    PEDIATRIC_PARQUET_URL,
    PEDIATRIC_REVISION,
    download_pinned_test_parquet,
    iter_pediatric_test_records,
)
from medrax_taixray_audit.provenance import (
    MEDRAX_COMMIT,
    MEDRAX_REPOSITORY,
    PATCH_FORK,
    SCORE_SEMANTICS,
    runtime_provenance,
)


TIMEOUT_SECONDS = 60
RETRIES = 6
REMOTE_CONNECT_TIMEOUT_SECONDS = 15

CHESTDR_FIGSHARE_ARTICLE = "https://doi.org/10.6084/m9.figshare.22302775.v1"
CHESTDR_ZIP_URL = "https://ndownloader.figshare.com/files/39673366"
CHESTDR_ZIP_MEMBER_CSV = "ChestDR/chestdr_published_19d.csv"
CHESTDR_MANIFEST_COMMIT = "3bfe04b153ab461baccbe574c9dc3dff84ab8b11"
CHESTDR_MANIFEST_REPOSITORY = "https://github.com/wllfore/MedFMC_fewshot_baseline"
CHESTDR_MANIFEST_CITATION = "https://doi.org/10.1038/s41597-023-02460-0"
CHESTDR_MANIFEST_URL = (
    "https://raw.githubusercontent.com/wllfore/MedFMC_fewshot_baseline/"
    f"{CHESTDR_MANIFEST_COMMIT}/data/ChestDR/test.txt"
)
CHESTDR_EXPECTED_TEST_ROWS = 3_869


CHESTDR_ENDPOINTS = (
    BinaryEndpoint(
        name="Effusion",
        score_key="Effusion",
        source_label="pleural_effusion",
        relation_note="Direct finding match: ChestDR pleural_effusion to XRV Effusion.",
    ),
    BinaryEndpoint(
        name="Cardiomegaly",
        score_key="Cardiomegaly",
        source_label="cardiomegaly",
        relation_note="Direct finding-name match.",
    ),
    BinaryEndpoint(
        name="Pneumonia",
        score_key="Pneumonia",
        source_label="pneumonia",
        relation_note="Direct finding-name match.",
    ),
    BinaryEndpoint(
        name="Atelectasis",
        score_key="Atelectasis",
        source_label="atelectasis",
        relation_note="Direct finding-name match.",
    ),
    BinaryEndpoint(
        name="Pneumothorax",
        score_key="Pneumothorax",
        source_label="pneumothorax",
        relation_note="Direct finding-name match.",
    ),
    BinaryEndpoint(
        name="Nodule",
        score_key="Nodule",
        source_label="nodule",
        relation_note="Direct finding-name match.",
    ),
    BinaryEndpoint(
        name="Edema",
        score_key="Edema",
        source_label="pulmonary_edema",
        relation_note="Direct finding match: ChestDR pulmonary_edema to XRV Edema.",
    ),
    BinaryEndpoint(
        name="Emphysema",
        score_key="Emphysema",
        source_label="emphysema",
        relation_note="Direct finding-name match.",
    ),
    BinaryEndpoint(
        name="Fibrosis",
        score_key="Fibrosis",
        source_label="fibrosis",
        relation_note="Direct finding-name match.",
    ),
    BinaryEndpoint(
        name="Pleural thickening",
        score_key="Pleural_Thickening",
        source_label="thickened_pleura",
        relation_note="Direct terminology match: ChestDR thickened_pleura to XRV Pleural_Thickening.",
    ),
)

PEDIATRIC_PNEUMONIA_ENDPOINTS = (
    BinaryEndpoint(
        name="Pneumonia",
        score_key="Pneumonia",
        source_label="Pneumonia",
        relation_note="Direct finding-name match to the held-out source binary label.",
    ),
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    title: str
    endpoints: tuple[BinaryEndpoint, ...]
    run: Callable[..., tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], tuple[str, ...]]]


def _get_text(url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        try:
            response = requests.get(
                url,
                timeout=TIMEOUT_SECONDS,
                headers={"User-Agent": "medrax-taixray-external-validation/0.1"},
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < RETRIES:
                time.sleep(2**attempt)
    raise RuntimeError(f"Request failed for {url}: {last_error}")


def _new_remotezip_session() -> requests.Session:
    """Create a bounded, retrying session for Figshare byte-range requests."""
    retry = Retry(
        total=RETRIES - 1,
        connect=RETRIES - 1,
        read=RETRIES - 1,
        status=RETRIES - 1,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class _RetryingRemoteZip:
    """Retry a stalled Figshare range request from a clean ZIP/session state.

    ``remotezip`` intentionally streams ZIP members.  A read timeout can happen
    while its response body is being consumed, after the HTTP request has
    succeeded, so the HTTP adapter alone cannot reliably retry it.  On each
    retry this wrapper closes the interrupted response, creates a fresh session
    and reconstructs the central directory before reading again.
    """

    _retryable_errors = (
        OSError,
        requests.RequestException,
        Urllib3HTTPError,
        RemoteZipError,
        BadZipFile,
    )

    def __init__(self, url: str, *, initial_buffer_size: int) -> None:
        self.url = url
        self.initial_buffer_size = initial_buffer_size
        self._session: requests.Session | None = None
        self._archive: RemoteZip | None = None

    def _close_current(self) -> None:
        if self._archive is not None:
            with suppress(Exception):
                self._archive.close()
            self._archive = None
        if self._session is not None:
            self._session.close()
            self._session = None

    def _ensure_open(self) -> RemoteZip:
        if self._archive is None:
            session = _new_remotezip_session()
            try:
                archive = RemoteZip(
                    self.url,
                    initial_buffer_size=self.initial_buffer_size,
                    session=session,
                    timeout=(REMOTE_CONNECT_TIMEOUT_SECONDS, TIMEOUT_SECONDS),
                    headers={"User-Agent": "medrax-taixray-external-validation/0.1"},
                )
            except Exception:
                session.close()
                raise
            self._session = session
            self._archive = archive
        return self._archive

    def _with_retries(self, operation: Callable[[RemoteZip], Any], operation_name: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(RETRIES):
            try:
                return operation(self._ensure_open())
            except self._retryable_errors as exc:
                last_error = exc
                self._close_current()
                if attempt + 1 < RETRIES:
                    delay = min(2**attempt, 30)
                    print(
                        "ChestDR remote ZIP: transient range-read failure; "
                        f"retrying {operation_name} in {delay}s "
                        f"({attempt + 2}/{RETRIES})."
                    )
                    time.sleep(delay)
        raise RuntimeError(
            f"ChestDR remote ZIP {operation_name} failed after {RETRIES} bounded attempts: "
            f"{last_error}"
        ) from last_error

    def read(self, member: str) -> bytes:
        return self._with_retries(lambda archive: archive.read(member), "a ZIP member")

    def namelist(self) -> list[str]:
        return self._with_retries(lambda archive: archive.namelist(), "the ZIP directory")

    def close(self) -> None:
        self._close_current()

    def __enter__(self) -> "_RetryingRemoteZip":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _checkpoint_path(checkpoint_dir: Path, key: str) -> Path:
    return checkpoint_dir / f"{key}_private_checkpoint.json"


def _load_checkpoint(path: Path, fingerprint: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"Checkpoint fingerprint mismatch at {path}; remove it rather than mixing runs."
        )
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise RuntimeError(f"Malformed private checkpoint at {path}.")
    return predictions


def _save_checkpoint(path: Path, fingerprint: str, predictions: list[dict[str, Any]]) -> None:
    """Persist only private, non-identifying restart state outside the repository."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_text(
        json.dumps({"fingerprint": fingerprint, "predictions": predictions}), encoding="utf-8"
    )
    temporary.replace(path)


def _source_sha256(path: Path) -> str:
    """Hash executable source so resumed predictions cannot span code revisions."""
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_resume_identity(
    adapter: MedRAXClassifierAdapter,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Bind a resumable run to weights, device, code, and runtime versions."""
    checkpoint_sha256 = adapter.model_file_sha256()
    if checkpoint_sha256 is None:
        raise RuntimeError("Unable to locate the cached XRV checkpoint for a resumable run.")
    adapter_source = inspect.getsourcefile(MedRAXClassifierAdapter)
    if adapter_source is None:
        raise RuntimeError("Unable to locate the MedRAX classifier adapter source file.")
    runtime = runtime_provenance()
    identity = {
        "medrax_commit": MEDRAX_COMMIT,
        "model_name": adapter.model_name,
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_device": str(adapter.device),
        "input_maxval": repr(adapter.input_maxval),
        "adapter_source_sha256": _source_sha256(Path(adapter_source)),
        "runner_source_sha256": _source_sha256(Path(__file__).resolve()),
        "torchxrayvision": str(runtime["torchxrayvision"]),
        "torch": str(runtime["torch"]),
        "torchvision": str(runtime["torchvision"]),
    }
    return identity, runtime


def _fingerprint(payload: dict[str, Any]) -> str:
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _assert_uint8_image(image_bytes: bytes) -> None:
    """Fail rather than silently inventing a scale for a high-bit-depth export."""
    with Image.open(BytesIO(image_bytes)) as image:
        if image.mode not in {"1", "L", "P", "RGB", "RGBA"}:
            raise RuntimeError(
                f"Expected an 8-bit image export, received Pillow mode {image.mode!r}. "
                "Declare an acquisition scale before evaluating higher-bit-depth inputs."
            )


def _predict_image_bytes(
    adapter: MedRAXClassifierAdapter, image_bytes: bytes, temporary_image: Path
) -> dict[str, float]:
    _assert_uint8_image(image_bytes)
    temporary_image.write_bytes(image_bytes)
    try:
        scores, input_maxval = adapter.predict(temporary_image)
    finally:
        temporary_image.unlink(missing_ok=True)
    if input_maxval != 255.0:
        raise RuntimeError(
            "This external runner expects an 8-bit image export; received declared "
            f"input scale {input_maxval}."
        )
    return scores


def _run_chestdr(
    adapter: MedRAXClassifierAdapter,
    checkpoint_dir: Path,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], tuple[str, ...]]:
    manifest = parse_test_manifest(_get_text(CHESTDR_MANIFEST_URL))
    if len(manifest) != CHESTDR_EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"ChestDR fixed test manifest has {len(manifest)} rows, expected "
            f"{CHESTDR_EXPECTED_TEST_ROWS}."
        )
    if limit is not None:
        if not 1 <= limit <= len(manifest):
            raise ValueError(f"--chestdr-limit must be between 1 and {len(manifest)}.")
        manifest = manifest[:limit]
    adapter_identity, runtime = _adapter_resume_identity(adapter)
    fingerprint = _fingerprint(
        {
            "dataset": "chestdr",
            "manifest_commit": CHESTDR_MANIFEST_COMMIT,
            "available_rows": CHESTDR_EXPECTED_TEST_ROWS,
            "n_requested": len(manifest),
            "adapter": adapter_identity,
        }
    )
    checkpoint = _checkpoint_path(checkpoint_dir, "chestdr")
    predictions = _load_checkpoint(checkpoint, fingerprint)
    if len(predictions) > len(manifest):
        raise RuntimeError("ChestDR checkpoint contains more predictions than the fixed manifest.")

    with _RetryingRemoteZip(CHESTDR_ZIP_URL, initial_buffer_size=1024 * 1024) as archive:
        published = parse_published_csv(archive.read(CHESTDR_ZIP_MEMBER_CSV).decode("utf-8"))
        validate_manifest_labels(manifest, published)
        names = set(archive.namelist())
        print(
            f"ChestDR: locked {len(manifest)} records from the pinned MedFMC "
            "testing manifest before model inference."
        )
        with tempfile.TemporaryDirectory(prefix="medrax_chestdr_") as tempdir:
            temporary_image = Path(tempdir) / "input.png"
            for ordinal, (image_id, labels) in enumerate(manifest, start=1):
                if ordinal <= len(predictions):
                    continue
                member = f"ChestDR/images/{image_id}.png"
                if member not in names:
                    raise RuntimeError(f"ChestDR ZIP does not contain expected image member {member}.")
                scores = _predict_image_bytes(adapter, archive.read(member), temporary_image)
                predictions.append({"labels": labels, "scores": scores})
                if ordinal % 25 == 0 or ordinal == len(manifest):
                    _save_checkpoint(checkpoint, fingerprint, predictions)
                    print(f"ChestDR: inferred {ordinal}/{len(manifest)} on {adapter.device}.")

    provenance = {
        "medrax_repository": MEDRAX_REPOSITORY,
        "medrax_commit": MEDRAX_COMMIT,
        "compatibility_patch": PATCH_FORK,
        "dataset_source": CHESTDR_FIGSHARE_ARTICLE,
        "dataset_public_zip": CHESTDR_ZIP_URL,
        "dataset_zip_member_csv": CHESTDR_ZIP_MEMBER_CSV,
        "test_manifest_repository": CHESTDR_MANIFEST_REPOSITORY,
        "test_manifest_commit": CHESTDR_MANIFEST_COMMIT,
        "test_manifest_url": CHESTDR_MANIFEST_URL,
        "test_manifest_origin": (
            "MedFMC few-shot baseline testing list: the remaining test images after its "
            "few-shot support-pool selection; pinned to the cited repository commit."
        ),
        "test_manifest_citation": CHESTDR_MANIFEST_CITATION,
        "adapter": {
            "device": str(adapter.device),
            "input_maxval": 255,
            "model_checkpoint_sha256": adapter_identity["checkpoint_sha256"],
            "adapter_source_sha256": adapter_identity["adapter_source_sha256"],
            "runner_source_sha256": adapter_identity["runner_source_sha256"],
        },
        "runtime": runtime,
    }
    cohort = {
        "fixed_test_rows_available": CHESTDR_EXPECTED_TEST_ROWS,
        "n_inferred": len(predictions),
        "selection": (
            "entire pinned MedFMC testing partition"
            if limit is None
            else "first records for a non-public smoke test; not a benchmark"
        ),
        "analysis_unit": "patient/image",
        "patient_grouping": "ChestDR releases one frontal radiograph for each distinct patient; image-level bootstrap is therefore patient-level for this cohort.",
        "raw_data_policy": "the remote ZIP is accessed by byte range; each PNG exists only in a temporary directory during its inference and is then deleted; aggregate results only are written to the repository",
    }
    limitations = (
        "ChestDR labels were initially derived from signed clinical reports by a radiology resident and confirmed by a senior radiologist; they are not a new blinded image re-read for this project.",
        "The partition is a pinned published MedFMC benchmark testing list, not claimed to be an official test split created by the original ChestDR dataset authors.",
        "Only direct finding-name/terminology matches to the MedRAX-selected TorchXRayVision output vocabulary are reported; no surrogate mapping is used for the remaining ChestDR labels.",
        "The result evaluates the classifier component only, not the GPT-driven MedRAX agent or a clinical workflow.",
        "External evaluation does not establish absence of every possible pretraining overlap, although ChestDR is absent from the classifier's declared source datasets.",
    )
    return predictions, provenance, cohort, limitations


def _run_pediatric_pneumonia(
    adapter: MedRAXClassifierAdapter,
    checkpoint_dir: Path,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], tuple[str, ...]]:
    if limit is not None:
        if not 1 <= limit <= PEDIATRIC_EXPECTED_ROWS:
            raise ValueError(
                f"--pediatric-limit must be between 1 and {PEDIATRIC_EXPECTED_ROWS}."
            )
    n_requested = limit if limit is not None else PEDIATRIC_EXPECTED_ROWS
    adapter_identity, runtime = _adapter_resume_identity(adapter)
    fingerprint = _fingerprint(
        {
            "dataset": "pediatric-pneumonia",
            "revision": PEDIATRIC_REVISION,
            "parquet_path": PEDIATRIC_PARQUET_PATH,
            "parquet_bytes": PEDIATRIC_PARQUET_BYTES,
            "parquet_sha256": PEDIATRIC_PARQUET_SHA256,
            "available_rows": PEDIATRIC_EXPECTED_ROWS,
            "expected_label_counts": PEDIATRIC_EXPECTED_LABEL_COUNTS,
            "n_requested": n_requested,
            "adapter": adapter_identity,
        }
    )
    checkpoint = _checkpoint_path(checkpoint_dir, "pediatric_pneumonia")
    predictions = _load_checkpoint(checkpoint, fingerprint)
    if len(predictions) > n_requested:
        raise RuntimeError("Pediatric checkpoint contains more predictions than the fixed test split.")

    print(
        "Pediatric pneumonia: acquiring the checksum-pinned 624-image source test file "
        "before inference."
    )
    with tempfile.TemporaryDirectory(prefix="medrax_pediatric_pneumonia_") as tempdir:
        parquet_path = download_pinned_test_parquet(Path(tempdir) / "test.parquet")
        temporary_image = Path(tempdir) / "input.jpg"
        print(
            f"Pediatric pneumonia: locked {n_requested} held-out source test records "
            "before inference."
        )
        for ordinal, (source_label, image_bytes) in enumerate(
            iter_pediatric_test_records(parquet_path), start=1
        ):
            if limit is not None and ordinal > limit:
                break
            if ordinal <= len(predictions):
                continue
            scores = _predict_image_bytes(adapter, image_bytes, temporary_image)
            predictions.append({"labels": {"Pneumonia": source_label}, "scores": scores})
            if ordinal % 25 == 0 or ordinal == n_requested:
                _save_checkpoint(checkpoint, fingerprint, predictions)
                print(f"Pediatric pneumonia: inferred {ordinal}/{n_requested} on {adapter.device}.")

    provenance = {
        "medrax_repository": MEDRAX_REPOSITORY,
        "medrax_commit": MEDRAX_COMMIT,
        "compatibility_patch": PATCH_FORK,
        "dataset_source": f"https://huggingface.co/datasets/{PEDIATRIC_DATASET}",
        "dataset_citation": "https://doi.org/10.17632/rscbjbr9sj.2",
        "public_acquisition_revision": PEDIATRIC_REVISION,
        "public_acquisition_split": "test",
        "pinned_parquet_path": PEDIATRIC_PARQUET_PATH,
        "pinned_parquet_url": PEDIATRIC_PARQUET_URL,
        "pinned_parquet_bytes": PEDIATRIC_PARQUET_BYTES,
        "pinned_parquet_sha256": PEDIATRIC_PARQUET_SHA256,
        "source_label_convention": {"0": "NORMAL", "1": "PNEUMONIA"},
        "adapter": {
            "device": str(adapter.device),
            "input_maxval": 255,
            "model_checkpoint_sha256": adapter_identity["checkpoint_sha256"],
            "adapter_source_sha256": adapter_identity["adapter_source_sha256"],
            "runner_source_sha256": adapter_identity["runner_source_sha256"],
        },
        "runtime": runtime,
    }
    cohort = {
        "fixed_test_rows_available": PEDIATRIC_EXPECTED_ROWS,
        "n_inferred": len(predictions),
        "selection": (
            "entire checksum-pinned held-out source test split"
            if limit is None
            else "first records for a non-public smoke test; not a benchmark"
        ),
        "analysis_unit": "image",
        "patient_grouping": "not exposed by the public acquisition view; no patient-level claim",
        "raw_data_policy": "the checksum-verified source Parquet and each decoded image exist only in a temporary directory during inference and are deleted on exit; aggregate results only are written to the repository",
    }
    limitations = (
        "This is a narrow binary pediatric domain-transfer test (normal versus pneumonia), not an adult multi-finding clinical validation.",
        "The public acquisition view does not expose patient grouping, so bootstrap intervals are image-level rather than patient-clustered.",
        "The pinned source convention is 0=NORMAL and 1=PNEUMONIA; its 234 normal and 390 pneumonia test labels are used as released.",
        "The source label is used as released; no model threshold was selected or tuned on this test split.",
        "The result evaluates the classifier component only, not the GPT-driven MedRAX agent or a clinical workflow.",
    )
    return predictions, provenance, cohort, limitations


CHESTDR = DatasetSpec(
    key="chestdr",
    display_name="ChestDR: pinned MedFMC testing partition",
    title="MedRAX classifier: ChestDR pinned MedFMC testing partition",
    endpoints=CHESTDR_ENDPOINTS,
    run=_run_chestdr,
)
PEDIATRIC_PNEUMONIA = DatasetSpec(
    key="pediatric_pneumonia",
    display_name="Pediatric pneumonia held-out test set",
    title="MedRAX classifier: pediatric pneumonia held-out test set",
    endpoints=PEDIATRIC_PNEUMONIA_ENDPOINTS,
    run=_run_pediatric_pneumonia,
)
SPECS = {spec.key: spec for spec in (CHESTDR, PEDIATRIC_PNEUMONIA)}
# Dataset-specific offsets keep each interval reproducible even if the CLI
# dataset order changes or a single cohort is rerun independently.
DATASET_SEED_OFFSETS = {"chestdr": 0, "pediatric_pneumonia": 10_000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(SPECS),
        default=tuple(SPECS),
        help="Dataset-scoped evaluations to run (default: both).",
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/external"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/tmp/medrax_external_checkpoints"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--chestdr-limit",
        type=int,
        help="Non-public ChestDR smoke-test cap; never present a limited run as full-test results.",
    )
    parser.add_argument(
        "--pediatric-limit",
        type=int,
        help="Non-public pediatric smoke-test cap; never present a limited run as test-set results.",
    )
    args = parser.parse_args()
    if args.bootstrap_repetitions < 1:
        raise ValueError("--bootstrap-repetitions must be positive.")
    adapter = MedRAXClassifierAdapter(device=args.device)
    limits = {
        "chestdr": args.chestdr_limit,
        "pediatric_pneumonia": args.pediatric_limit,
    }
    for key in args.datasets:
        spec = SPECS[key]
        predictions, provenance, cohort, limitations = spec.run(
            adapter, args.checkpoint_dir, limits[key]
        )
        bootstrap_seed = args.seed + DATASET_SEED_OFFSETS[key]
        endpoint_results, roc_inputs = evaluate_binary_predictions(
            predictions,
            spec.endpoints,
            repetitions=args.bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        write_external_artifacts(
            dataset_name=spec.display_name,
            title=spec.title,
            endpoint_results=endpoint_results,
            roc_inputs=roc_inputs,
            provenance=provenance,
            cohort=cohort,
            score_semantics=SCORE_SEMANTICS,
            output_dir=args.output_root / spec.key,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_seed=bootstrap_seed,
            limitations=limitations,
        )
        if limits[key] is None:
            _checkpoint_path(args.checkpoint_dir, key).unlink(missing_ok=True)
        print(f"{spec.display_name}: wrote aggregate-only artifacts to {args.output_root / spec.key}.")


if __name__ == "__main__":
    main()
