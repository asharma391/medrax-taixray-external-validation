"""Command-line runner for the data-free, reproducible TAIX-Ray audit."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .adapter import MedRAXClassifierAdapter, TAIXRAY_UINT16_RULE
from .dataset import TAIXRayClient
from .metrics import evaluate_predictions
from .provenance import (
    MEDRAX_COMMIT,
    SCORE_SEMANTICS,
    TAIXRAY_HF_REVISION,
    TAIXRAY_TEST_ROWS,
    runtime_provenance,
)
from .reporting import write_aggregate_artifacts
from .sampling import select_two_stage_patient_pilot


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_pilot(
    sample_size: int, seed: int, cache_dir: Path, manifest_path: Path, page_budget: int = 64
) -> dict[str, Any]:
    """Select, download, and locally manifest a data-policy-compliant pilot."""
    client = TAIXRayClient()
    def page_progress(pages_examined: int, unique_patients: int) -> None:
        if pages_examined % 8 == 0:
            print(
                f"Examined {pages_examined} random pages; found {unique_patients} "
                "unique in-memory patient pseudonyms."
            )

    selected, sampling_details = select_two_stage_patient_pilot(
        client.fetch_page,
        TAIXRAY_TEST_ROWS,
        sample_size,
        seed,
        page_budget=page_budget,
        progress=page_progress,
    )
    entries = []
    images_dir = cache_dir / "images"
    for ordinal, record in enumerate(selected, start=1):
        destination = images_dir / f"{record.row_idx}.png"
        if destination.exists():
            image_digest = _file_sha256(destination)
        else:
            image_digest = client.download_image(record, destination)
        entries.append(record.manifest_entry(image_digest))
        if ordinal % 25 == 0 or ordinal == sample_size:
            print(f"Prepared {ordinal}/{sample_size} image records.")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_revision": TAIXRAY_HF_REVISION,
        "test_rows_total": TAIXRAY_TEST_ROWS,
        "sampling": {
            "method": (
                "two-stage random-page pilot; one retained image per in-memory patient "
                "pseudonym, then a random ordering for final selection"
            ),
            "seed": seed,
            "n_requested": sample_size,
            "n_selected": len(entries),
            **sampling_details,
        },
        "image_encoding": {
            "input_maxval": TAIXRAY_UINT16_RULE,
            "assumption": (
                "TAIX-Ray public PNGs contain observed 12-bit-valued and wider-range uint16 "
                "encodings. The fixed v1 rule uses 4095 when observed max <=4095 and 65535 "
                "otherwise; it never normalizes by each image's observed maximum."
            ),
        },
        "records": entries,
    }
    _write_json(manifest_path, manifest)
    return manifest


def infer(
    manifest_path: Path, cache_dir: Path, predictions_path: Path, device: str
) -> list[dict[str, Any]]:
    """Run the focused classifier on locally cached pilot images."""
    manifest = _read_json(manifest_path)
    adapter = MedRAXClassifierAdapter(
        device=device, input_maxval=manifest["image_encoding"]["input_maxval"]
    )
    predictions: list[dict[str, Any]] = []
    records = manifest["records"]
    for ordinal, record in enumerate(records, start=1):
        image_path = cache_dir / "images" / f"{record['row_idx']}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing cached image: {image_path}")
        actual_digest = _file_sha256(image_path)
        if actual_digest != record["image_sha256"]:
            raise RuntimeError(f"Image checksum mismatch for row {record['row_idx']}.")
        scores, maxval = adapter.predict(image_path)
        predictions.append(
            {
                "row_idx": record["row_idx"],
                "labels": record["labels"],
                "scores": scores,
                "input_maxval": maxval,
            }
        )
        if ordinal % 25 == 0 or ordinal == len(records):
            print(f"Inferred {ordinal}/{len(records)} images on {adapter.device}.")

    output: dict[str, Any] = {
        "schema_version": 1,
        "scope": "MedRAX ChestXRayClassifierTool configuration only",
        "medrax_commit": MEDRAX_COMMIT,
        "score_semantics": SCORE_SEMANTICS,
        "adapter": {
            "device": str(adapter.device),
            "input_maxval": manifest["image_encoding"]["input_maxval"],
            "model_checkpoint_sha256": adapter.model_file_sha256(),
        },
        "sampling": manifest["sampling"],
        "image_encoding": manifest["image_encoding"],
        "runtime": runtime_provenance(),
        "predictions": predictions,
    }
    _write_json(predictions_path, output)
    return predictions


def evaluate(predictions_path: Path, output_dir: Path, repetitions: int, seed: int) -> None:
    """Compute pre-specified aggregate results and safe public artifacts."""
    prediction_bundle = _read_json(predictions_path)
    endpoint_results, plotting_inputs = evaluate_predictions(
        prediction_bundle["predictions"], repetitions=repetitions, seed=seed
    )
    sensitivity_analyses = {
        "grade_at_least_1": evaluate_predictions(
            prediction_bundle["predictions"],
            repetitions=repetitions,
            seed=seed + 1_000,
            positive_threshold=1,
        )[0],
        "grade_at_least_3": evaluate_predictions(
            prediction_bundle["predictions"],
            repetitions=repetitions,
            seed=seed + 2_000,
            positive_threshold=3,
        )[0],
    }
    encoding_stratified = {
        str(maxval): evaluate_predictions(
            [
                prediction
                for prediction in prediction_bundle["predictions"]
                if prediction.get("input_maxval") == maxval
            ],
            repetitions=repetitions,
            seed=seed + int(maxval),
        )[0]
        for maxval in (4095.0, 65535.0)
    }
    run_metadata = {
        "sampling": prediction_bundle["sampling"],
        "image_encoding": prediction_bundle["image_encoding"],
        "adapter": prediction_bundle["adapter"],
        "runtime": prediction_bundle["runtime"],
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "n_predictions": len(prediction_bundle["predictions"]),
        "encoding_strata": {
            str(maxval): sum(
                prediction.get("input_maxval") == maxval
                for prediction in prediction_bundle["predictions"]
            )
            for maxval in (4095.0, 65535.0)
        },
    }
    write_aggregate_artifacts(
        endpoint_results,
        plotting_inputs,
        run_metadata,
        output_dir,
        sensitivity_analyses=sensitivity_analyses,
        encoding_stratified=encoding_stratified,
    )
    print(f"Wrote aggregate-only results to {output_dir}.")


def _add_cohort_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=512, help="One-image-per-patient pilot size.")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument(
        "--page-budget",
        type=int,
        default=64,
        help="Initial number of uniformly sampled 100-row dataset-server pages.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/taixray_pilot"))
    parser.add_argument("--manifest", type=Path, default=Path("results/pilot_manifest.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Create a deterministic local pilot cache.")
    _add_cohort_arguments(fetch_parser)

    infer_parser = subparsers.add_parser("infer", help="Run model inference from a local cache.")
    infer_parser.add_argument("--manifest", type=Path, default=Path("results/pilot_manifest.json"))
    infer_parser.add_argument("--cache-dir", type=Path, default=Path(".cache/taixray_pilot"))
    infer_parser.add_argument("--predictions", type=Path, default=Path("results/predictions.json"))
    infer_parser.add_argument("--device", default="auto")

    evaluate_parser = subparsers.add_parser("evaluate", help="Write aggregate-only metrics and figures.")
    evaluate_parser.add_argument("--predictions", type=Path, default=Path("results/predictions.json"))
    evaluate_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    evaluate_parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    evaluate_parser.add_argument("--seed", type=int, default=20261107)

    all_parser = subparsers.add_parser("all", help="Fetch, infer, and evaluate a deterministic pilot.")
    _add_cohort_arguments(all_parser)
    all_parser.add_argument("--predictions", type=Path, default=Path("results/predictions.json"))
    all_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    all_parser.add_argument("--device", default="auto")
    all_parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fetch":
        create_pilot(args.n, args.seed, args.cache_dir, args.manifest, args.page_budget)
    elif args.command == "infer":
        infer(args.manifest, args.cache_dir, args.predictions, args.device)
    elif args.command == "evaluate":
        evaluate(args.predictions, args.output_dir, args.bootstrap_repetitions, args.seed)
    elif args.command == "all":
        create_pilot(args.n, args.seed, args.cache_dir, args.manifest, args.page_budget)
        infer(args.manifest, args.cache_dir, args.predictions, args.device)
        evaluate(args.predictions, args.output_dir, args.bootstrap_repetitions, args.seed)
    else:  # Defensive: argparse guarantees the command is known.
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
