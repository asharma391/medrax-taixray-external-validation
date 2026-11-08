import json

import numpy as np

from medrax_taixray_audit.external import (
    BinaryEndpoint,
    evaluate_binary_predictions,
    labels_from_mask_values,
    write_external_artifacts,
)


def test_labels_from_mask_values_uses_declared_class_ids_only():
    mask = np.array([[255, 1, 1], [6, 255, 255]], dtype=np.uint8)
    labels = labels_from_mask_values(
        mask,
        {"Atelectasis": 1, "Effusion": 6, "Pneumothorax": 13},
    )
    assert labels == {"Atelectasis": 1, "Effusion": 1, "Pneumothorax": 0}


def test_binary_endpoint_metrics_use_direct_source_mapping():
    endpoint = BinaryEndpoint(
        name="Pneumonia",
        score_key="Pneumonia",
        source_label="Pneumonia",
        relation_note="Direct match.",
    )
    predictions = [
        {"labels": {"Pneumonia": 0}, "scores": {"Pneumonia": 0.1}},
        {"labels": {"Pneumonia": 1}, "scores": {"Pneumonia": 0.9}},
        {"labels": {"Pneumonia": 0}, "scores": {"Pneumonia": 0.2}},
        {"labels": {"Pneumonia": 1}, "scores": {"Pneumonia": 0.8}},
    ]
    results, roc_inputs = evaluate_binary_predictions(
        predictions, [endpoint], repetitions=50, seed=7
    )
    result = results["Pneumonia"]
    assert result["n_evaluable"] == 4
    assert result["n_positive"] == 2
    assert result["auroc"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["auroc_95_ci"] is not None
    assert roc_inputs["Pneumonia"][0].tolist() == [0, 1, 0, 1]


def test_aggregate_artifact_records_deterministic_bootstrap_scheme(tmp_path):
    endpoint = BinaryEndpoint(
        name="Pneumonia",
        score_key="Pneumonia",
        source_label="Pneumonia",
        relation_note="Direct match.",
    )
    results, roc_inputs = evaluate_binary_predictions(
        [
            {"labels": {"Pneumonia": 0}, "scores": {"Pneumonia": 0.1}},
            {"labels": {"Pneumonia": 1}, "scores": {"Pneumonia": 0.9}},
        ],
        [endpoint],
        repetitions=5,
        seed=17,
    )
    write_external_artifacts(
        dataset_name="Test cohort",
        title="Test figure",
        endpoint_results=results,
        roc_inputs=roc_inputs,
        provenance={},
        cohort={},
        score_semantics="score",
        output_dir=tmp_path,
        bootstrap_repetitions=5,
        bootstrap_seed=17,
        limitations=(),
    )
    payload = json.loads((tmp_path / "aggregate_metrics.json").read_text())
    assert payload["bootstrap"] == {
        "repetitions": 5,
        "base_seed": 17,
        "endpoint_seed_scheme": (
            "AUROC=base_seed+endpoint_index; "
            "average_precision=base_seed+100+endpoint_index"
        ),
    }
