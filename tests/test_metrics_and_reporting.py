import json

import numpy as np

from medrax_taixray_audit.dataset import LABEL_COLUMNS
from medrax_taixray_audit.metrics import evaluate_predictions
from medrax_taixray_audit.reporting import write_aggregate_artifacts


def _prediction(grade: int, score: float) -> dict:
    return {
        "labels": {label: grade for label in LABEL_COLUMNS},
        "scores": {
            "Atelectasis": score,
            "Effusion": score,
            "Lung Opacity": score,
            "Cardiomegaly": score,
        },
    }


def test_metrics_use_only_non_questionable_observations_and_write_safe_artifacts(tmp_path):
    predictions = [
        _prediction(0, 0.1),
        _prediction(2, 0.9),
        _prediction(1, 0.5),
        _prediction(0, 0.2),
        _prediction(3, 0.8),
    ]
    results, plotting_inputs = evaluate_predictions(predictions, repetitions=30, seed=5)
    atelectasis = results["Atelectasis"]
    assert atelectasis["n_evaluable"] == 4
    assert atelectasis["n_excluded_questionable"] == 1
    assert atelectasis["auroc"] == 1.0
    inclusive, _ = evaluate_predictions(
        predictions, repetitions=30, seed=5, positive_threshold=1
    )
    assert inclusive["Atelectasis"]["n_excluded_questionable"] == 0

    write_aggregate_artifacts(
        results,
        plotting_inputs,
        {"n_predictions": 5},
        tmp_path,
        sensitivity_analyses={"grade_at_least_1": inclusive},
    )
    payload = json.loads((tmp_path / "aggregate_metrics.json").read_text())
    assert payload["run_metadata"]["n_predictions"] == 5
    assert (tmp_path / "RESULTS.md").exists()
    assert (tmp_path / "figures" / "primary_roc_curves.png").exists()
