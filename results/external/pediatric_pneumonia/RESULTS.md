# Pediatric pneumonia held-out test set results

Generated: 2026-11-07T23:23:11.600662+00:00

This report evaluates the `ChestXRayClassifierTool` configuration used in
MedRAX, not the GPT-driven MedRAX agent or a clinical workflow. It contains
aggregate metrics only; no source images, identifiers, image URLs, row-level
labels, or row-level scores are committed.

## Direct endpoint results

| Endpoint | Evaluable n | Positive n | Prevalence | AUROC (95% bootstrap CI) | AP (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pneumonia | 624 | 390 | 0.625 | 0.717 [0.677, 0.756] | 0.804 [0.761, 0.844] |

Intervals are 2,000-replicate percentile bootstrap intervals
over image-level observations, generated from base seed `20271107`
(endpoint-specific offsets are deterministic). No decision threshold was
selected or tuned on this test set. Scores are torchxrayvision operating-point-normalized model score in [0, 1]; not a calibrated probability.

## Interpretation boundaries

- This is a narrow binary pediatric domain-transfer test (normal versus pneumonia), not an adult multi-finding clinical validation.
- The public acquisition view does not expose patient grouping, so bootstrap intervals are image-level rather than patient-clustered.
- The pinned source convention is 0=NORMAL and 1=PNEUMONIA; its 234 normal and 390 pneumonia test labels are used as released.
- The source label is used as released; no model threshold was selected or tuned on this test split.
- The result evaluates the classifier component only, not the GPT-driven MedRAX agent or a clinical workflow.

The machine-readable aggregate result is in
[aggregate_metrics.json](aggregate_metrics.json).
