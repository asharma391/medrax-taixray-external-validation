# TAIX-Ray external-validation pilot results

Generated: 2026-11-07T10:12:57.120763+00:00

This report evaluates only the `ChestXRayClassifierTool` configuration used in
MedRAX, not the GPT-driven MedRAX agent or a clinical workflow. The test cohort
is a deterministic, one-image-per-patient sample from TAIX-Ray's public test
split; no images, patient identifiers, image URLs, or row-level labels are
committed to this repository.

## Pre-specified primary endpoints

| Endpoint | Evaluable n | Positive n | Prevalence | AUROC (95% bootstrap CI) | AP (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 472 | 365 | 0.773 | 0.706 [0.649, 0.762] | 0.873 [0.831, 0.914] |
| Effusion | 440 | 253 | 0.575 | 0.778 [0.732, 0.820] | 0.808 [0.754, 0.861] |
| Lung Opacity | 451 | 175 | 0.388 | 0.744 [0.694, 0.791] | 0.690 [0.624, 0.758] |
| Cardiomegaly | 434 | 251 | 0.578 | 0.740 [0.697, 0.785] | 0.797 [0.751, 0.843] |

Primary labels treat TAIX's grade 0 as absent and grades 2--4 as definite;
grade 1 (questionable) is excluded. Intervals are 2,000-replicate percentile
bootstrap intervals over the one-image-per-patient pilot observations.

## Pre-specified label-threshold sensitivity

| Endpoint | Definite (>=2) AUROC | Include questionable (>=1) AUROC | Moderate/severe (>=3) AUROC |
| --- | ---: | ---: | ---: |
| Atelectasis | 0.706 | 0.698 | 0.740 |
| Effusion | 0.778 | 0.759 | 0.832 |
| Lung Opacity | 0.744 | 0.690 | 0.884 |
| Cardiomegaly | 0.740 | 0.716 | 0.672 |

The score threshold is never tuned on this test pilot. These label-threshold
analyses test whether the rank-ordering result is sensitive to how the ordinal
radiologist grades are dichotomized.

## Image-encoding diagnostic

The public TAIX-Ray PNG export contained two observed uint16 range encodings;
the predeclared adapter used 4095 for the 0--4095 range and 65535 for the
wider range. This post-run stratification is descriptive only: the two groups
may also differ in acquisition, patients, or other unobserved factors, so it
does **not** establish that intensity encoding caused a performance difference.

| Endpoint | 0--4095 stratum AUROC (95% CI) | wider uint16 stratum AUROC (95% CI) |
| --- | ---: | ---: |
| Atelectasis | 0.758 [0.679, 0.830] | 0.648 [0.561, 0.730] |
| Effusion | 0.811 [0.747, 0.870] | 0.741 [0.678, 0.806] |
| Lung Opacity | 0.746 [0.670, 0.811] | 0.747 [0.684, 0.804] |
| Cardiomegaly | 0.804 [0.746, 0.860] | 0.669 [0.597, 0.737] |

## Interpretation boundaries

- Scores are torchxrayvision operating-point-normalized model score in [0, 1]; not a calibrated probability.
- TAIX labels are structured routine radiologist-report grades, not a consensus
  image re-read reference standard.
- This is an external-site pilot, not a full 42,928-image benchmark and not a
  clinical validation. It does not establish absence of all possible pretraining
  overlap.
- The published 16-bit compatibility patch is linked at https://github.com/asharma391/MedRAX; it makes
  the required acquisition scale explicit rather than performing unsafe
  per-image contrast scaling.

See [the protocol](../docs/protocol.md) for frozen revisions, label mapping,
data policy, and the full set of limitations. The aggregate machine-readable
results are in [aggregate_metrics.json](aggregate_metrics.json).
