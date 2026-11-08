# ChestDR: pinned MedFMC testing partition results

Generated: 2026-11-08T02:40:15.239656+00:00

This report evaluates the `ChestXRayClassifierTool` configuration used in
MedRAX, not the GPT-driven MedRAX agent or a clinical workflow. It contains
aggregate metrics only; no source images, identifiers, image URLs, row-level
labels, or row-level scores are committed.

## Direct endpoint results

| Endpoint | Evaluable n | Positive n | Prevalence | AUROC (95% bootstrap CI) | AP (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Effusion | 3869 | 1011 | 0.261 | 0.892 [0.881, 0.903] | 0.777 [0.753, 0.801] |
| Cardiomegaly | 3869 | 1055 | 0.273 | 0.876 [0.865, 0.887] | 0.732 [0.705, 0.759] |
| Pneumonia | 3869 | 705 | 0.182 | 0.599 [0.577, 0.621] | 0.248 [0.225, 0.276] |
| Atelectasis | 3869 | 156 | 0.040 | 0.623 [0.582, 0.660] | 0.056 [0.046, 0.072] |
| Pneumothorax | 3869 | 235 | 0.061 | 0.780 [0.753, 0.807] | 0.149 [0.127, 0.181] |
| Nodule | 3869 | 536 | 0.139 | 0.566 [0.539, 0.593] | 0.195 [0.170, 0.226] |
| Edema | 3869 | 67 | 0.017 | 0.913 [0.870, 0.950] | 0.248 [0.175, 0.367] |
| Emphysema | 3869 | 125 | 0.032 | 0.723 [0.671, 0.771] | 0.144 [0.097, 0.211] |
| Fibrosis | 3869 | 435 | 0.112 | 0.670 [0.644, 0.696] | 0.199 [0.174, 0.233] |
| Pleural thickening | 3869 | 385 | 0.100 | 0.671 [0.643, 0.699] | 0.184 [0.158, 0.220] |

Intervals are 2,000-replicate percentile bootstrap intervals
over image-level observations, generated from base seed `20261107`
(endpoint-specific offsets are deterministic). No decision threshold was
selected or tuned on this test set. Scores are torchxrayvision operating-point-normalized model score in [0, 1]; not a calibrated probability.

## Interpretation boundaries

- ChestDR labels were initially derived from signed clinical reports by a radiology resident and confirmed by a senior radiologist; they are not a new blinded image re-read for this project.
- The partition is a pinned published MedFMC benchmark testing list, not claimed to be an official test split created by the original ChestDR dataset authors.
- Only direct finding-name/terminology matches to the MedRAX-selected TorchXRayVision output vocabulary are reported; no surrogate mapping is used for the remaining ChestDR labels.
- The result evaluates the classifier component only, not the GPT-driven MedRAX agent or a clinical workflow.
- External evaluation does not establish absence of every possible pretraining overlap, although ChestDR is absent from the classifier's declared source datasets.

The machine-readable aggregate result is in
[aggregate_metrics.json](aggregate_metrics.json).
