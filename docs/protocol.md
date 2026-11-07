# Protocol: MedRAX classifier × TAIX-Ray external-validation pilot

## Question and scope

This project asks a narrow, reproducible question: how does the
`ChestXRayClassifierTool` used by [MedRAX](https://github.com/bowang-lab/MedRAX)
behave on a newly released, external chest-radiograph dataset after its input
precision is handled explicitly?

It is **not** an evaluation of the complete MedRAX agent, GPT-4o orchestration,
tool selection, report generation, clinical reasoning, or clinical utility.
MedRAX's full stack has additional model, API, and hardware dependencies that
are outside this focused audit.

The pilot uses the same TorchXRayVision DenseNet weight identifier and
preprocessing path called by MedRAX's classifier tool:

```text
torchxrayvision.models.DenseNet(weights="densenet121-res224-all")
XRayCenterCrop → DenseNet score vector
```

## Why a preprocessing patch is necessary

At the audited MedRAX revision, the classifier calls
`xrv.datasets.normalize(image, 255)` unconditionally. The default TAIX-Ray
images are unsigned 16-bit PNGs; a public test image has 12-bit-valued pixels
in the range 0–4095. TorchXRayVision rejects that input with a `ValueError`
before inference when it is given the fixed 8-bit bound.

The accompanying patch preserves the original `uint8` behavior and requires an
explicit `input_maxval` for non-8-bit integer data. This is intentional: a
`uint16` container alone does not reveal whether the stored signal is 12-bit,
full-range 16-bit, or another encoding. During the smoke audit, the public
TAIX-Ray export exhibited both 0--4095 and wider-range uint16 encodings. The
benchmark adapter therefore uses a frozen dataset-specific rule:
`max <= 4095 → 4095`; otherwise `65535`. This selects one of two declared
encoding scales and never divides by an image's own observed maximum. It is a
reproducibility convention for this export, not a claim that original DICOM
bit-depth metadata was recovered.

The patch also makes the classifier's `device="auto"` choose CUDA, then Apple
MPS, then CPU. That compatibility claim applies to this DenseNet classifier
only—not to every optional MedRAX tool.

## Frozen provenance

| Item | Locked reference |
| --- | --- |
| MedRAX source | `bowang-lab/MedRAX` commit `d243cc50348fa0f0c219cc623e2cf26b906be676` |
| Patch implementation | [asharma391/MedRAX](https://github.com/asharma391/MedRAX), `fix/portable-classifier-inputs` / corresponding `main` commit |
| TAIX-Ray dataset | [`TLAIM/TAIX-Ray`](https://huggingface.co/datasets/TLAIM/TAIX-Ray), `default` configuration, `test` split |
| Dataset revision | `7f43720e2c7ab6400608880c0cb8b950ac0e5afb` |
| TAIX-Ray source code | [`TruhnLab/TAIX-Ray`](https://github.com/TruhnLab/TAIX-Ray) commit `65b6cc310cbcb4ead339b567faf955f51218a9f3` |
| Dataset paper | Truhn et al., *Scientific Data* (2026), [doi:10.1038/s41597-026-07271-7](https://doi.org/10.1038/s41597-026-07271-7) |
| Dataset licence | CC BY 4.0 |
| Pilot model runtime | Recorded with the result: Python, PyTorch, TorchXRayVision, operating system, device, and model-weight SHA-256 |

TAIX-Ray's predefined test split contains 42,928 radiographs from 9,545
patients. The resource is publicly released, but its data-use notice still
requires responsible research use, prohibits re-identification efforts, and
states that it is not for clinical decision-making.

## Cohort construction

The protocol is compatible with the official TAIX-Ray `test` split. This
resource-constrained pilot uses a deterministic two-stage design:

1. Uniformly sample a fixed number of 100-row dataset-server pages from across
   the official split using a locked seed.
2. Within that candidate pool, retain no more than one image per hashed
   TAIX-Ray patient identifier (the identifier is used only in memory), then
   randomly order the remaining patient-level records with the same seed.
3. Fetch each selected image from the official source at runtime and record only the
   non-identifying row index, image byte SHA-256, labels, configuration, and
   scores needed to reproduce aggregate results.
4. Publish neither image pixels, signed image URLs, patient identifiers, nor
   report text.

This makes a limited local run an explicitly labelled **two-stage pilot**, not
a substitute for the full 42,928-image test-set evaluation. The exact seed,
page budget, candidate-row count, unique-patient count, selected sample size,
and final label counts are written before reporting any metric.

## Outcome definitions

TAIX-Ray's structured radiologist reports use ordinal labels. `HeartSize` is
`0=Normal`, `1=Borderline`, `2=Enlarged`, and `3=Massively`. The other reported
findings are `0=None`, `1=(+)` (questionable), `2=+` (mild), `3=++`
(moderate), and `4=+++` (severe).

The primary analysis calls a finding *definite* at grade `>=2`, calls grade
`0` negative, and excludes the uncertain grade `1`. For bilateral findings,
the maximum left/right grade is used only after excluding any ambiguous-only
case. The pre-specified mappings are:

| MedRAX/XRV output | TAIX-Ray reference construction | Status |
| --- | --- | --- |
| `Atelectasis` | `max(Atelectasis_Left, Atelectasis_Right)` | Direct concept match |
| `Effusion` | `max(PleuralEffusion_Left, PleuralEffusion_Right)` | Direct concept match |
| `Lung Opacity` | `max(PulmonaryOpacities_Left, PulmonaryOpacities_Right)` | Direct concept match |
| `Cardiomegaly` | `HeartSize >= 2` | Related phenotype; analysed separately |
| `Edema` | `PulmonaryCongestion` | Exploratory semantic proxy only; never pooled with primary endpoints |

Sensitivity analyses include `>=1` (including questionable) and `>=3`
(moderate/severe) thresholds. The protocol never chooses a threshold after
inspecting the test results.

## Metrics and uncertainty

The classifier outputs are **operating-point-normalized 0–1 scores**, not
calibrated probabilities. Therefore the primary metrics are ranking metrics:

- AUROC for each primary endpoint;
- average precision, reported with the positive prevalence;
- 95% percentile confidence intervals from 2,000 seeded bootstrap resamples.

For a one-image-per-patient pilot, image-level bootstrap is equivalent to a
patient-level bootstrap. For an all-image run, resampling is clustered by
patient. Fixed-threshold sensitivity and specificity at score `0.5` may be
shown as an exploratory source-operating-point diagnostic, never tuned on the
test set. Brier score and expected calibration error are intentionally omitted
because the scores are not calibrated probabilities.

## Interpretation limits

- TAIX-Ray labels are structured routine radiologist report labels, not an
  independent multi-reader image re-read reference standard.
- TAIX-Ray was publicly released in 2026 and comes from an external German ICU
  site, but absence from upstream model pretraining cannot be proven. Results
  should say “external site / no known listed overlap,” never “guaranteed
  unseen.”
- The study assesses four label-aligned classifier outputs. It cannot establish
  agent-level reasoning performance, safety, diagnosis quality, calibration,
  or clinical effectiveness.
- The data and model are research artifacts only. No output is a clinical
  recommendation or a substitute for radiologist review.

## Expected artifacts

The repository should publish source code, a dependency lock, tests, a
machine-readable aggregate result file, figures derived only from aggregates,
and this protocol. It must not redistribute TAIX-Ray images, identifiers,
reports, or signed download links.
