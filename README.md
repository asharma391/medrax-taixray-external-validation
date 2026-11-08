# MedRAX CXR classifier — external-validation audit

[![Tests](https://github.com/asharma391/medrax-taixray-external-validation/actions/workflows/test.yml/badge.svg)](https://github.com/asharma391/medrax-taixray-external-validation/actions/workflows/test.yml)

A transparent, data-free audit of the released
[`ChestXRayClassifierTool`](https://github.com/bowang-lab/MedRAX) configuration
used by MedRAX. It includes a pre-specified 512-patient [TAIX-Ray](https://doi.org/10.1038/s41597-026-07271-7)
pilot plus two separately reported external cohorts: the 3,869-image pinned
MedFMC ChestDR testing partition and a 624-image pediatric pneumonia test set.
The project also documents and fixes a real input-compatibility blocker:
unmodified MedRAX assumes all PNGs are 8-bit and rejects TAIX-Ray's valid
`uint16` images before inference.

> This is a tool-level external-validation pilot—not an evaluation of the full
> GPT-orchestrated MedRAX agent, a diagnostic system, or a clinical validation.

![TAIX-Ray pilot ROC curves](results/figures/primary_roc_curves.png)

## What was completed

1. Audited MedRAX at commit [`d243cc5`](https://github.com/bowang-lab/MedRAX/tree/d243cc50348fa0f0c219cc623e2cf26b906be676) and reproduced the 16-bit input failure using a public TAIX-Ray image.
2. Published a tested, conservative [compatibility patch](https://github.com/asharma391/MedRAX): explicit high-bit-depth scaling, classifier-only `device="auto"` support (CUDA → MPS → CPU), and corrected score semantics.
3. Locked a **512-image, one-image-per-patient**, two-stage random-page pilot before viewing model outcomes. The candidate frame contained 6,400 official test rows from 64 uniformly sampled pages and 1,434 unique in-memory patient pseudonyms.
4. Ran the MedRAX-selected `torchxrayvision` `densenet121-res224-all` classifier on Apple MPS, checked image byte hashes, and reported only aggregate metrics with 2,000 bootstrap replicates.
5. Evaluated the same released classifier on two distinct additional cohorts, with dataset-specific provenance, direct endpoint mappings, and aggregate-only outputs: the complete 3,869-image pinned MedFMC ChestDR testing partition and the complete 624-image pediatric pneumonia test split.

## Primary result

TAIX grades 2–4 are “definite” findings; grade 0 is negative; grade 1
(questionable) is excluded. AUROC is the primary metric because the underlying
TorchXRayVision outputs are operating-point-normalized scores, not calibrated
probabilities.

| Endpoint | Evaluable n | Positive n | AUROC (95% bootstrap CI) | Average precision (95% CI) |
| --- | ---: | ---: | ---: | ---: |
| Atelectasis | 472 | 365 | 0.706 (0.649–0.762) | 0.873 (0.831–0.914) |
| Effusion | 440 | 253 | 0.778 (0.732–0.820) | 0.808 (0.754–0.861) |
| Lung Opacity | 451 | 175 | 0.744 (0.694–0.791) | 0.690 (0.624–0.758) |
| Cardiomegaly / enlarged heart size | 434 | 251 | 0.740 (0.697–0.785) | 0.797 (0.751–0.843) |

The complete [aggregate results](results/RESULTS.md) include the
pre-specified ordinal-label sensitivities and a descriptive two-encoding
diagnostic. The latter is not a causal claim: acquisition and cohort factors
may differ between the 0–4095 and wider-range `uint16` image strata.

## Additional external evaluations

These results are reported separately from the TAIX-Ray pilot and are never
pooled. They evaluate only the released classifier component—not MedRAX's
GPT-driven agent, tool selection, clinical workflow, or clinical utility.

| Cohort | Locked cohort | Scope | Aggregate result |
| --- | --- | --- | --- |
| [ChestDR](results/external/chestdr/RESULTS.md) | Entire 3,869-image pinned MedFMC testing partition | 10 direct finding-name/terminology mappings | AUROC spans 0.566–0.913 across all 10 reported endpoints; effusion 0.892 (95% CI 0.881–0.903), cardiomegaly 0.876 (0.865–0.887), and edema 0.913 (0.870–0.950). |
| [Pediatric pneumonia](results/external/pediatric_pneumonia/RESULTS.md) | Entire 624-image checksum-pinned test split | Narrow pediatric pneumonia domain-transfer test | Pneumonia AUROC 0.717 (95% CI 0.677–0.756); AP 0.804 (0.761–0.844). |

The ChestDR list is a pinned, published MedFMC testing partition—not claimed
to be an official split created by the original ChestDR authors. Its ten
directly matched results, cohort provenance, and limitations are in the linked
aggregate report; see the [additional-evaluation protocol](docs/additional-external-protocol.md)
for the frozen sources, mappings, data-minimization policy, and interpretation
boundaries.

## Why this is useful

The original classifier line is effectively:

```python
xrv.datasets.normalize(img, 255)
```

TAIX-Ray's public images include `uint16` PNGs, so the call raises before
model inference. The [patch](https://github.com/asharma391/MedRAX) preserves
the upstream `uint8` path and refuses ambiguous wider-integer input unless an
explicit intensity scale is supplied. It deliberately does **not** normalize
each image by its own maximum, because that would silently change image
contrast and conceal an acquisition-format assumption.

For TAIX-Ray, the audit uses a fully documented range rule: images with
observed maxima ≤4095 use a 4095 scale; wider-range `uint16` images use 65535.
This is a reproducibility convention for the public PNG export, not a claim
that lost DICOM bit-depth metadata has been recovered.

## Reproduce

Requirements: Python 3.11–3.13, internet access to the public TAIX-Ray dataset
viewer, and sufficient local storage for the private temporary image cache. A
GPU is not required for the classifier-only run; `--device auto` uses CUDA,
then Apple MPS, then CPU.

```bash
git clone https://github.com/asharma391/medrax-taixray-external-validation.git
cd medrax-taixray-external-validation
uv sync --extra dev
uv run medrax-taixray-audit all \
  --n 512 --page-budget 64 --seed 20261107 \
  --cache-dir .cache/taixray_pilot \
  --manifest results/pilot_manifest.json \
  --predictions results/predictions.json \
  --output-dir results --bootstrap-repetitions 2000 --device auto
uv run pytest
```

The CLI has separate `fetch`, `infer`, and `evaluate` stages. This permits a
reviewer to inspect the locked cohort manifest before inference. Public dataset
rate limits are respected with page caching and backoff.

To reproduce the two additional aggregate-only evaluations:

```bash
PYTHONPATH=src uv run python scripts/run_additional_external_benchmarks.py \
  --datasets chestdr pediatric_pneumonia \
  --bootstrap-repetitions 2000 --device auto
```

The runner accesses ChestDR by temporary HTTP byte ranges and downloads the
pinned pediatric test Parquet only into a temporary directory; both source
images and restart checkpoints remain outside the repository.

## Data and ethics policy

This repository intentionally contains **no source images, report text,
patient/physician identifiers, signed image URLs, or row-level labels/scores**
from any cohort. Those local artifacts are ignored by Git. The public result
includes only aggregate metrics, derived ROC figures, software versions, model
checkpoint hashes, pinned dataset provenance, sample-design counts, and
encoding-stratum counts.

TAIX-Ray is CC-BY-4.0, but its data-use notice prohibits re-identification and
states that it is for research rather than clinical decision-making. Nothing in
this repository is medical advice or suitable for clinical use.

## Documentation

- [Protocol](docs/protocol.md): frozen provenance, outcome definitions,
  sampling design, uncertainty, ethics, and limitations.
- [Additional-evaluation protocol](docs/additional-external-protocol.md):
  frozen ChestDR and pediatric sources, direct mappings, data-minimization
  policy, and interpretation boundaries.
- [Project-selection audit](docs/project-selection.md): why MedRAX was a
  verifiable primary contribution while BioReason and dnaHNet were not used for
  unsupported performance claims.
- [Aggregate results](results/RESULTS.md): primary, sensitivity, and
  encoding-stratified findings.
- [Compatibility patch](https://github.com/asharma391/MedRAX): direct fork of
  MedRAX with unit-tested input/device/score-semantics changes.

## Important limitations

- The pilot is not the full 42,928-radiograph TAIX-Ray test split.
- TAIX labels are structured routine radiologist-report grades, not consensus
  image re-reads.
- ChestDR labels are released report-derived annotations; its pinned MedFMC
  partition is not claimed as the original authors' official split.
- The pediatric result is a binary image-level domain-transfer test; public
  acquisition does not expose patient grouping.
- “External” means a 2026 public release from an external German ICU site; it
  does not prove absence of every possible pretraining overlap; likewise,
  additional cohorts are absent from the classifier's declared source list but
  this does not prove absence of every possible upstream overlap.
- The tested component is one classifier tool, not MedRAX's LLM agent, tool
  selection, or medical reasoning system.

## Citation

If this audit is useful, please cite the original MedRAX and TAIX-Ray work
alongside this repository. See [CITATION.cff](CITATION.cff).
