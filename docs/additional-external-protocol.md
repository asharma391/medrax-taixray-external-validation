# Protocol: additional external classifier evaluations

## Question and scope

This protocol freezes two dataset-scoped evaluations of the
`ChestXRayClassifierTool` configuration used by
[MedRAX](https://github.com/bowang-lab/MedRAX). They extend, but are not pooled
with, the TAIX-Ray pilot. Each asks how the same released TorchXRayVision
DenseNet ranks a direct, released reference label on a distinct public cohort.

They are **not** evaluations of MedRAX's GPT-driven agent, tool selection,
medical reasoning, report generation, clinical workflow, calibration, or
clinical utility. They are also not an attempt to diagnose individual people.

The evaluator uses the weight identifier and preprocessing path selected by
MedRAX:

```text
torchxrayvision.models.DenseNet(weights="densenet121-res224-all")
XRayCenterCrop -> DenseNet score vector
```

The resulting 0--1 values are operating-point-normalized model scores, not
calibrated probabilities. AUROC and average precision are therefore the
pre-specified metrics; no threshold is selected or tuned on either test set.

## Frozen datasets and cohort construction

### ChestDR: pinned MedFMC testing partition

The source images and labels are the public
[ChestDR release](https://doi.org/10.6084/m9.figshare.22302775.v1), whose
dataset paper describes 4,848 frontal radiographs from 4,848 individual
patients at two Chinese hospitals. The complete 3,869-row testing list is
read from the published [MedFMC few-shot baseline
repository](https://github.com/wllfore/MedFMC_fewshot_baseline), pinned at
commit `3bfe04b153ab461baccbe574c9dc3dff84ab8b11`:

```text
data/ChestDR/test.txt
```

That repository explains that `test.txt` contains the images remaining after
its few-shot support-pool selection. It is consequently described here as a
**pinned published MedFMC testing partition**, not as an official split created
by the original ChestDR dataset authors. Its provenance is cited as Wang et
al., *Scientific Data* (2023),
[doi:10.1038/s41597-023-02460-0](https://doi.org/10.1038/s41597-023-02460-0).

Before image inference, the runner verifies all 3,869 manifest labels against
the official CSV inside the immutable ChestDR archive, requires unique image
identifiers, and requires every listed image member to exist. It then evaluates
the whole pinned partition; a command-line cap is permitted only for a local
smoke test and is never written as a benchmark result. Since the source release
states one image per patient, an image-level bootstrap is patient-equivalent
for this uniquely identified cohort.

Only exact or direct terminology matches are reported:

| ChestDR reference label | XRV / MedRAX score | Mapping |
| --- | --- | --- |
| `pleural_effusion` | `Effusion` | Direct finding match |
| `cardiomegaly` | `Cardiomegaly` | Direct finding match |
| `pneumonia` | `Pneumonia` | Direct finding match |
| `atelectasis` | `Atelectasis` | Direct finding match |
| `pneumothorax` | `Pneumothorax` | Direct finding match |
| `nodule` | `Nodule` | Direct finding match |
| `pulmonary_edema` | `Edema` | Direct finding match |
| `emphysema` | `Emphysema` | Direct finding match |
| `fibrosis` | `Fibrosis` | Direct finding match |
| `thickened_pleura` | `Pleural_Thickening` | Direct terminology match |

No surrogate mapping is made for the other ChestDR labels; in particular,
`consolidation` is not relabelled as the broader XRV `Lung Opacity` output.

### Pediatric pneumonia: pinned held-out image test split

The second evaluation uses the `test` split of
[`hf-vision/chest-xray-pneumonia`](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia),
pinned at revision `c1a67c18df1a52ec332ffc00418f4eaa61a9f2bc`. It contains
624 released test images and is evaluated in full. Its sole direct mapping is
the source binary label `PNEUMONIA` to the model's `Pneumonia` score.

This is intentionally framed as a narrow **pediatric pneumonia domain-transfer
test**, not an adult multi-finding benchmark. The public acquisition view does
not expose patient grouping, so its bootstrap intervals are image-level and no
patient-level claim is made. The underlying released dataset source is cited
at [doi:10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2).

## Reproducibility and data minimization

The run records the exact MedRAX commit, DenseNet weight-file SHA-256,
TorchXRayVision/PyTorch/Torchvision versions, acquisition revision or pinned
manifest commit, endpoint mappings, and random seeds in an aggregate JSON
artifact. All 95% intervals use 2,000 seeded percentile bootstrap replicates.
The default base seed is `20261107`, with a fixed offset of `0` for ChestDR
and `10000` for pediatric pneumonia, so a cohort's intervals do not change if
the command-line dataset order changes; the exact base seed and endpoint seed
scheme are retained in each aggregate result.

ChestDR is accessed via HTTP byte ranges, and both datasets are processed one
image at a time in a temporary directory. Restart checkpoints containing
in-memory labels and scores are stored outside the repository and are deleted
after a successful complete run. The public repository contains only source
code, aggregate metrics, and figures; it never contains source images, image
identifiers, source URLs, row-level labels, row-level scores, reports, or
patient information.

## Interpretation boundaries

- The reference labels are released dataset annotations, not new blinded
  multi-reader image interpretations made for this project.
- These cohorts are absent from the classifier's declared training-source list
  for `densenet121-res224-all` in the
  [TorchXRayVision model card](https://huggingface.co/torchxrayvision/densenet121-res224-all).
  That supports an external-cohort description; it does not prove absence of
  every possible upstream or pretraining overlap.
- A result for the classifier tool cannot be generalized to MedRAX as an
  agent, a diagnosis, a recommendation, or clinical performance.
- No test result should be used for patient care.
