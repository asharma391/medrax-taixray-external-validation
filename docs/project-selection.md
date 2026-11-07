# Project selection and feasibility audit

This note records why the completed contribution centers on MedRAX rather than
presenting an unverifiable result for every project initially considered. The
selection is a research-scope decision, not a ranking of the projects.

Assessment date: 2026-11-07.

| Project | Public assets checked | What can be completed honestly on the available machine | Decision |
| --- | --- | --- | --- |
| [MedRAX](https://github.com/bowang-lab/MedRAX) | Code (Apache-2.0), an embedded TorchXRayVision CXR classifier, and runnable model weights | A tool-level input-compatibility patch and external classifier validation. The full agent needs GPT/API access and several large optional tools, so it is not claimed. | Selected |
| [BioReason](https://github.com/bowang-lab/BioReason) | Code and benchmark datasets (Apache-2.0); no original BioReason checkpoint or GitHub release located | A reproducibility environment or temporal ClinVar harness could be prepared, but a published-model external result cannot be run without the missing checkpoint and CUDA/vLLM setup. | Documented, not used for performance claims |
| dnaHNet | ICML 2026 paper; no official code or checkpoint repository located | A future evaluation protocol can be proposed, but no actual model benchmark or bug fix can be claimed. Original experiments use large GPU resources beyond this local host. | Documented, not used for performance claims |

## MedRAX: selected because the scope is verifiable

MedRAX includes an independently usable CXR classifier that delegates to
TorchXRayVision's `densenet121-res224-all` model. An audit of the released
source exposed two concrete issues relevant to external reproducibility:

1. Its input path unconditionally normalizes as if all PNGs were 8-bit. This
   rejects valid 16-bit TAIX-Ray PNGs before inference.
2. Its prose calls the model outputs probabilities even though the underlying
   model applies operating-point normalization. They are bounded scores, not
   calibrated probabilities.

The patch is intentionally small: an explicit high-bit-depth intensity-scale
parameter, classifier-specific portable device selection, tests for 8/12/16-bit
handling, and corrected score semantics. TAIX-Ray provides a compatible, recent
external evaluation setting: its 2026 public release is CC BY 4.0, has a
predefined test split, and includes structured radiologist report grades for
several concepts aligned with the classifier outputs.

This choice supports a clear claim: an *external classifier audit*, not a
clinical validation or a full-agent benchmark.

## BioReason: meaningful future work, no fabricated result

The BioReason README still states that checkpoints will be released soon.
Its evaluation script expects a user-supplied checkpoint directory, and the
public Hugging Face collection contains datasets rather than a checkpoint for
the original BioReason model. The open
[dependency-reproducibility issue](https://github.com/bowang-lab/BioReason/issues/20)
also highlights unresolved version/API questions around the TRL training stack.

A legitimate follow-on project would freeze a CUDA/Python 3.11 training
environment with smoke tests and build a temporally held-out ClinVar protocol.
That would be useful infrastructure, but it cannot honestly supply a
BioReason-model result until weights and an appropriate GPU environment are
available. It is therefore not presented as a completed experiment.

## dnaHNet: no implementation to evaluate yet

The [dnaHNet ICML 2026 paper](https://proceedings.mlr.press/v306/shah26b.html)
describes a promising tokenizer-free genomic foundation model, but no official
source implementation or checkpoint was found in the Wang Lab organization or
the paper's released materials at the assessment date. The reported training
and efficiency experiments are GPU-scale; an Apple Silicon laptop cannot
reproduce them faithfully.

An appropriate future external task is its gene-essentiality setting using the
2026 *Pseudomonas aeruginosa* PA14 reference resource (CC BY), with careful
training-overlap checks. Until official weights are available, this remains a
protocol—not a result.

## Why TAIX-Ray rather than a larger VQA benchmark

Several new chest-X-ray VQA resources are derived from MIMIC-CXR or require
credentialed/controlled access. The present repository is meant to be openly
inspectable and reproducible without redistributing protected data. TAIX-Ray
is an ungated, publicly hosted CC BY 4.0 resource with a direct mapping to the
MedRAX classifier component, so it is the strongest choice for this scoped
audit.

## Integrity rule used throughout

The public repository distinguishes **completed, directly verified runs** from
**protocols that are ready for a future run**. It does not convert unavailable
checkpoints, inaccessible datasets, or unreproduced scaling claims into
performance statements.
