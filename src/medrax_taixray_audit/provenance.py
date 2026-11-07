"""Immutable provenance constants for the planned audit."""

from __future__ import annotations

import platform
import sys
from typing import Any


MEDRAX_REPOSITORY = "https://github.com/bowang-lab/MedRAX"
MEDRAX_COMMIT = "d243cc50348fa0f0c219cc623e2cf26b906be676"
PATCH_FORK = "https://github.com/asharma391/MedRAX"

TAIXRAY_DATASET = "https://huggingface.co/datasets/TLAIM/TAIX-Ray"
TAIXRAY_HF_REVISION = "7f43720e2c7ab6400608880c0cb8b950ac0e5afb"
TAIXRAY_DOI = "https://doi.org/10.1038/s41597-026-07271-7"
TAIXRAY_TEST_ROWS = 42_928
TAIXRAY_SPLIT = "test"

XRV_WEIGHT_IDENTIFIER = "densenet121-res224-all"
SCORE_SEMANTICS = (
    "TorchXRayVision operating-point-normalized model score in [0, 1]; "
    "not a calibrated probability."
)


def runtime_provenance() -> dict[str, Any]:
    """Capture versions that materially affect an inference run."""
    import numpy
    import sklearn
    import torch
    import torchxrayvision
    import torchvision

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchxrayvision": torchxrayvision.__version__,
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
    }
