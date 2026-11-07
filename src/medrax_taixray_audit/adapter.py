"""A focused adapter for the exact public XRV classifier used by MedRAX."""

from __future__ import annotations

from hashlib import sha256
from numbers import Real
from pathlib import Path
from typing import Optional, Union

import numpy as np
import skimage.io
import torch
import torchvision
import torchxrayvision as xrv

from .provenance import SCORE_SEMANTICS, XRV_WEIGHT_IDENTIFIER


def resolve_device(device: Union[str, torch.device, None] = "auto") -> torch.device:
    """Prefer CUDA, then MPS, then CPU for this DenseNet-only adapter."""
    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device auto instead.")
    if resolved.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS requested but unavailable; use --device auto instead.")
    return resolved


TAIXRAY_UINT16_RULE = "taixray_uint16_range_v1"


def normalization_maxval(
    image: np.ndarray, input_maxval: Optional[Union[Real, str]] = None
) -> float:
    """Return a safe, declared scale for TorchXRayVision normalization.

    This preserves the upstream 8-bit path. For uint16 (or other wider unsigned)
    images, callers must supply the acquisition scale; a uint16 container cannot
    distinguish full-range 16-bit from 12-bit values stored in uint16.
    """
    image = np.asarray(image)
    if image.size == 0:
        raise ValueError("Input image is empty.")

    if input_maxval == TAIXRAY_UINT16_RULE:
        if image.dtype != np.uint16:
            raise ValueError("The TAIX-Ray range rule requires a uint16 image.")
        # The public PNG export contains both observed 12-bit-valued images
        # (0--4095) and images using the wider uint16 range. This is a fixed
        # encoding rule, not per-image contrast scaling: it selects a declared
        # conventional scale, never the image's observed maximum.
        return 4095.0 if image.max() <= 4095 else 65535.0

    if input_maxval is not None:
        if not isinstance(input_maxval, Real) or not np.isfinite(input_maxval):
            raise ValueError("input_maxval must be finite and positive.")
        if input_maxval <= 0:
            raise ValueError("input_maxval must be positive.")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > input_maxval:
            raise ValueError("Input pixels are incompatible with the declared input_maxval.")
        return float(input_maxval)

    if image.dtype == np.uint8:
        return 255.0
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Float images must be finite and scaled to [0, 1].")
        return 1.0
    if np.issubdtype(image.dtype, np.unsignedinteger):
        raise ValueError(
            "Non-8-bit unsigned images require input_maxval (for example, 4095 for "
            "known 12-bit values stored in uint16)."
        )
    raise TypeError(f"Unsupported image dtype: {image.dtype}")


class MedRAXClassifierAdapter:
    """Faithfully execute MedRAX's embedded XRV DenseNet classifier path.

    This is intentionally not a full MedRAX-agent evaluation: it evaluates only
    the classifier selected by the upstream tool and the public preprocessing
    sequence relevant to its image input.
    """

    def __init__(
        self,
        device: Union[str, torch.device, None] = "auto",
        input_maxval: Optional[Union[Real, str]] = None,
        model_name: str = XRV_WEIGHT_IDENTIFIER,
    ) -> None:
        self.device = resolve_device(device)
        self.input_maxval = input_maxval
        self.model_name = model_name
        self.model = xrv.models.DenseNet(weights=model_name).to(self.device).eval()
        self.transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

    def preprocess(self, image_path: Path) -> tuple[torch.Tensor, float]:
        image = skimage.io.imread(str(image_path))
        if image.ndim > 2:
            image = image[:, :, 0]
        maxval = normalization_maxval(image, self.input_maxval)
        image = xrv.datasets.normalize(image, maxval)
        image = self.transform(image[None, :, :])
        return torch.from_numpy(image).unsqueeze(0).to(self.device), maxval

    def predict(self, image_path: Path) -> tuple[dict[str, float], float]:
        """Return the 18 model scores and the declared image intensity scale."""
        image, maxval = self.preprocess(image_path)
        with torch.inference_mode():
            scores = self.model(image).detach().cpu()[0].numpy()
        return (
            {
                pathology: float(score)
                for pathology, score in zip(xrv.datasets.default_pathologies, scores)
            },
            maxval,
        )

    def model_file_sha256(self) -> Optional[str]:
        """Hash a locally cached XRV checkpoint when its path can be located."""
        cache = Path.home() / ".torchxrayvision" / "models_data"
        matches = sorted(cache.glob("*densenet121*.pt"))
        if not matches:
            return None
        digest = sha256()
        with matches[0].open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @property
    def score_semantics(self) -> str:
        return SCORE_SEMANTICS
