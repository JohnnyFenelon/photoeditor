"""Image Quality Assessment scoring.

Two strategies, selected automatically at runtime:

1. **AI (preferred):** a pre-trained no-reference IQA model via ``pyiqa``
   (MUSIQ / NIMA). Captures aesthetic + technical quality.
2. **Heuristic fallback:** a pure OpenCV / NumPy score combining sharpness
   (Laplacian variance), exposure (how close mean brightness is to mid-tones),
   and contrast (pixel standard deviation). Used whenever the AI stack is
   unavailable or fails to load, so the app never crashes (PRD FR-2.3).

Both strategies expose a single :class:`Scorer.score` method returning a float
where higher is better. Scores are only meaningful *relative* to other images
scored by the same strategy.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Longest edge used when scoring. Smaller = faster; quality cues survive.
SCORE_MAX_SIZE = 1024


class Scorer:
    """Scores image quality, preferring AI and falling back to heuristics."""

    def __init__(self, prefer_ai: bool = True):
        self.backend = "heuristic"
        self.model_name = "OpenCV/NumPy heuristic"
        self._iqa = None
        self._torch = None
        if prefer_ai:
            self._try_load_ai()

    # ------------------------------------------------------------------ AI ---
    def _try_load_ai(self) -> None:
        """Attempt to load a pyiqa model. Any failure leaves the heuristic."""
        try:
            import torch  # type: ignore
            import pyiqa  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
            # MUSIQ is a strong no-reference aesthetic+technical model. If its
            # weights cannot be fetched, try the lighter NIMA, then give up.
            last_err: Exception | None = None
            for name in ("musiq", "nima"):
                try:
                    self._iqa = pyiqa.create_metric(name, device=device)
                    self.model_name = f"pyiqa:{name} ({device})"
                    self.backend = "ai"
                    self._torch = torch
                    return
                except Exception as exc:  # weights download / arch issue
                    last_err = exc
            if last_err:
                raise last_err
        except Exception:
            # Stay on heuristic; this is an expected, supported path.
            self.backend = "heuristic"
            self.model_name = "OpenCV/NumPy heuristic"
            self._iqa = None
            self._torch = None

    @property
    def is_ai(self) -> bool:
        return self.backend == "ai"

    # --------------------------------------------------------------- score ---
    def score(self, img: Image.Image) -> float:
        """Return a quality score for *img* (higher is better)."""
        if self._iqa is not None and self._torch is not None:
            try:
                return self._score_ai(img)
            except Exception:
                # Degrade gracefully mid-batch rather than abort.
                pass
        return self._score_heuristic(img)

    def _score_ai(self, img: Image.Image) -> float:
        import torchvision.transforms.functional as TF  # type: ignore

        tensor = TF.to_tensor(img).unsqueeze(0)
        with self._torch.no_grad():  # type: ignore[union-attr]
            value = self._iqa(tensor)  # type: ignore[misc]
        return float(value.item())

    # ----------------------------------------------------------- heuristic ---
    @staticmethod
    def _score_heuristic(img: Image.Image) -> float:
        """Blend sharpness, exposure and contrast into one score."""
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

        sharpness = _laplacian_variance(gray)          # focus / detail
        mean = float(gray.mean())
        std = float(gray.std())                          # contrast
        # Exposure: peak at mid grey (~128), penalise blown / crushed frames.
        exposure = 1.0 - abs(mean - 128.0) / 128.0       # 0..1

        # Normalise the unbounded cues to comparable ranges, then weight them.
        # Weights favour sharpness (the dominant culling signal) while still
        # rewarding well-exposed, punchy images.
        sharp_norm = np.log1p(sharpness)                 # compress large range
        contrast_norm = std / 64.0                       # ~0..~2
        return 0.6 * sharp_norm + 0.25 * (exposure * 5.0) + 0.15 * contrast_norm


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the classic focus/sharpness measure.

    Uses OpenCV when available (faster, well-tested), otherwise a small NumPy
    convolution so the heuristic works with zero extra dependencies.
    """
    try:
        import cv2  # type: ignore

        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except Exception:
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap = _convolve2d(gray, kernel)
        return float(lap.var())


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Minimal 'valid' 2D convolution via stride tricks (no SciPy needed)."""
    kh, kw = kernel.shape
    padded = np.pad(image, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="edge")
    sub_shape = (image.shape[0], image.shape[1], kh, kw)
    strides = padded.strides * 2
    windows = np.lib.stride_tricks.as_strided(padded, sub_shape, strides)
    return np.einsum("ijkl,kl->ij", windows, kernel)
