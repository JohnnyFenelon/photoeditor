"""Image discovery and loading, including optional RAW decoding.

Everything here is defensive: a single corrupt or unsupported file must never
crash a batch (PRD non-functional requirement: 0% crash rate on bad files).
"""

from __future__ import annotations

import os
from typing import Optional

from PIL import Image, ImageOps

# Standard formats handled directly by Pillow.
STANDARD_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

# RAW formats handled via rawpy (if installed).
RAW_EXTS = {".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf"}

SUPPORTED_EXTS = STANDARD_EXTS | RAW_EXTS

try:  # rawpy is optional; absence simply disables RAW decoding.
    import rawpy  # type: ignore

    HAS_RAWPY = True
except Exception:  # pragma: no cover - depends on environment
    HAS_RAWPY = False


def is_supported(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        return HAS_RAWPY
    return ext in STANDARD_EXTS


def is_raw(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTS


def scan_folder(folder: str) -> list[str]:
    """Return a sorted list of supported image paths in *folder* (non-recursive)."""
    try:
        entries = os.listdir(folder)
    except OSError:
        return []

    paths = []
    for name in entries:
        full = os.path.join(folder, name)
        if os.path.isfile(full) and is_supported(full):
            paths.append(full)
    paths.sort(key=lambda p: os.path.basename(p).lower())
    return paths


def load_image(path: str, max_size: Optional[int] = None) -> Image.Image:
    """Load *path* as an RGB :class:`PIL.Image`.

    Args:
        path: image file path (standard or RAW).
        max_size: if given, downscale so the longest edge is at most this many
            pixels. Used for fast scoring and lightweight previews; ``None``
            loads the full-resolution image for final export.

    Raises:
        Exception: propagated to the caller, which is expected to catch it so a
            single bad file does not abort a batch.
    """
    if is_raw(path):
        img = _load_raw(path)
    else:
        img = Image.open(path)
        # Honour EXIF orientation so portrait shots are not scored sideways.
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

    if max_size is not None:
        img = downscale(img, max_size)
    return img


def _load_raw(path: str) -> Image.Image:
    if not HAS_RAWPY:
        raise RuntimeError("RAW support requires the 'rawpy' package.")
    with rawpy.imread(path) as raw:  # type: ignore[union-attr]
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False)
    return Image.fromarray(rgb).convert("RGB")


def downscale(img: Image.Image, max_size: int) -> Image.Image:
    """Return a copy scaled so the longest edge <= *max_size* (never upscales)."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_size:
        return img
    scale = max_size / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)
