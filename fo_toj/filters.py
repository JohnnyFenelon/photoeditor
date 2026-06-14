"""The 10 one-click filters defined in the PRD (section 4.3).

Each filter is a pure function ``Image -> Image`` operating on an RGB image, so
the exact same code runs on the small preview and on the full-resolution image
at save time. Filters never mutate their input.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Public, ordered list of filter names for the dropdown.
FILTER_NAMES = [
    "Original",
    "Black & White",
    "Sepia Vintage",
    "Soft Dream",
    "Crisp Sharpen",
    "Warm Sunset",
    "Cool Cinematic",
    "High Contrast",
    "Brightness Boost",
    "Dramatic Vignette",
]


def _original(img: Image.Image) -> Image.Image:
    return img.copy()


def _black_white(img: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.25)  # high-contrast mono
    return gray.convert("RGB")


def _sepia(img: Image.Image) -> Image.Image:
    gray = np.asarray(ImageOps.grayscale(img), dtype=np.float32)
    # Classic warm sepia tone mapping.
    r = np.clip(gray * 1.07 + 20, 0, 255)
    g = np.clip(gray * 0.74 + 10, 0, 255)
    b = np.clip(gray * 0.43, 0, 255)
    out = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _soft_dream(img: Image.Image) -> Image.Image:
    radius = max(1.5, min(img.size) / 400.0)  # scale blur with image size
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    # Blend a touch of the sharp original back for a soft-focus glow.
    out = Image.blend(blurred, img, alpha=0.35)
    return ImageEnhance.Brightness(out).enhance(1.05)


def _crisp_sharpen(img: Image.Image) -> Image.Image:
    out = img.filter(
        ImageFilter.UnsharpMask(radius=2.0, percent=150, threshold=2)
    )
    return ImageEnhance.Contrast(out).enhance(1.05)


def _warm_sunset(img: Image.Image) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * 1.15 + 12, 0, 255)  # red up
    arr[..., 1] = np.clip(arr[..., 1] * 1.05, 0, 255)        # green slight
    arr[..., 2] = np.clip(arr[..., 2] * 0.92, 0, 255)        # blue down
    out = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return ImageEnhance.Brightness(out).enhance(1.08)


def _cool_cinematic(img: Image.Image) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * 0.93, 0, 255)        # red down
    arr[..., 2] = np.clip(arr[..., 2] * 1.18 + 8, 0, 255)    # blue up
    out = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return ImageEnhance.Contrast(out).enhance(1.18)


def _high_contrast(img: Image.Image) -> Image.Image:
    out = ImageEnhance.Contrast(img).enhance(1.6)
    return ImageEnhance.Color(out).enhance(1.1)


def _brightness_boost(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(1.2)  # +20% exposure


def _dramatic_vignette(img: Image.Image) -> Image.Image:
    w, h = img.size
    # Radial falloff mask: bright centre -> dark edges.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    norm = dist / max_dist
    # Keep the centre untouched, darken progressively toward corners.
    mask = np.clip(1.0 - (norm ** 2.2) * 0.85, 0.15, 1.0)
    arr = np.asarray(img, dtype=np.float32)
    arr *= mask[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


# Name -> function map. Keep keys identical to FILTER_NAMES.
FILTERS = {
    "Original": _original,
    "Black & White": _black_white,
    "Sepia Vintage": _sepia,
    "Soft Dream": _soft_dream,
    "Crisp Sharpen": _crisp_sharpen,
    "Warm Sunset": _warm_sunset,
    "Cool Cinematic": _cool_cinematic,
    "High Contrast": _high_contrast,
    "Brightness Boost": _brightness_boost,
    "Dramatic Vignette": _dramatic_vignette,
}


def apply_filter(img: Image.Image, name: str) -> Image.Image:
    """Apply the named filter to *img*, returning a new image.

    Unknown names fall back to ``Original`` so the UI can never wedge.
    """
    func = FILTERS.get(name, _original)
    return func(img)
