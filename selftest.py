"""Headless smoke test for the FO_TOJ engine (no GUI).

Generates a batch of synthetic images of varying sharpness/exposure, runs the
scorer, verifies the sharpest well-exposed image wins, then applies every
filter and round-trips a save. Exits non-zero on any failure.
"""

import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageFilter

from fo_toj import image_io
from fo_toj.filters import FILTER_NAMES, apply_filter
from fo_toj.scoring import Scorer


def make_test_batch(folder: str) -> str:
    """Create several images; return the path of the intended 'best' one."""
    rng = np.random.default_rng(42)

    # A sharp, well-exposed, high-detail image (should win).
    base = rng.integers(40, 215, size=(600, 800, 3), dtype=np.uint8)
    sharp = Image.fromarray(base, "RGB")
    sharp_path = os.path.join(folder, "sharp_good.jpg")
    sharp.save(sharp_path, quality=95)

    # Blurry version of the same content.
    blurry = sharp.filter(ImageFilter.GaussianBlur(6))
    blurry.save(os.path.join(folder, "blurry.jpg"), quality=95)

    # Too dark.
    dark = Image.fromarray((base * 0.15).astype(np.uint8), "RGB")
    dark.save(os.path.join(folder, "underexposed.jpg"), quality=95)

    # Too bright / blown out.
    bright = Image.fromarray(np.clip(base.astype(np.int16) + 120, 0, 255).astype(np.uint8), "RGB")
    bright.save(os.path.join(folder, "overexposed.jpg"), quality=95)

    # A corrupt file that must be skipped, not crash.
    with open(os.path.join(folder, "corrupt.jpg"), "wb") as f:
        f.write(b"not a real jpeg")

    # An unsupported file type that must be ignored by the scan.
    with open(os.path.join(folder, "notes.txt"), "w") as f:
        f.write("ignore me")

    return sharp_path


def main() -> int:
    failures = []

    with tempfile.TemporaryDirectory() as folder:
        expected_best = make_test_batch(folder)

        # --- scan ---
        paths = image_io.scan_folder(folder)
        print(f"[scan] found {len(paths)} supported images")
        if len(paths) != 5:  # 4 valid jpgs + 1 corrupt jpg (still .jpg)
            failures.append(f"expected 5 image paths, got {len(paths)}")

        # --- score ---
        scorer = Scorer(prefer_ai=True)
        print(f"[scorer] backend={scorer.backend} model={scorer.model_name}")

        best_path, best_score = None, float("-inf")
        scored, skipped = 0, 0
        for p in paths:
            try:
                img = image_io.load_image(p, max_size=1024)
                s = scorer.score(img)
                scored += 1
                print(f"   {os.path.basename(p):20s} -> {s:.3f}")
                if s > best_score:
                    best_score, best_path = s, p
            except Exception as e:
                skipped += 1
                print(f"   {os.path.basename(p):20s} -> SKIPPED ({e.__class__.__name__})")

        print(f"[score] scored={scored} skipped={skipped} best={os.path.basename(best_path or '')}")
        if skipped != 1:
            failures.append(f"expected 1 skipped (corrupt) file, got {skipped}")
        if best_path != expected_best:
            failures.append(
                f"best should be {os.path.basename(expected_best)}, "
                f"got {os.path.basename(best_path or 'None')}"
            )

        # --- filters ---
        full = image_io.load_image(expected_best)
        for name in FILTER_NAMES:
            out = apply_filter(full, name)
            if out.size != full.size:
                failures.append(f"filter '{name}' changed size {full.size}->{out.size}")
            if out.mode != "RGB":
                failures.append(f"filter '{name}' produced mode {out.mode}")
            print(f"[filter] {name:18s} ok  size={out.size}")

        # --- save round-trip (jpg q95 + png) ---
        for ext, name in ((".jpg", "Warm Sunset"), (".png", "Black & White")):
            out_path = os.path.join(folder, f"result{ext}")
            final = apply_filter(full, name)
            if ext == ".png":
                final.save(out_path, format="PNG")
            else:
                final.save(out_path, format="JPEG", quality=95, subsampling=0)
            reloaded = Image.open(out_path)
            reloaded.load()
            print(f"[save] {ext} ok  ({os.path.getsize(out_path)} bytes)")
            if reloaded.size != full.size:
                failures.append(f"saved {ext} size mismatch")

    print("\n" + ("=" * 50))
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
