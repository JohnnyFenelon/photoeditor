# FO_TOJ — AI Photo Culling & One-Click Filters

FO_TOJ scans a folder of photos, automatically picks the **highest-quality
shot** using an Image Quality Assessment (IQA) model, and lets you apply one of
**10 professional filters** before exporting — turning hours of manual culling
and editing into a 60-second workflow.

Windows 10/11 desktop app, built in Python with a dark-mode CustomTkinter UI.

---

## Features

- **📁 Folder ingestion** — scan any folder for JPG, PNG, TIFF, BMP, WEBP, and
  (when `rawpy` is installed) RAW files (`.cr2`, `.nef`, `.arw`, `.dng`, …).
- **🤖 AI "Find Best Photo"** — scores every image and auto-selects the winner.
  - **Preferred:** pre-trained no-reference IQA via `pyiqa` (MUSIQ / NIMA).
  - **Graceful fallback:** if the AI stack is missing or fails to load, an
    OpenCV/NumPy heuristic (Laplacian sharpness + exposure + contrast) takes
    over automatically. **The app never crashes for lack of AI.**
- **Responsive UI** — scoring runs on a background thread with a live progress
  bar; the window never freezes.
- **🎨 10 one-click filters** — Original, Black & White, Sepia Vintage, Soft
  Dream, Crisp Sharpen, Warm Sunset, Cool Cinematic, High Contrast, Brightness
  Boost, Dramatic Vignette. Applied instantly to the preview.
- **💾 Export** — save as JPEG (quality 95) or PNG via a native Save dialog.
  Filters preview on a downscaled copy but are rendered on the **full-resolution
  image** at save time.
- **Robust** — corrupt/unsupported files are skipped, never fatal.

---

## Quick start (from source)

```powershell
# 1. Install the core dependencies
py -m pip install -r requirements.txt

# 2. Run
py run.py
#   ...or:  py -m fo_toj
```

The core install (CustomTkinter + Pillow + NumPy + OpenCV + rawpy) is enough for
full functionality using the heuristic scorer.

### Enabling the AI scorer (optional)

```powershell
py -m pip install torch pyiqa
```

On first run the MUSIQ model weights (~104 MB) download automatically and are
cached. If `torch`/`pyiqa` are unavailable or fail to load, FO_TOJ silently uses
the heuristic scorer — no configuration needed. The status bar shows which
engine is active (`pyiqa:musiq` vs `OpenCV/NumPy heuristic`).

> Verified working on Python 3.14 with `torch 2.12.0` + `pyiqa 0.1.15` (CPU). If
> a wheel isn't available for your platform, the heuristic fallback is automatic.

---

## Building the standalone .exe

```powershell
py -m pip install pyinstaller
py build_exe.py        # or:  pyinstaller fo_toj.spec
```

The result is `dist/FO_TOJ.exe` — a single-file, windowed executable that runs
on any Windows 10/11 machine **without Python installed**. The heavy optional AI
stack is excluded from the build to keep the exe small; the bundled app uses the
heuristic scorer.

---

## How it works

```
fo_toj/
├── app.py        # CustomTkinter GUI, threading, event loop  (PRD §7 wireframe)
├── scoring.py    # AI IQA + heuristic fallback scorer         (PRD §4.2)
├── filters.py    # the 10 one-click filters                   (PRD §4.3)
├── image_io.py   # folder scan, image + RAW loading, downscale (PRD §4.1)
└── __main__.py   # `python -m fo_toj`
run.py            # launcher
selftest.py       # headless engine smoke test
fo_toj.spec       # PyInstaller build spec
```

**Memory strategy:** images are scored at ≤1024px and previewed at ≤1400px to
keep RAM low; only the final export reloads and processes the full resolution.

---

## Testing

```powershell
py selftest.py
```

This builds a synthetic batch (sharp / blurry / under- / over-exposed / corrupt),
confirms the scorer picks the sharp well-exposed image and skips the corrupt
file, runs all 10 filters, and round-trips JPEG + PNG saves.

---

## Workflow

1. **Select Folder** → see the image count.
2. **AI: Find Best Photo** → watch the progress bar; the best shot loads.
3. **Apply Filter** → pick from the dropdown; preview updates live.
4. **Save Result** → choose location/format; full-res image is exported.

Built to the FO_TOJ PRD v1.0.
