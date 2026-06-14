# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for FO_TOJ -> standalone Windows .exe.

Build with:   pyinstaller fo_toj.spec
Output:       dist/FO_TOJ.exe  (single-file, windowed)

customtkinter ships data files (themes/assets) that PyInstaller cannot detect
automatically, so we collect them explicitly. torch/pyiqa are intentionally
NOT bundled — they are optional and huge; the app falls back to the heuristic
scorer when they are absent (and most end-users won't have them installed).
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("customtkinter")
hiddenimports = collect_submodules("customtkinter")

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Exclude the heavy optional AI stack from the frozen exe.
    excludes=["torch", "torchvision", "pyiqa", "scipy", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FO_TOJ",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # windowed app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
