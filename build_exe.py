"""Build the standalone FO_TOJ.exe with PyInstaller.

Usage:
    py -m pip install pyinstaller
    py build_exe.py

Produces dist/FO_TOJ.exe — a single-file windowed executable that needs no
Python install on the target machine (PRD packaging requirement).
"""

import subprocess
import sys


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed. Run:  py -m pip install pyinstaller")
        return 1

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "fo_toj.spec"]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
