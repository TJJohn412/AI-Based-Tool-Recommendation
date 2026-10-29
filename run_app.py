from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv" / "Scripts" / "python.exe"
APP_PATH = ROOT / "app" / "app.py"

if __name__ == "__main__":
    if not VENV.exists():
        print("Virtual environment not found. Create it first with: python -m venv .venv")
        sys.exit(1)
    if not APP_PATH.exists():
        print(f"App file not found: {APP_PATH}")
        sys.exit(1)
    import subprocess

    subprocess.run([str(VENV), "-m", "streamlit", "run", str(APP_PATH)], check=False)
