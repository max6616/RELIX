#!/usr/bin/env python3
"""Run RELIX directly from this source checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "RELIX"))
from relix.cli import main

if __name__ == "__main__":
    main()
