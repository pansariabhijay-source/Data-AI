"""
CLI runner script — convenience wrapper around main.py.

Usage:
    python scripts/run_pipeline.py --data data/sample_classification.csv --target species
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main import main

if __name__ == "__main__":
    main()
