"""Import helpers for bootcampviztools from repo root."""
from pathlib import Path
import sys

# Add repo root to sys.path so notebooks in subfolders can import bootcampviztools.py
repo_root = Path.cwd()
while repo_root != repo_root.parent and not (repo_root / "bootcampviztools.py").exists():
    repo_root = repo_root.parent

if (repo_root / "bootcampviztools.py").exists():
    sys.path.insert(0, str(repo_root))
else:
    raise FileNotFoundError("bootcampviztools.py not found; open this notebook inside DS-Online")

from bootcampviztools import (
    pinta_distribucion_categoricas,
    plot_categorical_numerical_relationship,
    plot_grouped_histograms,
)
