"""
Shared path helpers for notebooks/Final_Figures scripts so they all agree on
where the project root and data tables live.
"""

import os
from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_default_data_root = PROJECT_ROOT / "data"
DATA_ROOT = Path(os.environ.get("FINAL_FIGURES_DATA_ROOT", _default_data_root))


def data_path(relative: Union[str, Path]) -> Path:
    """Convenience helper to build paths inside the data directory."""
    return DATA_ROOT / Path(relative)

