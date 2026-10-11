"""Loading of the single NYT split that every experiment shares.

The split is materialised on disk by `scripts/make_splits.py`; this module
only reads it.  Nothing here re-splits, so no experiment can accidentally
work on a different train/val/test partition.
"""

import pandas as pd

from .config import SPLITS_DIR

SPLIT_NAMES = ("train", "val", "test")


def load_split(name):
    """Read one split by name ('train' | 'val' | 'test')."""
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing - run `uv run python -m scripts.make_splits` first"
        )
    return pd.read_csv(path)


def load_splits():
    """Return (train, val, test) as DataFrames."""
    return tuple(load_split(n) for n in SPLIT_NAMES)
