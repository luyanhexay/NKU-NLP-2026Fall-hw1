"""Project paths, constants and the global seed.

Every experiment imports its paths and its class order from here so that the
split, the label ordering and the random seed can never drift apart.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"
FIGS_DIR = ROOT / "report" / "figs"

# One seed for the split, one for the models.  Every experiment uses the
# same split files, so results are comparable regardless of the seed.
SEED = 42

# Sorted class order, used for both the label->id mapping and every
# per-class report.  Sorted (not frequency-ordered) so the confusion
# matrices of different experiments line up.
LABELS = ["business", "politics", "sports"]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}

for _d in (SPLITS_DIR, RESULTS_DIR, CACHE_DIR, FIGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
