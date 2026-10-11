"""Materialise the single NYT 80/10/10 split shared by every experiment.

Run once.  All six experiments read these files, so they cannot silently
disagree about which document belongs to which split.

    uv run python -m scripts.make_splits
"""

import json

import numpy as np
import pandas as pd

from src.config import DATA_DIR, LABELS, SEED, SPLITS_DIR
from src.metrics import evaluate


def main():
    frame = pd.read_csv(DATA_DIR / "nyt.csv")
    print(f"NYT: {len(frame)} documents, labels={list(frame['label'].unique())}")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(frame))

    n_train = int(0.80 * len(frame))
    n_val = int(0.10 * len(frame))
    boundaries = {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }

    summary = {"seed": SEED, "n_total": len(frame), "splits": {}}
    for name, index in boundaries.items():
        part = frame.iloc[index].reset_index(drop=True)
        part.insert(0, "idx", index)          # keep the original row id for traceability
        part.to_csv(SPLITS_DIR / f"{name}.csv", index=False)

        counts = part["label"].value_counts().to_dict()
        summary["splits"][name] = {"n": len(part), "label_counts": counts,
                                   "indices": [int(i) for i in index]}
        print(f"{name:>5}: n={len(part):5d}  {counts}")

    (SPLITS_DIR / "meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Reference point: the majority class of the training split, applied to
    # the test split.  With 75% of NYT in one class this scores well above
    # chance on accuracy while carrying no information at all, so every
    # reported accuracy in the report is read against it.
    train = pd.read_csv(SPLITS_DIR / "train.csv")
    test = pd.read_csv(SPLITS_DIR / "test.csv")
    majority = train["label"].value_counts().idxmax()
    evaluate(
        "baseline_majority",
        test["label"],
        [majority] * len(test),
        meta={"rule": f"always predict '{majority}'", "labels": LABELS},
    )


if __name__ == "__main__":
    main()
