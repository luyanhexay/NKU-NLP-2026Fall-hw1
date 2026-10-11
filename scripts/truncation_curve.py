"""How much topical information does a document's head carry?

This exists to test one specific claim made before the experiments: that the
inverted-pyramid convention of news writing puts enough of the topic signal in
the first 64 tokens that truncating there costs little.  BERT is the only
model restricted to a document's head, so comparing BERT against models that
read everything confounds two things - the representation and the prefix.

Here the confound is removed.  We take the *same* classifier (binary
bag-of-words + logistic regression, the strongest classical configuration and
one with no randomness at all) and feed it only the first N tokens of each
document, sweeping N.  Whatever the curve loses as N shrinks is the cost of
reading a prefix, measured with the representation held fixed.

    uv run python -m scripts.truncation_curve
"""

import json
import pickle

from src.config import CACHE_DIR, LABELS, RESULTS_DIR
from src.data import load_splits
from src.features import bow_vectorizer
from src.linear import fit_lr
from src.metrics import evaluate

# None means "the whole document", the ceiling the curve is read against.
LENGTHS = [16, 32, 64, 128, 256, 512, None]
FULL_NAME = "bow_binary"          # the reference run the curve is compared to


def main():
    train, val, test = load_splits()
    # Reuse the tokenisation Task 1 already cached rather than re-tokenising.
    token_lists = {
        name: pickle.loads((CACHE_DIR / f"tokens_nyt_{name}.pkl").read_bytes())
        for name in ("train", "val", "test")
    }

    reference = json.loads((RESULTS_DIR / f"{FULL_NAME}.json").read_text(encoding="utf-8"))
    full_score = reference["macro_f1"]
    print(f"reference: {FULL_NAME} on whole documents -> macro-F1 {full_score:.4f}\n")

    curve = []
    for length in LENGTHS:
        label = "full" if length is None else str(length)
        # Truncate to a prefix, then build the vocabulary on the truncated
        # training text, so the comparison stays internally consistent.
        parts = {
            name: [tokens if length is None else tokens[:length] for tokens in lists]
            for name, lists in token_lists.items()
        }

        vectorizer = bow_vectorizer(binary=True, min_df=1)
        X_train = vectorizer.fit_transform(parts["train"])
        model, val_f1, chosen_c = fit_lr(
            X_train, train["label"],
            vectorizer.transform(parts["val"]), val["label"],
            LABELS,
        )
        predictions = model.predict(vectorizer.transform(parts["test"]))

        record = evaluate(
            f"trunc_{label}", test["label"], predictions,
            meta={"prefix_tokens": length, "classifier": "binary BoW + LR", "C": chosen_c},
        )
        curve.append({"tokens": length, "label": label,
                      "accuracy": record["accuracy"], "macro_f1": record["macro_f1"]})
        print()

    (RESULTS_DIR / "truncation_curve.json").write_text(
        json.dumps({"reference_macro_f1": full_score, "curve": curve}, indent=2),
        encoding="utf-8",
    )

    print(f"{'tokens':>8} | {'Acc':>7} | {'Macro-F1':>8} | {'loss vs full':>12}")
    for point in curve:
        loss = point["macro_f1"] - full_score
        print(f"{point['label']:>8} | {point['accuracy']:>7.4f} | "
              f"{point['macro_f1']:>8.4f} | {loss:>+12.4f}")
    print(f"\nwrote {RESULTS_DIR / 'truncation_curve.json'}")


if __name__ == "__main__":
    main()
