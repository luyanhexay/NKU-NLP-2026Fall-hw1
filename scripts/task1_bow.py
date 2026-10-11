"""Task 1 - Binary bag-of-words and word-frequency bag-of-words.

Both representations share one vocabulary, one tokenisation and one
classifier; only the value stored per (document, term) pair differs.

    uv run python -m scripts.task1_bow                 # min_df=1, the reported run
    uv run python -m scripts.task1_bow --min-df 5 --suffix _mindf5
"""

import argparse

from src.config import LABELS
from src.data import load_splits
from src.features import bow_vectorizer
from src.linear import fit_lr
from src.metrics import evaluate
from src.text import tokenize

REPRESENTATIONS = (
    (True, "bow_binary", "binary bag-of-words"),
    (False, "bow_freq", "word-frequency bag-of-words"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-df", type=int, default=1,
        help="drop terms occurring in fewer than this many training documents "
             "(1 = the literal whole-corpus vocabulary)",
    )
    parser.add_argument("--suffix", default="", help="appended to the result filename")
    args = parser.parse_args()

    train, val, test = load_splits()
    train_tokens = tokenize(train["text"], "nyt_train")
    val_tokens = tokenize(val["text"], "nyt_val")
    test_tokens = tokenize(test["text"], "nyt_test")

    for binary, name, description in REPRESENTATIONS:
        print(f"\n=== {name} (min_df={args.min_df}) ===")

        vectorizer = bow_vectorizer(binary=binary, min_df=args.min_df)
        X_train = vectorizer.fit_transform(train_tokens)   # vocabulary from train only
        X_val = vectorizer.transform(val_tokens)
        X_test = vectorizer.transform(test_tokens)

        vocab_size = len(vectorizer.vocabulary_)
        print(f"    |V| = {vocab_size}  (fitted on the training split only)")

        model, val_f1, chosen_c = fit_lr(
            X_train, train["label"], X_val, val["label"], LABELS
        )

        evaluate(
            f"{name}{args.suffix}",
            test["label"],
            model.predict(X_test),
            meta={
                "representation": description,
                "classifier": "LogisticRegression(lbfgs), C tuned on val",
                "tokenizer": "nltk.word_tokenize",
                "min_df": args.min_df,
                "vocab_size": vocab_size,
                "C": chosen_c,
                "val_macro_f1": float(val_f1),
            },
        )


if __name__ == "__main__":
    main()
