"""Compare BERT and bag-of-words on genuinely the same words.

The earlier version of this comparison put the two models at the same nominal
N, but N counts nltk words on one side and BERT wordpieces on the other, and a
word can become several wordpieces.  At N=64 BERT therefore reads about 54
words, not 64, so "the same first N words" was an unstated assumption rather
than an established premise.

This script removes it per document, not on average: for document i and BERT
length L we find how many of *that document's* words lie completely inside the
first L wordpieces, and truncate the bag-of-words input for document i to
exactly that many words.  The two models then see the same text.

    uv run python -m scripts.matched_prefix
"""

import json
import pickle

import numpy as np
from transformers import AutoTokenizer

from src.config import CACHE_DIR, LABELS, RESULTS_DIR
from src.data import load_split
from src.features import bow_vectorizer
from src.linear import fit_lr
from src.metrics import evaluate
from src.text import tokenize

MODEL_NAME = "google-bert/bert-base-uncased"
LENGTHS = [64, 128, 256, 512]


def covered_words(word_ids, length):
    """Count words whose every wordpiece falls inside the first `length`."""
    last_position = {}
    for position, word in enumerate(word_ids[:length]):
        if word is not None:
            last_position[word] = position
    return sum(1 for last in last_position.values() if last < length)


def per_document_lengths(texts_tokens, tokenizer):
    """covered[i][L] = number of words of document i inside a prefix of L."""
    result = []
    for doc_tokens in texts_tokens:
        encoding = tokenizer(doc_tokens, is_split_into_words=True,
                             truncation=False, add_special_tokens=True)
        ids = encoding.word_ids()
        result.append({L: covered_words(ids, L) for L in LENGTHS})
    return result


def main():
    train, val, test = load_splits_and_tokens()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    covered = {
        "train": per_document_lengths(train[1], tokenizer),
        "val": per_document_lengths(val[1], tokenizer),
        "test": per_document_lengths(test[1], tokenizer),
    }

    rows = []
    for length in LENGTHS:
        bert_path = RESULTS_DIR / ("bert.json" if length == 64 else f"bert_len{length}.json")
        if not bert_path.exists():
            continue
        bert = json.loads(bert_path.read_text(encoding="utf-8"))

        # Truncate *per document* to the words BERT actually sees at this length.
        def truncate(tokens, counts):
            return [toks[:max(counts[i][length], 1)] for i, toks in enumerate(tokens)]

        parts = {name: truncate(tokens, covered[name])
                 for name, (_, tokens) in (("train", train), ("val", val), ("test", test))}

        vectorizer = bow_vectorizer(binary=True, min_df=1)
        X_train = vectorizer.fit_transform(parts["train"])
        model, _, _ = fit_lr(X_train, train[0]["label"],
                             vectorizer.transform(parts["val"]), val[0]["label"], LABELS)
        predictions = model.predict(vectorizer.transform(parts["test"]))

        record = evaluate(
            f"matched_{length}", test[0]["label"], predictions,
            meta={"bert_prefix_wordpieces": length,
                  "mean_words_seen": round(float(np.mean([c[length] for c in covered["test"]])), 1),
                  "classifier": "binary BoW + LR, truncated per document"},
        )
        rows.append((length, record["macro_f1"], bert["macro_f1"]))
        print()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{真正同输入下的对比。词袋一侧按\emph{每篇文档}实际被 BERT 前缀覆盖"
        r"到的词数截断，而不是按同一个名义 $N$，因为 $N$ 在两侧分别是词与子词。}",
        r"\label{tab:matched}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        "BERT 前缀（子词） & 覆盖词数 & 词袋 & BERT & 差值 \\\\",
        r"\midrule",
    ]
    for length, bow_score, bert_score in rows:
        mean_words = json.loads((RESULTS_DIR / f"matched_{length}.json")
                                .read_text(encoding="utf-8"))["meta"]["mean_words_seen"]
        delta = round(bert_score, 4) - round(bow_score, 4)
        lines.append(f"{length} & {mean_words:.0f} & {bow_score:.4f} & {bert_score:.4f} & {delta:+.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (RESULTS_DIR.parent / "report" / "tables" / "matched.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    print(f"{'prefix':>7} | {'words seen':>10} | {'BoW':>7} | {'BERT':>7} | {'gap':>8}")
    for length, bow_score, bert_score in rows:
        mean_words = json.loads((RESULTS_DIR / f"matched_{length}.json")
                                .read_text(encoding="utf-8"))["meta"]["mean_words_seen"]
        print(f"{length:>7} | {mean_words:>10.0f} | {bow_score:>7.4f} | "
              f"{bert_score:>7.4f} | {bert_score - bow_score:>+8.4f}")


def load_splits_and_tokens():
    """Split frames paired with their cached nltk tokens."""
    out = []
    for name in ("train", "val", "test"):
        frame = load_split(name)
        tokens = pickle.loads((CACHE_DIR / f"tokens_nyt_{name}.pkl").read_bytes())
        out.append((frame, tokens))
    return tuple(out)


if __name__ == "__main__":
    main()
