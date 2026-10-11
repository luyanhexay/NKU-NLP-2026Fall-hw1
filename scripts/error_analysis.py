"""Where the two models' mistakes fall, by document length, and a worked example.

Two things the report was missing.  The prefix sweep shows *how much* signal
truncation costs; it does not show *which documents* pay it.  Stratifying the
test set by length answers that, and a single concrete misclassification makes
the mechanism readable in a way rates cannot.

    uv run python -m scripts.error_analysis
"""

import json
import pickle

import pandas as pd
from transformers import AutoTokenizer

from src.config import CACHE_DIR, RESULTS_DIR
from src.data import load_split
from src.text import tokenize

MODEL_NAME = "google-bert/bert-base-uncased"

# Bins in nltk words, chosen to match the corpus's shape (median ~672).
BINS = [(0, 200), (200, 400), (400, 700), (700, 1000), (1000, 10**9)]
BIN_LABELS = ["<200", "200--400", "400--700", "700--1000", ">1000"]


def words_covered_by_prefix(tokenizer, doc_tokens, prefix=64):
    encoding = tokenizer(doc_tokens, is_split_into_words=True,
                         truncation=False, add_special_tokens=True)
    ids = encoding.word_ids()[:prefix]
    last = {}
    for position, word in enumerate(ids):
        if word is not None:
            last[word] = position
    return sum(1 for pos in last.values() if pos < prefix)


def main():
    test = load_split("test")
    tokens = tokenize(test["text"], "nyt_test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bert = pd.read_csv(RESULTS_DIR / "bert_test_predictions.csv")
    bow = pd.read_csv(RESULTS_DIR / "bow_binary_test_predictions.csv")
    merged = bert.merge(bow, on="idx", suffixes=("_bert", "_bow"))
    assert (merged["label_bert"] == merged["label_bow"]).all()

    lengths = [len(t) for t in tokens]
    covered = [words_covered_by_prefix(tokenizer, t) for t in tokens]

    frame = pd.DataFrame({
        "idx": test["idx"].to_numpy(),
        "length": lengths,
        "seen_fraction": [c / max(n, 1) for c, n in zip(covered, lengths)],
        "label": merged["label_bert"].to_numpy(),
        "bert_wrong": (merged["prediction_bert"] != merged["label_bert"]).to_numpy(),
        "bow_wrong": (merged["prediction_bow"] != merged["label_bow"]).to_numpy(),
    })

    rows = []
    for (low, high), label in zip(BINS, BIN_LABELS):
        part = frame[(frame["length"] > low) & (frame["length"] <= high)]
        if part.empty:
            continue
        rows.append({
            "bin": label,
            "docs": len(part),
            "seen": part["seen_fraction"].mean(),
            "bow_err": part["bow_wrong"].mean(),
            "bert_err": part["bert_wrong"].mean(),
        })

    table = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{测试集按文档长度分层。``可见比例''是 BERT 的 64 个子词前缀"
        r"平均覆盖到的文档比例。两个模型的错误率都以该层文档数为分母。}",
        r"\label{tab:by-length}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        "长度（词） & 篇数 & 可见 & 词袋错 & BERT 错 \\\\",
        r"\midrule",
    ]
    for row in rows:
        table.append(f"{row['bin']} & {row['docs']} & {row['seen']*100:.1f}\\% & "
                     f"{row['bow_err']*100:.1f}\\% & {row['bert_err']*100:.1f}\\% \\\\")
    total = frame
    table.append(r"\midrule")
    table.append(f"全部 & {len(total)} & {total['seen_fraction'].mean()*100:.1f}\\% & "
                 f"{total['bow_wrong'].mean()*100:.1f}\\% & {total['bert_wrong'].mean()*100:.1f}\\% \\\\")
    table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (RESULTS_DIR.parent / "report" / "tables" / "by_length.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8")

    print(f"{'bin':>10} | {'docs':>5} | {'seen':>6} | {'BoW err':>8} | {'BERT err':>9}")
    for row in rows:
        print(f"{row['bin']:>10} | {row['docs']:>5} | {row['seen']*100:>5.1f}% | "
              f"{row['bow_err']*100:>7.1f}% | {row['bert_err']*100:>8.1f}%")
    print(f"{'all':>10} | {len(total):>5} | {total['seen_fraction'].mean()*100:>5.1f}% | "
          f"{total['bow_wrong'].mean()*100:>7.1f}% | {total['bert_wrong'].mean()*100:>8.1f}%")

    # Candidates for a worked example: BERT wrong, bag-of-words right, long
    # document, and the true class is the one BERT over-predicts least - i.e.
    # a case where the missing tail plausibly holds the decisive words.
    candidates = frame[(frame["bert_wrong"]) & (~frame["bow_wrong"])].copy()
    candidates = candidates.sort_values("seen_fraction")   # sees the least
    print(f"\nBERT 错而词袋对的文档：{len(candidates)} 篇")
    for _, row in candidates.head(8).iterrows():
        print(f"  idx={row['idx']:>5} len={row['length']:>5} saw={row['seen_fraction']*100:>5.1f}% "
              f"true={row['label']}")
    candidates.to_csv(RESULTS_DIR / "error_candidates.csv", index=False)

    (RESULTS_DIR / "error_by_length.json").write_text(
        json.dumps({"bins": rows,
                    "all": {"docs": len(total),
                            "seen": float(total["seen_fraction"].mean()),
                            "bow_err": float(total["bow_wrong"].mean()),
                            "bert_err": float(total["bert_wrong"].mean())}},
                   indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS_DIR / 'error_by_length.json'}")


if __name__ == "__main__":
    main()
