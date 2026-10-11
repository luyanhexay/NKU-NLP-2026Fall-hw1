"""Six small measurements that replace six passages of hedging.

The report used to end several arguments with "we did not run the experiment
that would settle this".  Most of those were cheap to settle, so this script
runs them and the prose can state the result in a sentence instead.

  E1  class composition of each length bin (the >1000 bin drives a claim)
  E2  the bag-of-words matched curve on a denser grid, to see whether the
      gap's non-monotonicity comes from that curve being flat
  E3  GloVe's train-minus-test gap against a leak-free baseline
  E4  how separable the three embedding spaces are, independently of the
      classifier that is fitted on top of them
  E5  where the class-discriminative words actually sit in a document
  E6  where BERT sends the sports documents it misclassifies

    uv run python -m scripts.hedge_experiments
"""

import json
import pickle

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.config import CACHE_DIR, DATA_DIR, LABELS, RESULTS_DIR
from src.data import load_split, load_splits
from src.features import (
    bow_vectorizer, load_glove, mean_document_vectors, token_coverage,
)
from src.linear import fit_lr
from src.metrics import evaluate
from src.text import tokenize

MODEL_NAME = "google-bert/bert-base-uncased"
BINS = [(0, 200), (200, 400), (400, 700), (700, 1000), (1000, 10 ** 9)]
BIN_LABELS = ["<200", "200-400", "400-700", "700-1000", ">1000"]


def tokens_of(name):
    return pickle.loads((CACHE_DIR / f"tokens_nyt_{name}.pkl").read_bytes())


# --------------------------------------------------------------------------
# E1 - what is in the longest bin
# --------------------------------------------------------------------------

def e1_long_documents_are_sports():
    test = load_split("test")
    lengths = [len(t) for t in tokens_of("test")]
    frame = pd.DataFrame({"label": test["label"], "n": lengths})
    out = {}
    for (lo, hi), label in zip(BINS, BIN_LABELS):
        part = frame[(frame["n"] > lo) & (frame["n"] <= hi)]
        if len(part):
            out[label] = {lab: int((part["label"] == lab).sum()) for lab in LABELS}
    return out


# --------------------------------------------------------------------------
# E2 - the matched bag-of-words curve, denser
# --------------------------------------------------------------------------

def e2_matched_curve_denser():
    """The gap widened between the first two matched points; is the cause that
    the bag-of-words curve is flat there, or that BERT's is not?"""
    train, val, test = load_splits()
    tr, va, te = (tokens_of(n) for n in ("train", "val", "test"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def covered(doc_tokens, length):
        ids = tokenizer(doc_tokens, is_split_into_words=True, truncation=False,
                        add_special_tokens=True).word_ids()[:length]
        last = {}
        for pos, word in enumerate(ids):
            if word is not None:
                last[word] = pos
        return sum(1 for p in last.values() if p < length)

    lengths = [64, 96, 128, 192]
    counts = {}
    for L in lengths:
        counts[L] = {"train": [covered(t, L) for t in tr],
                     "val": [covered(t, L) for t in va],
                     "test": [covered(t, L) for t in te]}

    curve = []
    for L in lengths:
        parts = {}
        for name, docs in (("train", tr), ("val", va), ("test", te)):
            parts[name] = [toks[:max(counts[L][name][i], 1)]
                           for i, toks in enumerate(docs)]
        vec = bow_vectorizer(binary=True, min_df=1)
        model, _, _ = fit_lr(vec.fit_transform(parts["train"]), train["label"],
                             vec.transform(parts["val"]), val["label"], LABELS)
        record = evaluate(f"dense_{L}", test["label"],
                          model.predict(vec.transform(parts["test"])),
                          meta={"wordpieces": L,
                                "mean_words": float(np.mean(counts[L]["test"]))})
        curve.append({"wordpieces": L,
                      "mean_words": float(np.mean(counts[L]["test"])),
                      "bow_macro_f1": record["macro_f1"]})
        print()
    return curve


# --------------------------------------------------------------------------
# E3 - is GloVe's test score inflated by its pretraining corpus?
# --------------------------------------------------------------------------

def e3_train_test_gap():
    """Is GloVe's test score inflated by its own pretraining corpus?

    Bag-of-words is a bad reference here: with 127k features on 9k documents it
    memorises the training split outright (train accuracy 1.0000), so its
    train-test gap is large for a reason that has nothing to do with leakage.
    The right reference is Word2Vec trained on the training split only, which
    provably never saw a test document.  If GloVe's corpus contained our test
    text, GloVe would look relatively better on test than that reference does.
    """
    train, val, test = load_splits()
    tr, va, te = (tokens_of(n) for n in ("train", "val", "test"))

    spaces = {
        "glove": load_glove(DATA_DIR / "glove.6B.100d.txt"),
        "w2v_nyt_train_only": gensim_vectors_cached(tr),
    }
    out = {}
    for name, vecs in spaces.items():
        Xtr, _ = mean_document_vectors(tr, vecs)
        Xva, _ = mean_document_vectors(va, vecs)
        Xte, _ = mean_document_vectors(te, vecs)
        model, _, _ = fit_lr(Xtr, train["label"], Xva, val["label"], LABELS, scale=True)
        train_acc = float((model.predict(Xtr) == train["label"]).mean())
        test_acc = float((model.predict(Xte) == test["label"]).mean())
        out[name] = {"train_accuracy": train_acc, "test_accuracy": test_acc,
                     "test_over_train": test_acc / train_acc}
    return out


# --------------------------------------------------------------------------
# E4 - how separable is each embedding space, before any classifier
# --------------------------------------------------------------------------

def e4_class_separation():
    """A classifier on top can hide a poorly shaped space.  This measures the
    space itself: the ratio of between-class scatter to within-class scatter
    of the document vectors, with no classifier involved."""
    train = load_split("train")
    tr = tokens_of("train")
    from gensim.models import Word2Vec

    ag_tokens = tokenize(pd.read_csv(DATA_DIR / "ag.csv")["text"], "ag_text")
    spaces = {
        "glove": load_glove(DATA_DIR / "glove.6B.100d.txt"),
        "w2v_nyt": gensim_vectors_cached(tr),
        "w2v_ag": gensim_vectors_cached(ag_tokens, key="ag"),
    }
    out = {}
    for name, vecs in spaces.items():
        X, _ = mean_document_vectors(tr, vecs)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        overall = X.mean(axis=0)
        between = 0.0
        within = 0.0
        for lab in LABELS:
            part = X[(train["label"] == lab).to_numpy()]
            centre = part.mean(axis=0)
            between += len(part) * float(((centre - overall) ** 2).sum())
            within += float(((part - centre) ** 2).sum())
        out[name] = {"between_over_within": between / max(within, 1e-9)}
    return out


def gensim_vectors_cached(tokens, key="nyt"):
    """Train the same Word2Vec the report uses, or reuse a saved copy."""
    from gensim.models import Word2Vec
    path = CACHE_DIR / f"w2v_{key}.model"
    if path.exists():
        model = Word2Vec.load(str(path))
    else:
        model = Word2Vec(tokens, vector_size=100, window=5, min_count=5,
                         sg=1, negative=5, epochs=5, workers=8, seed=42)
        model.save(str(path))
    return {w: model.wv[w] for w in model.wv.index_to_key}


# --------------------------------------------------------------------------
# E5 - where do the discriminative words actually sit?
# --------------------------------------------------------------------------

def e5_position_of_discriminative_words():
    """The report says the informative part of a sports story tends to sit
    past the truncation point.  This measures it instead of asserting it."""
    train, val, test = load_splits()
    tr, te = tokens_of("train"), tokens_of("test")

    vec = bow_vectorizer(binary=True, min_df=1)
    X = vec.fit_transform(tr)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(X, train["label"])
    inverse = {i: w for w, i in vec.vocabulary_.items()}

    # the terms that most favour each class, skipping the short stop-list
    drop = {"mr.", "ms.", "its", "or", "new", "large", "state", "house", "care"}
    tops = {}
    for ci, lab in enumerate(clf.classes_):
        coef = clf.coef_[ci]
        picked = []
        for idx in np.argsort(coef)[::-1]:
            term = inverse[idx]
            if term in drop or len(term) < 3 or any(c.isdigit() for c in term):
                continue
            picked.append(term)
            if len(picked) == 50:
                break
        tops[lab] = set(picked)

    per_class = {lab: {"docs": 0, "anything_past_64": 0, "first_hit_past_64": 0,
                       "hits_in_head": 0, "hits_total": 0}
                 for lab in LABELS}
    for tokens, lab in zip(te, test["label"]):
        wanted = tops[lab]
        hits = [i for i, t in enumerate(tokens) if t in wanted]
        stats = per_class[lab]
        stats["docs"] += 1
        stats["hits_total"] += len(hits)
        stats["hits_in_head"] += sum(1 for i in hits if i < 64)
        if any(i >= 64 for i in hits):
            stats["anything_past_64"] += 1
        if hits and min(hits) >= 64:
            stats["first_hit_past_64"] += 1
    for lab, stats in per_class.items():
        stats["doc_share_with_hits_past_64"] = stats["anything_past_64"] / max(stats["docs"], 1)
        stats["hit_share_in_first_64"] = stats["hits_in_head"] / max(stats["hits_total"], 1)
    return per_class


# --------------------------------------------------------------------------
# E6 - where BERT sends the sports articles it gets wrong
# --------------------------------------------------------------------------

def e6_bert_sports_errors():
    path = RESULTS_DIR / "bert_test_predictions.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    wrong = frame[(frame["label"] == "sports") & (frame["prediction"] != "sports")]
    share = wrong["prediction"].value_counts().to_dict()
    n_sports = int((frame["label"] == "sports").sum())
    return {"n_sports_docs": n_sports, "n_errors": int(len(wrong)),
            "predicted_as": {k: int(v) for k, v in share.items()}}


if __name__ == "__main__":
    results = {}
    print("=== E1 length bins by class ===")
    results["E1_bin_composition"] = e1_long_documents_are_sports()
    for k, v in results["E1_bin_composition"].items():
        print(f"  {k:>10}: {v}")

    print("\n=== E6 where BERT sends misclassified sports ===")
    results["E6_bert_sports_errors"] = e6_bert_sports_errors()
    print("  ", results["E6_bert_sports_errors"])

    print("\n=== E3 train vs test accuracy ===")
    results["E3_train_test_gap"] = e3_train_test_gap()
    for k, v in results["E3_train_test_gap"].items():
        print(f"  {k:>12}: train {v['train_accuracy']:.4f}  test {v['test_accuracy']:.4f}  gap {v['gap']:+.4f}")

    print("\n=== E4 embedding-space separability (no classifier) ===")
    results["E4_class_separation"] = e4_class_separation()
    for k, v in results["E4_class_separation"].items():
        print(f"  {k:>10}: between/within = {v['between_over_within']:.3f}")

    print("\n=== E5 position of discriminative words ===")
    results["E5_word_positions"] = e5_position_of_discriminative_words()
    for k, v in results["E5_word_positions"].items():
        print(f"  {k:>9}: {v['doc_share_with_hits_past_64']*100:.1f}% of docs have a hit past 64; "
              f"{v['hit_share_in_first_64']*100:.1f}% of hits are in the first 64")

    print("\n=== E2 denser matched curve (bag-of-words only) ===")
    results["E2_dense_curve"] = e2_matched_curve_denser()
    for row in results["E2_dense_curve"]:
        print(f"  {row['wordpieces']:>4} wp -> {row['mean_words']:.0f} words: "
              f"BoW {row['bow_macro_f1']:.4f}")

    (RESULTS_DIR / "hedge_experiments.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {RESULTS_DIR / 'hedge_experiments.json'}")
