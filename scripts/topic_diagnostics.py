"""Measurements the report's explanations rest on.

Three claims in the discussion were made without a supporting measurement:

  1. that the classes are "almost lexically disjoint" (used to explain why
     bag-of-words suffices here);
  2. that the discriminative words occur far above min_count (used to explain
     why the Zipfian long tail does not bite);
  3. that the corpus is news from 2013-2014 (used to explain why a 2014
     pre-trained embedding is not as domain-shifted as expected).

This script produces the numbers for all three so they can be stated as
measurements rather than assertions.

    uv run python -m scripts.topic_diagnostics
"""

import collections
import json
import pickle
import re

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import CACHE_DIR, LABELS, RESULTS_DIR
from src.data import load_split
from src.features import bow_vectorizer

TOP_TERMS = 50


def class_vocabulary(train, train_tokens):
    """How much of the *discriminative* vocabulary do the classes share?

    Overall vocabulary overlap is uninformative - every class uses "the" and
    "said".  What matters for a bag-of-words classifier is whether the terms
    that actually separate the classes are shared.  We take the top terms by
    logistic-regression weight for each class and intersect those.
    """
    vectorizer = bow_vectorizer(binary=True, min_df=2)
    X = vectorizer.fit_transform(train_tokens)
    model = LogisticRegression(C=1.0, max_iter=3000).fit(X, train["label"])
    inverse = {index: term for term, index in vectorizer.vocabulary_.items()}

    top = {}
    for class_index, class_name in enumerate(model.classes_):
        weights = model.coef_[class_index]
        best = np.argsort(weights)[-TOP_TERMS:][::-1]
        top[class_name] = {inverse[i]: float(weights[i]) for i in best}

    pairwise = {}
    for i, first in enumerate(LABELS):
        for second in LABELS[i + 1:]:
            shared = set(top[first]) & set(top[second])
            pairwise[f"{first}|{second}"] = sorted(shared)

    # For reference, the same comparison on the frequent vocabulary.
    counts = collections.Counter()
    for tokens, label in zip(train_tokens, train["label"]):
        counts.update(set(tokens))
    frequent = {
        label: {t for t in counts if counts[t] >= 200}
        for label in LABELS
    }
    return top, pairwise, {label: len(v) for label, v in frequent.items()}


def keyword_frequencies(train_tokens, keywords):
    """Raw occurrence counts, so "far above min_count" becomes a number."""
    counts = collections.Counter()
    for tokens in train_tokens:
        counts.update(tokens)
    return {word: counts[word] for word in keywords}


def keyword_document_frequencies(train_tokens, keywords):
    """How many *documents* each word appears in.

    min_df is a document-frequency threshold, so a word's occurrence count
    cannot justify a claim about which min_df would keep it.  The report made
    exactly that slip once; this is the quantity that supports the claim.
    """
    wanted = set(keywords)
    docs = collections.Counter()
    for tokens in train_tokens:
        docs.update(wanted.intersection(tokens))
    return {word: docs[word] for word in keywords}


def top_terms_for_frequency(
    train, train_tokens, per_class=4,
    drop=("mr.", "ms.", "its", "or", "new", "large", "state", "house", "care"),
):
    """The words whose frequencies we report, chosen by the model, not by hand.

    Picking them by hand invites the objection that they were chosen *for*
    being frequent, which would make "the discriminative words are frequent"
    circular.  Taking the highest-weighted terms per class keeps the selection
    independent of how often they occur.

    A few highest-weighted tokens are function words or honorifics (``mr.``,
    ``its``); they discriminate between these three sections only weakly in the
    sense we care about, so a short stop-list removes them before the cut.
    """
    vectorizer = bow_vectorizer(binary=True, min_df=2)
    X = vectorizer.fit_transform(train_tokens)
    model = LogisticRegression(C=1.0, max_iter=3000).fit(X, train["label"])
    inverse = {index: term for term, index in vectorizer.vocabulary_.items()}

    chosen = {}
    for class_index, class_name in enumerate(model.classes_):
        weights = model.coef_[class_index]
        picked = []
        for index in np.argsort(weights)[::-1]:
            term = inverse[index]
            if term in drop or any(c.isdigit() for c in term) or len(term) < 3:
                continue
            picked.append(term)
            if len(picked) == per_class:
                break
        chosen[class_name] = picked
    return chosen


def year_mentions(train):
    """Which years the corpus actually talks about."""
    years = collections.Counter()
    for text in train["text"]:
        years.update(re.findall(r"\b(19[89]\d|20[0-2]\d)\b", str(text)))
    total = sum(years.values())
    top = years.most_common(12)
    # Mass in the two years a 2014 embedding would have seen.
    recent = sum(count for year, count in years.items() if 2012 <= int(year) <= 2014)
    return {"top_years": top, "all_years": sorted(years.items()),
            "total_mentions": total,
            "share_2012_2014": round(recent / total, 4) if total else 0.0,
            "share_2010_2014": round(
                sum(c for y, c in years.items() if 2010 <= int(y) <= 2014) / total, 4
            ) if total else 0.0}


def main():
    train = load_split("train")
    train_tokens = pickle.loads((CACHE_DIR / "tokens_nyt_train.pkl").read_bytes())

    print("=== 1. discriminative vocabulary overlap ===")
    top, pairwise, frequent = class_vocabulary(train, train_tokens)
    for class_name, terms in top.items():
        print(f"  {class_name:9s} top-{TOP_TERMS}: " +
              ", ".join(list(terms)[:14]))
    for pair, shared in pairwise.items():
        print(f"  {pair:20s} shared: {len(shared):2d} {shared[:8]}")
    print(f"  frequent-vocab sizes (count>=200): {frequent}")

    print("\n=== 2. keyword frequencies in the 9215 training documents ===")
    chosen = top_terms_for_frequency(train, train_tokens)
    keywords = [w for words in chosen.values() for w in words]
    print("  chosen per class (by model weight, not frequency):", chosen)
    docs_df = keyword_document_frequencies(train_tokens, keywords)
    for word, count in keyword_frequencies(train_tokens, keywords).items():
        print(f"  {word:12s} 出现 {count:6d} 次，见于 {docs_df[word]:5d} 篇")

    print("\n=== 3. years mentioned in the corpus ===")
    years = year_mentions(train)
    print("  top years:", years["top_years"])
    print(f"  full span: {years['all_years'][0]} .. {years['all_years'][-1]} "
          f"({len(years['all_years'])} distinct years)")
    print(f"  total mentions {years['total_mentions']}, "
          f"share 2012-2014 {years['share_2012_2014']}, "
          f"share 2010-2014 {years['share_2010_2014']}")

    payload = {
        "top_terms_per_class": {c: list(t)[:20] for c, t in top.items()},
        "top_term_overlap": {pair: shared for pair, shared in pairwise.items()},
        "top_term_count": TOP_TERMS,
        "frequent_vocab_sizes": frequent,
        "keyword_frequencies": keyword_frequencies(train_tokens, keywords),
        "keyword_document_frequencies": keyword_document_frequencies(train_tokens, keywords),
        "keyword_selection": {"rule": "highest logistic-regression weight per class",
                              "per_class": 4, "chosen": chosen},
        "years": years,
    }
    (RESULTS_DIR / "topic_diagnostics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {RESULTS_DIR / 'topic_diagnostics.json'}")


if __name__ == "__main__":
    main()
