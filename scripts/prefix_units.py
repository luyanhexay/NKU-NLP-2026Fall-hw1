"""How many words does a BERT prefix of L wordpieces actually cover?

The prefix comparison in the report puts BERT and bag-of-words at the same
nominal prefix length.  They are not the same unit: the bag-of-words sweep
truncates a list of nltk word tokens, while BERT's max_length counts wordpieces,
and one word can become several wordpieces.  So "the same N" does not mean "the
same first N words", and without this conversion the comparison's premise is
unstated rather than established.

Here we tokenise each document with nltk, feed the token list to BERT's
tokenizer with is_split_into_words=True, and use the tokenizer's word_ids() to
map each wordpiece back to the word it came from.  For a prefix of L
wordpieces we then count how many words are *completely* inside the prefix -
the same information a bag-of-words model sees when it is truncated to that
many words.

    uv run python -m scripts.prefix_units
"""

import json
import statistics

from transformers import AutoTokenizer

from src.config import RESULTS_DIR
from src.data import load_split
from src.text import tokenize

MODEL_NAME = "google-bert/bert-base-uncased"
LENGTHS = [64, 128, 256, 512]


def words_covered(word_ids, length):
    """Words whose every wordpiece lies inside the first `length` positions."""
    prefix = word_ids[:length]
    seen, complete = {}, set()
    for position, word in enumerate(prefix):
        if word is None:                       # [CLS] / [SEP]
            continue
        if word not in seen:
            seen[word] = [position, position]
        else:
            seen[word][1] = position
    # A word is fully covered if its last wordpiece is inside the prefix.
    for word, (first, last) in seen.items():
        if last < length:
            complete.add(word)
    return len(complete)


def main():
    test = load_split("test")
    tokens = tokenize(test["text"], "nyt_test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    per_length = {length: [] for length in LENGTHS}
    wordpiece_counts = []
    for doc_tokens in tokens:
        encoding = tokenizer(doc_tokens, is_split_into_words=True,
                             truncation=False, add_special_tokens=True)
        ids = encoding.word_ids()
        wordpiece_counts.append(len(ids))
        for length in LENGTHS:
            per_length[length].append(words_covered(ids, length))

    print(f"documents: {len(tokens)}")
    print(f"mean wordpieces per document: {statistics.mean(wordpiece_counts):.0f}")
    print(f"{'wordpieces':>11} | {'words covered (mean)':>21} | {'median':>7}")
    summary = {}
    for length in LENGTHS:
        counts = per_length[length]
        mean = statistics.mean(counts)
        median = statistics.median(counts)
        summary[length] = {"mean_words": round(mean, 1),
                           "median_words": int(median)}
        print(f"{length:>11} | {mean:>21.1f} | {median:>7.0f}")

    (RESULTS_DIR / "prefix_units.json").write_text(
        json.dumps({"mean_wordpieces_per_doc": round(statistics.mean(wordpiece_counts), 1),
                    "wordpieces_to_words": summary}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {RESULTS_DIR / 'prefix_units.json'}")


if __name__ == "__main__":
    main()
