"""Document representations: bag-of-words and mean-pooled word embeddings.

Both families reduce a document to a single fixed-length vector that a
Logistic Regression can consume, which is what makes them comparable in the
report.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

EMBED_DIM = 100


# --------------------------------------------------------------------------
# Task 1 - bag of words
# --------------------------------------------------------------------------

def bow_vectorizer(binary, min_df):
    """A CountVectorizer that consumes pre-tokenised documents.

    `analyzer=lambda doc: doc` treats each input element (a token list) as
    already analysed, so nltk's tokenisation is the only tokenisation that
    happens.  The vectoriser must be fitted on the training split only:
    fitting on the full corpus would leak the test vocabulary.
    """
    return CountVectorizer(
        analyzer=lambda tokens: tokens,
        binary=binary,
        min_df=min_df,
        lowercase=True,
        dtype=np.float32,
    )


# --------------------------------------------------------------------------
# Task 2 - word embeddings
# --------------------------------------------------------------------------

def load_glove(path, dim=EMBED_DIM):
    """Read a GloVe text file into {word: np.ndarray(dim)}."""
    vectors = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def gensim_vectors(model):
    """Expose a trained gensim model through the same {word: vector} interface."""
    return {word: model.wv[word] for word in model.wv.index_to_key}


def mean_document_vectors(token_lists, vectors, dim=EMBED_DIM):
    """Average the embeddings of the in-vocabulary words of each document.

    Returns (matrix, coverage) where `coverage[i]` is how many tokens of
    document i had a vector.  A document whose every token is out of
    vocabulary becomes the zero vector, which is why the coverage counts are
    reported: they bound how much of the corpus each embedding actually saw.
    """
    matrix = np.zeros((len(token_lists), dim), dtype=np.float32)
    coverage = np.zeros(len(token_lists), dtype=np.int32)

    for i, tokens in enumerate(token_lists):
        total = np.zeros(dim, dtype=np.float32)
        found = 0
        for token in tokens:
            vector = vectors.get(token)
            if vector is not None:
                total += vector
                found += 1
        if found:
            matrix[i] = total / found
        coverage[i] = found

    return matrix, coverage


def token_coverage(token_lists, vectors):
    """Fraction of all tokens (not documents) that have a vector."""
    total = sum(len(t) for t in token_lists)
    hit = sum(1 for tokens in token_lists for token in tokens if token in vectors)
    return hit / total if total else 0.0
