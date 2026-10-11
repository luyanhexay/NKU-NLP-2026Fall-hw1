"""Task 2 - document vectors from word embeddings, three ways.

    glove   : public 100-d GloVe vectors (glove.6B.100d), no training
    w2v_ag  : 100-d Word2Vec trained on AG News
    w2v_nyt : 100-d Word2Vec trained on the NYT *training split only*

All three go through the same mean-pooling and the same Logistic Regression,
so the only thing that varies is where the vectors came from.

    uv run python -m scripts.task2_embed --variant glove
    uv run python -m scripts.task2_embed --variant w2v_ag
    uv run python -m scripts.task2_embed --variant w2v_nyt
"""

import argparse
import shutil
import time
import urllib.request
import zipfile

import pandas as pd
from tqdm import tqdm

from src.config import DATA_DIR, LABELS, SEED
from src.data import load_splits
from src.features import (
    EMBED_DIM,
    gensim_vectors,
    load_glove,
    mean_document_vectors,
    token_coverage,
)
from src.linear import fit_lr
from src.metrics import evaluate
from src.text import tokenize

GLOVE_URLS = (
    "https://nlp.stanford.edu/data/glove.6B.zip",
    "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
)
GLOVE_MEMBER = "glove.6B.100d.txt"

def w2v_kwargs(seed):
    return dict(
        vector_size=EMBED_DIM, window=5, min_count=5,
        sg=1, negative=5, epochs=5, workers=8, seed=seed,
    )


# --------------------------------------------------------------------------
# GloVe acquisition
# --------------------------------------------------------------------------

def _download(url, destination, attempts=6, timeout=30):
    """Download `url`, resuming a partial file and verifying the byte count.

    Two failure modes urllib will not report on its own drive this: it does
    not raise when a server closes the connection early (`read` just returns
    b"" and the loop ends, so a truncated download looks successful), and
    without an explicit timeout a stalled connection blocks forever.  The
    size check and the timeout below are what turn those into a retry.
    """
    print(f"downloading {url}")
    for attempt in range(1, attempts + 1):
        have = destination.stat().st_size if destination.exists() else 0
        request = urllib.request.Request(url)
        if have:
            request.add_header("Range", f"bytes={have}-")

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if have and response.status != 206:      # server ignored the range
                    have = 0
                remaining = int(response.headers.get("content-length", 0))
                expected = have + remaining
                with tqdm(total=expected, initial=have, unit="B", unit_scale=True,
                          desc=destination.name) as bar:
                    with open(destination, "ab" if have else "wb") as fh:
                        while chunk := response.read(1 << 20):
                            fh.write(chunk)
                            bar.update(len(chunk))
            written = destination.stat().st_size
            if expected and written != expected:
                raise OSError(f"short download: {written} of {expected} bytes")
            return
        except Exception as error:                       # noqa: BLE001 - retry below
            print(f"  attempt {attempt}/{attempts} failed: {error}")
            time.sleep(2)

    raise RuntimeError(f"could not download {url} after {attempts} attempts")


def ensure_glove():
    """Download glove.6B.zip once and extract only the 100-d file."""
    target = DATA_DIR / GLOVE_MEMBER
    if target.exists():
        return target

    archive = DATA_DIR / "glove.6B.zip"
    if not (archive.exists() and zipfile.is_zipfile(archive)):
        # A partial file is deliberately kept: _download resumes from its
        # length with a Range request, so an interrupted attempt is not lost.
        last_error = None
        for url in GLOVE_URLS:
            try:
                _download(url, archive)
                if not zipfile.is_zipfile(archive):
                    raise zipfile.BadZipFile(f"{archive.name} is not a valid zip")
                break
            except Exception as error:                   # noqa: BLE001 - fall through to the mirror
                last_error = error
                print(f"  failed: {error}")
        else:
            archive.unlink(missing_ok=True)
            raise RuntimeError(f"could not download GloVe: {last_error}")

    with zipfile.ZipFile(archive) as zf:
        with zf.open(GLOVE_MEMBER) as source, open(target, "wb") as destination:
            shutil.copyfileobj(source, destination)
    print(f"extracted {target}")
    return target


# --------------------------------------------------------------------------
# The three vector sources
# --------------------------------------------------------------------------

def build_glove():
    vectors = load_glove(ensure_glove(), dim=EMBED_DIM)
    print(f"GloVe: {len(vectors)} vectors, dim={EMBED_DIM}")
    return vectors, {
        "embedding": "glove.6B.100d (pre-trained, unmodified)",
        "training_corpus": "Wikipedia 2014 + Gigaword 5 (as distributed)",
        "vocab": len(vectors),
    }


def build_w2v_ag(seed):
    from gensim.models import Word2Vec

    ag_tokens = tokenize(pd.read_csv(DATA_DIR / "ag.csv")["text"], "ag_text")
    print(f"training Word2Vec on AG News: {len(ag_tokens)} documents")

    model = Word2Vec(ag_tokens, **w2v_kwargs(seed))
    print(f"  vocabulary {len(model.wv)} words")
    return gensim_vectors(model), {
        "embedding": "Word2Vec(skip-gram) trained on AG News",
        "training_corpus": "AG News, 90000 documents",
        "vocab": len(model.wv),
        "w2v_seed": seed,
    }


def build_w2v_nyt(train_tokens, seed):
    from gensim.models import Word2Vec

    print(f"training Word2Vec on the NYT training split: {len(train_tokens)} documents")
    model = Word2Vec(train_tokens, **w2v_kwargs(seed))
    print(f"  vocabulary {len(model.wv)} words")
    return gensim_vectors(model), {
        "embedding": "Word2Vec(skip-gram) trained on NYT",
        "training_corpus": "NYT training split only (9215 documents)",
        "vocab": len(model.wv),
        "w2v_seed": seed,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("glove", "w2v_ag", "w2v_nyt"), required=True)
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="seed for Word2Vec training; GloVe has no randomness so it ignores this",
    )
    parser.add_argument("--suffix", default="", help="appended to the result filename")
    args = parser.parse_args()

    train, val, test = load_splits()
    train_tokens = tokenize(train["text"], "nyt_train")
    val_tokens = tokenize(val["text"], "nyt_val")
    test_tokens = tokenize(test["text"], "nyt_test")

    print(f"\n=== {args.variant} (seed={args.seed}) ===")
    # Only the NYT variant depends on the training split; the other two ignore it.
    if args.variant == "w2v_nyt":
        vectors, meta = build_w2v_nyt(train_tokens, args.seed)
    elif args.variant == "w2v_ag":
        vectors, meta = build_w2v_ag(args.seed)
    else:
        vectors, meta = build_glove()

    X_train, coverage_train = mean_document_vectors(train_tokens, vectors)
    X_val, _ = mean_document_vectors(val_tokens, vectors)
    X_test, _ = mean_document_vectors(test_tokens, vectors)

    # How much of the corpus the embedding actually saw.  A document with no
    # in-vocabulary token collapses to the zero vector, so this bounds every
    # score reported below.
    token_cov = token_coverage(train_tokens, vectors)
    empty_docs = int((coverage_train == 0).sum())
    meta.update({
        "dim": EMBED_DIM,
        "pooling": "mean of in-vocabulary word vectors",
        "token_coverage_train": round(float(token_cov), 4),
        "train_docs_with_no_known_token": empty_docs,
    })
    print(f"    token coverage={token_cov:.4f}  empty documents={empty_docs}/{len(train_tokens)}")

    model, val_f1, chosen_c = fit_lr(
        X_train, train["label"], X_val, val["label"], LABELS, scale=True
    )

    evaluate(
        f"embed_{args.variant}{args.suffix}",
        test["label"],
        model.predict(X_test),
        meta={
            **meta,
            "classifier": "LogisticRegression(lbfgs) on standardised vectors, C tuned on val",
            "C": chosen_c,
            "val_macro_f1": float(val_f1),
        },
    )


if __name__ == "__main__":
    main()
