"""Tokenisation with nltk, cached on disk.

Tokenising the 11.5k NYT documents takes minutes while every downstream
representation is seconds, so the token lists are cached once per corpus and
reused by Task 1 and Task 2.
"""

import io
import pickle
import urllib.request
import zipfile

import nltk

from .config import CACHE_DIR

# nltk's own downloader refuses to fetch through a proxy ("Security
# Violation [pathsec.urlopen]"), which blocks any machine behind one.  The
# packages are plain zips on GitHub, so if the downloader cannot run we fetch
# them ourselves into a project-local nltk_data directory.
_PUNKT_PACKAGES = ("punkt", "punkt_tab")
_NLTK_DATA_DIR = CACHE_DIR / "nltk_data"
_NLTK_PACKAGE_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/{name}.zip"
)


def _has(package):
    try:
        nltk.data.find(f"tokenizers/{package}")
        return True
    except LookupError:
        return False


def _fetch_directly(package):
    """Download and unpack one nltk_data package without nltk's downloader."""
    print(f"  fetching nltk resource '{package}' directly")
    destination = _NLTK_DATA_DIR / "tokenizers"
    destination.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(_NLTK_PACKAGE_URL.format(name=package)) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(destination)


def _ensure_punkt():
    """Make sure the tokenizer data is available, however we can get it."""
    if str(_NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(_NLTK_DATA_DIR))

    for package in _PUNKT_PACKAGES:
        if _has(package):
            continue
        try:
            nltk.download(package, quiet=True)
        except Exception:                          # noqa: BLE001 - fall back below
            pass
        if not _has(package):
            _fetch_directly(package)
        if not _has(package):
            raise RuntimeError(f"could not obtain the nltk resource '{package}'")


def tokenize(texts, cache_key):
    """Return a list of token lists, memoised under `cache_key`.

    The cache key names the corpus (e.g. "nyt_train"), not the caller, so a
    second experiment over the same corpus reuses the first one's tokens.

    Punctuation is kept: the assignment names `word_tokenize` as the
    tokenizer and does not ask for further filtering, so the only pruning in
    this project happens through the vocabulary's `min_df` in Task 1.
    """
    path = CACHE_DIR / f"tokens_{cache_key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())

    _ensure_punkt()
    from nltk import word_tokenize

    tokens = [word_tokenize(text) for text in texts]
    path.write_bytes(pickle.dumps(tokens))
    return tokens
