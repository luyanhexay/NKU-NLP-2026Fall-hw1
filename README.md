# NLP Course Assignment 1 — Text Classification

Three families of document representation compared on a three-class New York
Times classification task: bag-of-words, mean-pooled word embeddings, and a
fine-tuned BERT.  The report is `report/main.pdf`; its source is
`report/main.tex`.

## What is here

```
src/                     shared library
  config.py              paths, the global seed, the class order
  data.py                reads the split written to disk
  text.py                nltk tokenisation, cached
  features.py            BoW vectorisers, GloVe loading, mean pooling
  linear.py              logistic regression with C chosen on validation
  metrics.py             accuracy + macro-F1, one JSON per experiment
scripts/
  make_splits.py         writes the single shared 80/10/10 split
  task1_bow.py           Task 1 — binary and word-frequency BoW
  task2_embed.py         Task 2 — GloVe / Word2Vec on AG News / on NYT
  task3_bert.py          Task 3 — bert-base-uncased fine-tuning
  truncation_curve.py    macro-F1 against prefix length, representation fixed
  prefix_units.py        wordpieces-to-words conversion for the BERT prefixes
  matched_prefix.py      BERT vs BoW on the same words *per document*
  error_analysis.py      error rates by document length + a worked example
  hedge_experiments.py   six small measurements that replace six hedges
  topic_diagnostics.py   vocabulary overlap, keyword frequency, year mentions
  collect_results.py     results/ -> report tables and macros
  make_figs.py           results/ -> report figures
  build_report.sh        regenerate the generated inputs, then compile the PDF
results/                 one JSON per experiment (the evidence)
report/                  LaTeX source, generated tables, figures
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The lock file points at the default PyPI index.  On a slow link (for example
behind a proxy in mainland China, where PyPI itself often stalls), install
through a mirror instead:

```bash
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ uv sync
```

A CUDA GPU is used for Task 3 if one is present and the CPU is used otherwise.
The reported Task 3 numbers were produced on an 8 GB laptop GPU.

## Data

Two CSVs are expected in `data/`:

| file | columns | rows |
| --- | --- | --- |
| `data/nyt.csv` | `text`, `label` | 11,519 |
| `data/ag.csv` | `text` | 90,000 |

They are not committed (they are course-provided and large).  `glove.6B.zip`
is downloaded into `data/` automatically on the first Task 2 GloVe run.

## Running

Run from the project root.  The split must be created first — every other
script reads it and refuses to run without it.

```bash
uv run python -m scripts.make_splits                        # once

uv run python -m scripts.task1_bow
uv run python -m scripts.task2_embed --variant glove
uv run python -m scripts.task2_embed --variant w2v_ag
uv run python -m scripts.task2_embed --variant w2v_nyt
uv run python -m scripts.task3_bert

# the measurements the discussion rests on
uv run python -m scripts.truncation_curve                   # prefix-length sweep
uv run python -m scripts.prefix_units                       # BERT prefix: wordpieces -> words
uv run python -m scripts.matched_prefix                     # BERT vs BoW, same words per doc
uv run python -m scripts.error_analysis                      # needs the two *_test_predictions.csv
uv run python -m scripts.hedge_experiments                   # length bins, leakage check, word positions
uv run python -m scripts.topic_diagnostics                  # vocabulary / frequency / year stats

uv run python -m scripts.collect_results                    # tables + macros
uv run python -m scripts.make_figs                          # figures
```

Appendix runs, referenced in the report:

```bash
uv run python -m scripts.task1_bow --min-df 5 --suffix _mindf5
uv run python -m scripts.task3_bert --max-len 128 --suffix _len128
uv run python -m scripts.task3_bert --max-len 256 --suffix _len256
uv run python -m scripts.task3_bert --max-len 512 --suffix _len512
```

Repeated runs (the report quotes ranges, not single points, for every
configuration with randomness):

```bash
for s in 7 123; do
  uv run python -m scripts.task2_embed --variant w2v_nyt --seed $s --suffix _seed$s
  uv run python -m scripts.task2_embed --variant w2v_ag  --seed $s --suffix _seed$s
  uv run python -m scripts.task3_bert --seed $s --suffix _seed$s
done
```

Each run writes one JSON to `results/`.  `collect_results.py` and
`make_figs.py` only read those files, so the report can be rebuilt without
re-training anything.

## Building the report

The report is set in the ACL conference style (`report/acl.sty`, the same
style EMNLP and NAACL use), in `preprint` mode — author shown, page numbers
on, no anonymous-review line numbers.

It is compiled with XeLaTeX rather than pdfLaTeX, for two reasons. The body
uses Times New Roman itself (`\usepackage{times}` would silently substitute
the metric-compatible Nimbus Roman clone, which does not match the figures
glyph for glyph), and the text is Chinese, which needs `xeCJK`. Times New
Roman and SimSun are present on Windows and macOS; on a Linux machine without
them, point `\setmainfont` / `\setCJKmainfont` at whatever equivalents are
installed.

```bash
./scripts/build_report.sh          # regenerates tables + figures, then the PDF
```

That script runs the compiler with `report/` as the working directory, which
is required: the source refers to `tables/` and `figs/` by relative path, so
compiling `report/main.tex` from the project root fails to find them.
Equivalently, by hand:

```bash
cd report
latexmk -xelatex -outdir=build main.tex
```

## Reproducibility notes

- **One split for everything.**  `scripts/make_splits.py` shuffles NYT under
  a fixed seed and writes `splits/{train,val,test}.csv`; nothing downstream
  re-splits.  This is what makes the six comparisons in the report valid.
- **The vocabulary is built on the training split only**, at `min_df=1` (the
  literal whole-training-corpus vocabulary the assignment describes).  The
  appendix repeats Task 1 at `min_df=5` to show the choice is not
  load-bearing.
- **Word2Vec on NYT is trained on the training split only.**  It is an
  unsupervised model, so training it on all of NYT would not leak labels, but
  it would let test documents influence the vectors used to classify them.
- **`C` is chosen on the validation split** by macro-F1; the test split is
  read once per experiment, at the end.
- **No class weighting.**  The classifier is identical across the five
  non-BERT experiments; only the representation changes.  The class imbalance
  is handled at the metric level (macro-F1) rather than by reweighting, so
  that the comparison stays clean.
- **Prefix lengths are converted before BERT is compared with bag-of-words.**
  BERT's `max_length` counts wordpieces, the bag-of-words sweep counts nltk
  words, and one word is often several wordpieces: 64 wordpieces cover about
  54 words.  `prefix_units.py` measures the conversion and `matched_prefix.py`
  truncates *each document* to the words BERT actually sees at that length, so
  the two models read the same text rather than the same nominal number.

## A note on the required BERT configuration

The assignment fixes `max_length=64`.  The median NYT document is 672 tokens,
so BERT sees roughly the first tenth of each article while the bag-of-words
and mean-pooling baselines see all of it.  The required configuration is used
as specified; the `--max-len 512` appendix run exists to separate the
representation's contribution from the truncation's.
