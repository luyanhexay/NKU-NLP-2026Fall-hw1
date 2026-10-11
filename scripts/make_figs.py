"""Build the report figures from results/*.json.

    uv run python -m scripts.make_figs

Design notes, because each figure here was rebuilt once after looking at it:

* A fixed 0--1 axis is dishonest-by-omission on data that lives in a narrow
  band.  The score figure and the confusion matrices both used to do that, and
  the effect was that every method looked identical and every confusion matrix
  looked like the same dark diagonal.  The scales below are chosen so that the
  differences the text discusses are actually visible.
* One word becomes several BERT wordpieces, so a figure that puts both models on
  one x-axis has to convert units first (see scripts/prefix_units.py).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from src.config import FIGS_DIR, LABELS, RESULTS_DIR

# The figures sit next to the LaTeX body text, so they use Times New Roman
# itself rather than a metric-compatible substitute.  On WSL the Windows font
# files are the real thing; if they are absent we say so instead of silently
# substituting a different typeface.
TNR_CANDIDATES = {
    "regular": "/mnt/c/Windows/Fonts/times.ttf",
    "bold": "/mnt/c/Windows/Fonts/timesbd.ttf",
    "italic": "/mnt/c/Windows/Fonts/timesi.ttf",
    "bold_italic": "/mnt/c/Windows/Fonts/timesbi.ttf",
}

# (result name, short label) in the order the report discusses them
MAIN = [
    ("baseline_majority", "Majority\nbaseline"),
    ("bow_binary", "Binary\nBoW"),
    ("bow_freq", "Word-freq\nBoW"),
    ("embed_glove", "GloVe\n6B"),
    ("embed_w2v_ag", "W2V\nAG News"),
    ("embed_w2v_nyt", "W2V\nNYT"),
    ("bert", "BERT\n(64 subwords)"),
]
MODELS = MAIN[1:]

ACCENT = "#1f4e79"
HIGHLIGHT = "#c00000"


def use_times_new_roman():
    found = [Path(p) for p in TNR_CANDIDATES.values() if Path(p).exists()]
    if not found:
        print("WARNING: Times New Roman not found; falling back to the default serif")
        plt.rcParams["font.family"] = "serif"
        return
    for path in found:
        font_manager.fontManager.addfont(str(path))
    plt.rcParams["font.family"] = "Times New Roman"


def load(name):
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------
# Figure 1 -- scores
# --------------------------------------------------------------------------

def figure_scores():
    """Two panels, because the score data spans two very different scales.

    Panel (a) holds the majority baseline and the six models together on 0--1,
    which is the only way to see that the baseline is far below everything
    else.  On that scale the six models are one blob, so panel (b) drops the
    baseline and zooms onto the models, where the ordering is visible.
    """
    entries = [(name, label, load(name)) for name, label in MAIN]
    entries = [e for e in entries if e[2]]
    models = [e for e in entries if e[0] != "baseline_majority"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                      gridspec_kw={"width_ratios": [1, 1.55]})

    # ---- (a) everything on one axis
    xs = np.arange(len(entries))
    band_lo = min(r["macro_f1"] for _, _, r in models)
    band_hi = max(r["accuracy"] for _, _, r in models)
    left.axhspan(band_lo, band_hi, color=ACCENT, alpha=0.09, zorder=0)
    for offset, (key, marker, colour, label) in zip(
        (-0.13, 0.13),
        (("accuracy", "o", ACCENT, "Accuracy"), ("macro_f1", "s", "#e08214", "Macro-F1")),
    ):
        left.scatter(xs + offset, [r[key] for _, _, r in entries],
                     marker=marker, s=17, color=colour, label=label, zorder=3)
    left.set_xticks(xs)
    left.set_xticklabels([label for _, label, _ in entries], fontsize=6, rotation=90)
    left.set_ylim(0, 1.05)
    left.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    left.set_ylabel("score", fontsize=7.5)
    left.tick_params(labelsize=6.5)
    left.set_title("(a) all configurations", fontsize=7.5)
    left.legend(fontsize=6.5, frameon=False, loc="center right")
    left.spines[["top", "right"]].set_visible(False)

    # ---- (b) the six models, zoomed
    xs2 = np.arange(len(models))
    acc = [r["accuracy"] for _, _, r in models]
    f1 = [r["macro_f1"] for _, _, r in models]
    for offset, (values, marker, colour) in zip(
        (-0.12, 0.12),
        ((acc, "o", ACCENT), (f1, "s", "#e08214")),
    ):
        right.scatter(xs2 + offset, values, marker=marker, s=20, color=colour, zorder=3)
        for x, value in zip(xs2 + offset, values):
            right.annotate(f"{value:.4f}", (x, value), textcoords="offset points",
                           xytext=(0, 5), ha="center", fontsize=5.6)
    right.set_xticks(xs2)
    right.set_xticklabels([label for _, label, _ in models], fontsize=6)
    right.set_ylim(min(f1) - 0.004, max(acc) + 0.010)   # headroom for the labels
    right.set_title("(b) the six models, zoomed", fontsize=7.5)
    right.tick_params(labelsize=6.5)
    right.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / "scores.pdf")
    fig.savefig(FIGS_DIR / "scores.png", dpi=200)
    plt.close(fig)
    print("wrote figs/scores.pdf")


# --------------------------------------------------------------------------
# Figure 2 -- confusion matrices
# --------------------------------------------------------------------------

def figure_confusions():
    """Off-diagonal only, because the diagonal is what hides the errors.

    Row-normalised and coloured on 0--1, every panel was a dark diagonal on
    white: the cells that carry the argument - business against politics, and
    BERT's sports errors - were all below 0.10 and therefore invisible.  The
    diagonal is masked and the colour scale runs to the largest off-diagonal
    value, so the differences between panels can be seen.
    """
    records = [(label, load(name)) for name, label in MODELS]
    records = [(label, rec) for label, rec in records if rec]
    cols = 3
    rows = int(np.ceil(len(records) / cols))

    matrices = []
    for _, rec in records:
        m = np.asarray(rec["confusion_matrix"], dtype=float)
        m = m / m.sum(axis=1, keepdims=True)
        np.fill_diagonal(m, np.nan)          # the diagonal is not the story
        matrices.append(m)
    vmax = max(np.nanmax(m) for m in matrices)

    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.3 * rows))
    for index, (ax, (label, _)) in enumerate(zip(np.ravel(axes), records)):
        matrix = matrices[index]
        ax.imshow(np.ma.masked_invalid(matrix), cmap="OrRd", vmin=0, vmax=vmax)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                if i == j:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                               facecolor="0.90", edgecolor="none"))
                    continue
                value = matrix[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value > vmax * 0.6 else "black")
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels([lab[:4] for lab in LABELS], fontsize=6.5)
        ax.set_yticklabels([lab[:4] for lab in LABELS], fontsize=6.5)
        ax.set_title(label.replace("\n", " "), fontsize=8)
        # labels only on the outer edges
        if index % cols == 0:
            ax.set_ylabel("true", fontsize=7)
        else:
            ax.set_yticklabels([])
        if index >= len(records) - cols:
            ax.set_xlabel("predicted", fontsize=7)
        else:
            ax.set_xticklabels([])
        ax.tick_params(length=0)

    for ax in np.ravel(axes)[len(records):]:
        ax.axis("off")

    # a shared colour bar, so the reader knows what the shading means;
    # letting matplotlib reserve the space avoids pushing it off the canvas
    sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=0, vmax=vmax))
    bar = fig.colorbar(sm, ax=list(np.ravel(axes)[:len(records)]),
                       location="right", shrink=0.55, pad=0.02, aspect=18)
    bar.set_ticks([0, vmax / 2, vmax])
    bar.ax.set_yticklabels([f"{v:.2f}" for v in (0, vmax / 2, vmax)], fontsize=6)
    bar.set_label("share of the row", fontsize=6)
    fig.savefig(FIGS_DIR / "confusions.pdf")
    fig.savefig(FIGS_DIR / "confusions.png", dpi=200)
    plt.close(fig)
    print("wrote figs/confusions.pdf")


# --------------------------------------------------------------------------
# Figure 3 -- truncation and the matched comparison
# --------------------------------------------------------------------------

def figure_truncation():
    """Two questions, two panels, one shared y-range.

    Panel (a) is the truncation-cost sweep: the classifier is held fixed and
    *every* document is cut to the same number of words.  Panel (b) is the
    matched comparison: here each document is cut to the words BERT actually
    sees at a given max_length.  The two answer different questions and give
    different bag-of-words values at a similar word count, but both plot
    macro-F1, so they share a y-range to stay comparable.
    """
    path = RESULTS_DIR / "truncation_curve.json"
    if not path.exists():
        print("skipping figs/truncation.pdf (run scripts.truncation_curve first)")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    curve = data["curve"]
    full = data["reference_macro_f1"]

    tokens = [p["tokens"] for p in curve if p["tokens"] is not None]
    scores = [p["macro_f1"] for p in curve if p["tokens"] is not None]

    units = {}
    units_path = RESULTS_DIR / "prefix_units.json"
    if units_path.exists():
        units = json.loads(units_path.read_text(encoding="utf-8"))["wordpieces_to_words"]
    matched = []
    for length in (64, 128, 256, 512):
        bert_path = RESULTS_DIR / ("bert.json" if length == 64 else f"bert_len{length}.json")
        bow_path = RESULTS_DIR / f"matched_{length}.json"
        if bert_path.exists() and bow_path.exists():
            bert = json.loads(bert_path.read_text(encoding="utf-8"))["macro_f1"]
            bow_m = json.loads(bow_path.read_text(encoding="utf-8"))["macro_f1"]
            matched.append((units.get(str(length), {}).get("mean_words", length), bow_m, bert))

    # a shared range keeps the two panels comparable
    all_values = list(scores) + [v for _, bow_m, bert in matched for v in (bow_m, bert)]
    lo, hi = min(all_values) - 0.006, max(all_values) + 0.006

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    left.plot(tokens, scores, marker="o", markersize=3.5, linewidth=1.2, color=ACCENT)
    for x, y in zip(tokens, scores):
        left.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6),
                      ha="center", fontsize=6)
    left.axhline(full, color="0.45", linestyle=":", linewidth=1.0)
    # the annotation goes below the line, away from the 512 point label
    left.annotate(f"full document ({full:.4f})", (max(tokens), full),
                  textcoords="offset points", xytext=(-4, -11), ha="right",
                  fontsize=6, color="0.35")
    left.axvline(64, color=HIGHLIGHT, linestyle="--", linewidth=0.9)
    left.annotate("$max\\_length=64$", (64, lo), textcoords="offset points",
                  xytext=(3, 3), ha="left", va="bottom", fontsize=6, color=HIGHLIGHT)
    left.set_xscale("log", base=2)
    left.set_xticks(tokens)
    left.set_xticklabels([str(t) for t in tokens], fontsize=6.5)
    left.set_xlabel("words kept (same for every document)", fontsize=6.5)
    left.set_ylabel("macro-F1", fontsize=7.5)
    left.set_title("(a) truncation cost", fontsize=7.5)
    left.set_ylim(lo, hi)
    left.tick_params(labelsize=6.5)

    if matched:
        mx, m_bow, m_bert = zip(*matched)
        right.plot(mx, m_bow, marker="o", markersize=3.5, linewidth=1.2,
                   color=ACCENT, label="bag-of-words")
        right.plot(mx, m_bert, marker="s", markersize=3.5, linewidth=1.2,
                   color=HIGHLIGHT, linestyle="--", label="BERT")
        for x, bow_m, bert in matched:
            right.annotate(f"{round(bert, 4) - round(bow_m, 4):+.4f}",
                           (x, (bow_m + bert) / 2), textcoords="offset points",
                           xytext=(5, 0), ha="left", fontsize=6, color="0.25")
        right.set_xscale("log", base=2)
        right.set_xticks([m[0] for m in matched])
        right.set_xticklabels([f"{m[0]:.0f}" for m in matched], fontsize=6.5)
        right.set_xlabel("words seen (per document)", fontsize=6.5)
        right.set_title("(b) matched comparison", fontsize=7.5)
        right.set_ylim(lo, hi)
        right.tick_params(labelsize=6.5)
        right.legend(fontsize=6.5, frameon=False, loc="lower right")

    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "truncation.pdf")
    fig.savefig(FIGS_DIR / "truncation.png", dpi=200)
    plt.close(fig)
    print("wrote figs/truncation.pdf")


# --------------------------------------------------------------------------
# Figure 4 -- year mentions
# --------------------------------------------------------------------------

def figure_years():
    """The distribution's shape is the argument, so it gets a figure.

    As a six-cell table this data could not do its job: the table came out
    without 2014 - the year the text's claim is about - and its percentage row
    did not sum to 100 because only six of forty-seven years were shown.  A
    bar per year, ascending, shows the shape and makes the highlighted window
    readable at a glance.
    """
    path = RESULTS_DIR / "topic_diagnostics.json"
    if not path.exists():
        print("skipping figs/years.pdf (run scripts.topic_diagnostics first)")
        return
    years = json.loads(path.read_text(encoding="utf-8"))["years"]
    counts = {int(y): c for y, c in years["all_years"]}
    total = years["total_mentions"]

    span = list(range(min(counts), max(counts) + 1))
    values = [counts.get(y, 0) for y in span]
    window = range(2010, 2015)
    colors = [HIGHLIGHT if y in window else ACCENT for y in span]

    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.bar(span, values, color=colors, width=0.78)
    inside = sum(counts.get(y, 0) for y in window)
    ax.set_xlabel("year mentioned in the article", fontsize=7.5)
    ax.set_ylabel("mentions", fontsize=7.5)
    ax.set_xticks([y for y in span if y % 5 == 0])
    ax.tick_params(labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(f"2010--2014\n{inside / total * 100:.1f}% of all mentions",
                xy=(2012, max(values)), xytext=(1996, max(values) * 0.92),
                fontsize=6.5, color=HIGHLIGHT, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=HIGHLIGHT, linewidth=0.7))
    # the bulk of the mass sits before 2015, which is what the text claims
    ax.annotate("95% of all mentions fall in 2014 or earlier",
                xy=(2014, counts.get(2014, 0)), xytext=(1982, max(values) * 0.62),
                fontsize=6.5, color="0.35", ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color="0.55", linewidth=0.7))
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "years.pdf")
    fig.savefig(FIGS_DIR / "years.png", dpi=200)
    plt.close(fig)
    print("wrote figs/years.pdf")


if __name__ == "__main__":
    use_times_new_roman()
    figure_scores()
    figure_confusions()
    figure_truncation()
    figure_years()
