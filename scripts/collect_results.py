"""Turn results/*.json into every table and macro the report uses.

Nothing in report/main.tex carries a hand-copied number: the tables are
generated here and \\input by the LaTeX source, so the PDF cannot drift away
from what the experiments actually produced.

    uv run python -m scripts.collect_results
"""

import json

import pandas as pd

from src.config import LABELS, RESULTS_DIR

# Report order: reference point first, then Task 1, Task 2, Task 3, appendix.
ORDER = [
    ("baseline_majority", "平凡基线（全猜 sports）", "参照"),
    ("bow_binary", "二值词袋", "Task 1"),
    ("bow_freq", "词频词袋", "Task 1"),
    ("embed_glove", "GloVe 6B（100 维）", "Task 2"),
    ("embed_w2v_ag", "W2V（AG News，100 维）", "Task 2"),
    ("embed_w2v_nyt", "W2V（NYT，100 维）", "Task 2"),
    ("bert", "BERT-base（64 子词，3 轮）", "Task 3"),
    ("bow_binary_mindf5", "二值词袋（min_df=5）", "附录"),
    ("bow_freq_mindf5", "词频词袋（min_df=5）", "附录"),
    ("bert_len512", "BERT-base（512 子词，3 轮）", "附录"),
]

MAIN_EMBEDDINGS = ["embed_glove", "embed_w2v_ag", "embed_w2v_nyt"]
LABEL_CN = {"business": "财经", "politics": "政治", "sports": "体育"}
MAIN_SIX = [name for name, _, task in ORDER if task in ("Task 1", "Task 2", "Task 3")]


def load(name):
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def macro(record, key, default="--", fmt="{:.4f}"):
    return fmt.format(record[key]) if record else default


def tex(text):
    """Escape a label for LaTeX.  Labels are stored with plain underscores so
    the same strings can go into the CSV; escaping happens only at the point
    where they are typeset."""
    return text.replace("_", r"\_")


def main():
    records = {name: load(name) for name, _, _ in ORDER}

    # ---------------------------------------------------------------- CSV
    rows = []
    for name, label, task in ORDER:
        record = records[name]
        if record is None:
            print(f"  (missing: {name})")
            continue
        rows.append({
            "task": task,
            "experiment": label,
            "result_file": name,
            "accuracy": round(record["accuracy"], 4),
            "macro_f1": round(record["macro_f1"], 4),
            **{f"f1_{lab}": round(record["per_class_f1"][lab], 4) for lab in LABELS},
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "summary.csv", index=False)

    # ------------------------------------------------------- console table
    print(f"\n| {'Task':<8} | {'Experiment':<39} | {'Acc':>6} | {'Macro-F1':>8} |")
    for row in rows:
        print(f"| {row['task']:<8} | {row['experiment']:<39} | {row['accuracy']:>6.4f} "
              f"| {row['macro_f1']:>8.4f} |")

    tables = RESULTS_DIR.parent / "report" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ results
    _write_results_table(tables, rows)

    # ------------------------------------------------------------- task 1
    _write_task1_table(tables, records)

    # ----------------------------------------------------------- coverage
    _write_coverage_table(tables, records)

    # ------------------------------------------------------------ appendix
    _write_appendix_table(tables, records)

    # ---------------------------------------------------------- truncation
    _write_truncation_table(tables)

    # --------------------------------------------------------- diagnostics
    _write_dataset_table(tables)
    _write_discriminative_table(tables)
    _write_keyword_freq_table(tables)

    # ------------------------------------------------------------- errors
    _write_error_table(tables)

    # -------------------------------------------------------- repetitions
    _write_repetition_table(tables)

    # ------------------------------------------------------ prefix compare
    _write_prefix_compare_table(tables)

    # -------------------------------------------------------------- macros
    _write_macros(tables, records)

    print(f"\nwrote results/summary.csv and 5 files under {tables}")


def _write_results_table(tables, rows):
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        " & ".join(["任务", "配置", "准确率", "宏平均 F1"] + [f"{LABEL_CN[l]} F1" for l in LABELS]) + r" \\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        if previous is not None and row["task"] != previous:
            lines.append(r"\midrule")
        task = row["task"] if row["task"] != previous else ""
        previous = row["task"]
        cells = [task, tex(row["experiment"]), f"{row['accuracy']:.4f}", f"{row['macro_f1']:.4f}"]
        cells += [f"{row[f'f1_{lab}']:.4f}" for lab in LABELS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_task1_table(tables, records):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Task 1 的结果（末列为验证集，其余为测试集）。两种表示共用同一份含 "
        r"\resultVocabSize{} 个词的训练词表，区别只在每个位置上存什么值。}",
        r"\label{tab:task1}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "表示方法 & $C$ & 准确率 & 宏平均 F1 & 验证集宏平均 F1 \\\\",
        r"\midrule",
    ]
    for name, label in (("bow_binary", "二值词袋"), ("bow_freq", "词频词袋")):
        record = records[name]
        if record is None:
            continue
        meta = record["meta"]
        lines.append(
            f"{label} & {meta['C']:g} & "
            f"{record['accuracy']:.4f} & {record['macro_f1']:.4f} & {meta['val_macro_f1']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (tables / "task1.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_coverage_table(tables, records):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{三种词向量对 NYT 训练集的覆盖率。覆盖率界定每种词向量能起作用的"
        r"范围：一篇文档若所有词都不在词表中，其文档向量会退化为零向量，"
        r"而三种词向量下这样的文档数都是零。}",
        r"\label{tab:coverage}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        "词向量来源 & 词表规模 & 词元覆盖率 \\\\",
        r"\midrule",
    ]
    for name, label in (
        ("embed_glove", "GloVe 6B"),
        ("embed_w2v_ag", "Word2Vec（AG News）"),
        ("embed_w2v_nyt", "Word2Vec（NYT）"),
    ):
        record = records[name]
        if record is None:
            continue
        meta = record["meta"]
        lines.append(
            f"{label} & {meta['vocab']:,} & {meta['token_coverage_train']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (tables / "coverage.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_appendix_table(tables, records):
    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\footnotesize",
        r"\caption{附录实验。前两行把 Task 1 的 \texttt{min\_df} 改为 5；"
        r"最后一行把 BERT 的截断长度改为 512，用于把表示能力与截断的代价分开。}",
        r"\label{tab:appendix}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        "配置 & $\\lvert V \\rvert$ & 准确率 & 宏平均 F1 \\\\",
        r"\midrule",
    ]
    any_row = False
    for name, label in (
        ("bow_binary_mindf5", "二值词袋，min_df=5"),
        ("bow_freq_mindf5", "词频词袋，min_df=5"),
        ("bert_len512", "BERT，max_len 512"),
    ):
        record = records[name]
        if record is None:
            continue
        any_row = True
        vocab = record["meta"].get("vocab_size")
        vocab_cell = f"{vocab:,}" if vocab else "--"
        lines.append(
            f"{tex(label)} & {vocab_cell} & {record['accuracy']:.4f} & {record['macro_f1']:.4f} \\\\"
        )
    if not any_row:
        lines.append(r"\multicolumn{4}{c}{\emph{尚无附录实验记录}}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (tables / "appendix.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_truncation_table(tables):
    """The prefix-length sweep, rendered as the table the report cites."""
    path = RESULTS_DIR / "truncation_curve.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    full = data["reference_macro_f1"]

    lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        "前缀词数 & 准确率 & 宏平均 F1 & 相对全文 \\\\",
        r"\midrule",
    ]
    for point in data["curve"]:
        name = "全文" if point["tokens"] is None else str(point["tokens"])
        loss = ("--" if point["tokens"] is None
                else f"{round(point['macro_f1'], 4) - round(full, 4):+.4f}")
        lines.append(
            f"{name} & {point['accuracy']:.4f} & {point['macro_f1']:.4f} & {loss} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "truncation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _class_length_stats():
    """Per-class document counts and lengths, in the pipeline's token unit."""
    import pickle
    import statistics

    import pandas as pd
    from src.config import CACHE_DIR
    from src.data import load_splits

    train, val, test = load_splits()
    frame = pd.concat([train, val, test]).reset_index(drop=True)
    lengths = []
    for name in ("train", "val", "test"):
        lengths += pickle.loads((CACHE_DIR / f"tokens_nyt_{name}.pkl").read_bytes())
    table = pd.DataFrame({"label": frame["label"], "n": [len(t) for t in lengths]})

    rows = []
    for label in ("sports", "politics", "business"):
        part = table[table["label"] == label]["n"]
        rows.append({"label": label, "docs": len(part),
                     "share": len(part) / len(table),
                     "median": statistics.median(part), "mean": part.mean()})
    all_n = table["n"]
    rows.append({"label": "全部", "docs": len(table), "share": 1.0,
                 "median": statistics.median(all_n), "mean": all_n.mean()})
    return rows


def _write_dataset_table(tables):
    """The table a data section should carry: the classes themselves.

    Counts, shares and lengths are comparable attributes on three comparable
    items, which is what makes a table the right device here.
    """
    rows = _class_length_stats()
    name_cn = {"sports": "体育", "politics": "政治", "business": "财经", "全部": "全部"}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{NYT 语料按类别统计。长度一律按 nltk 词元计。}",
        r"\label{tab:dataset}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "类别 & 篇数 & 占比 & 中位长度 & 平均长度 \\\\",
        r"\midrule",
    ]
    for row in rows[:-1]:
        lines.append(f"{name_cn[row['label']]} & {row['docs']:,} & {row['share']*100:.1f}\\% & "
                     f"{row['median']:.0f} & {row['mean']:.0f}\\\\")
    last = rows[-1]
    lines += [
        r"\midrule",
        f"全部 & {last['docs']:,} & 100.0\\% & {last['median']:.0f} & {last['mean']:.0f} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (tables / "dataset.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_discriminative_table(tables):
    """The overlap is the point, so it gets three columns of its own.

    An earlier version put the two overlap counts in one cell and padded the
    row with a ten-word list; the list dominated the width while the actual
    claim - that the classes share almost no discriminative vocabulary - was
    buried.  Here the matrix is the table and the words are one short example
    per row.
    """
    path = RESULTS_DIR / "topic_diagnostics.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    name_cn = {"business": "财经", "politics": "政治", "sports": "体育"}
    order = ("business", "politics", "sports")
    pairs = data["top_term_overlap"]

    def overlap(a, b):
        if a == b:
            return "--"
        return str(len(pairs["|".join(sorted((a, b)))]))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{每个类别权重最高的 50 个词中任意两类的交集大小。这里的词表未剔除"
        r"任何通用词，唯一的非零交集 \emph{mr.} 正是这一类词；表 \ref{tab:keyword-freq} "
        r"用的则是剔除后的词表。对角线是类别与自身，最右列各给两个例词。}",
        r"\label{tab:discriminative}",
        r"\begin{tabular}{lccc@{\hspace{8pt}}l}",
        r"\toprule",
        "类别 & 财经 & 政治 & 体育 & 例词 \\\\",
        r"\midrule",
    ]
    for key in order:
        terms = data["top_terms_per_class"].get(key, [])
        # skip the same short stop-list the frequency table uses
        drop = {"mr.", "ms.", "its", "or", "new", "large", "state", "house", "care"}
        shown = [t for t in terms if t not in drop][:2]
        lines.append(f"{name_cn[key]} & " + " & ".join(overlap(key, o) for o in order)
                     + " & " + "、".join(shown) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tables / "discriminative.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_keyword_freq_table(tables):
    """Discriminative words against the thresholds they have to clear.

    The class is written once per group instead of on all twelve rows, and a
    final row carries the two thresholds - without them the table cannot show
    the thing the prose claims, that every word sits far above the threshold.
    """
    path = RESULTS_DIR / "topic_diagnostics.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    selection = data.get("keyword_selection", {}).get("chosen") or {}
    freqs = data["keyword_frequencies"]
    doc_freqs = data.get("keyword_document_frequencies") or {}
    if not selection:
        return
    name_cn = {"business": "财经", "politics": "政治", "sports": "体育"}
    groups = [k for k in ("sports", "politics", "business") if selection.get(k)]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{每个类别中跳过少量通用词后权重最高的四个词，及其在 9{,}215 篇"
        r"训练文档中的\emph{出现次数}与\emph{出现文档数}，末行给出各自的门槛"
        r"（前者是 Task 2 的 \code{min\_count}，后者是附录的 \code{min\_df}）。"
        r"词序是模型的权重序，不是频次序。}",
        r"\label{tab:keyword-freq}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        "类别 & 词 & 次数 & 篇数 \\\\",
        r"\midrule",
    ]
    for gi, key in enumerate(groups):
        for wi, word in enumerate(selection.get(key, [])):
            label = name_cn[key] if wi == 0 else ""
            lines.append(f"{label} & \\emph{{{word}}} & {freqs.get(word, 0):,} & "
                         f"{doc_freqs.get(word, 0):,} \\\\")
        if gi < len(groups) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\midrule",
        "门槛 & -- & 5 & 5 \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (tables / "keyword_freq.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_year_table(tables):
    """Year mentions, ascending.  A figure would show the shape better; kept as
    a table until the figure pass so nothing is lost in the meantime."""
    path = RESULTS_DIR / "topic_diagnostics.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    years = data["years"]
    top = sorted(years["top_years"][:6])            # ascending, not by count
    total = years["total_mentions"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{训练集中提及的年份（取提及最多的六年后按年份升序排列），"
        r"以及它们在全部 \resultYearTotal{} 次年份提及中的占比。}",
        r"\label{tab:years}",
        r"\begin{tabular}{l" + "r" * len(top) + "}",
        r"\toprule",
        "年份 & " + " & ".join(year for year, _ in top) + r" \\",
        r"\midrule",
        "提及次数 & " + " & ".join(f"{count:,}" for _, count in top) + r" \\",
        r"占比（\%） & " + " & ".join(f"{count / total * 100:.1f}" for _, count in top) + r" \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (tables / "years.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _diagnostics_macros():
    """Numbers from topic_diagnostics.json that the prose quotes."""
    path = RESULTS_DIR / "topic_diagnostics.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for pair, macro_name in (("business|politics", "resultOverlapBizPol"),
                             ("business|sports", "resultOverlapBizSpo"),
                             ("politics|sports", "resultOverlapPolSpo")):
        if pair in data["top_term_overlap"]:
            out[macro_name] = str(len(data["top_term_overlap"][pair]))
    counts = data["keyword_frequencies"]
    if counts:
        lowest = min(counts.values())
        out["resultMinKeywordFreq"] = f"{lowest:,}".replace(",", "{,}")
        out["resultMinKeywordMultiple"] = str(lowest // 5)
    doc_freq = data.get("keyword_document_frequencies") or {}
    if doc_freq:
        out["resultMinKeywordDocFreq"] = f"{min(doc_freq.values()):,}".replace(",", "{,}")
    years = data["years"]
    if years.get("top_years"):
        out["resultYearPeak"] = years["top_years"][0][0]
    if years.get("share_2010_2014") is not None:
        out["resultYearShare"] = f"{years['share_2010_2014'] * 100:.1f}"
    if years.get("total_mentions"):
        out["resultYearTotal"] = f"{years['total_mentions']:,}".replace(",", "{,}")
    return out


def _write_error_table(tables):
    """Where the residual errors sit, counted in documents.

    Figure 2 is row-normalised, which hides class size: sports is 75% of the
    test set, so 1% of sports is more documents than 4% of business.  The
    prose quotes document counts, so they need to be readable somewhere.
    """
    labels = {"bert": "BERT（64 子词）", "bow_binary": "二值词袋",
              "embed_w2v_ag": "W2V（AG）"}
    rows = []
    for name, label in labels.items():
        record = load(name)
        if not record:
            continue
        matrix = record["confusion_matrix"]
        rows.append((label,
                     matrix[0][1] + matrix[0][2],
                     matrix[1][0] + matrix[1][2],
                     matrix[2][0] + matrix[2][1]))
    if not rows:
        return

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{测试集上误差的归属（单位：篇），\emph{按真实类别分组}：三列分别是真实为"
        r"财经、政治、体育的文档中被误判到其他类的篇数，一格一个数。"
        r"图 \ref{fig:confusions} 按行归一化"
        r"展示，比例会被类别规模掩盖，这里换算成篇数。}",
        r"\label{tab:errors}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        "配置 & 财经 & 政治 & 体育 \\\\",
        r"\midrule",
    ]
    for label, biz, pol, sports in rows:
        lines.append(f"{tex(label)} & {biz} & {pol} & {sports} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tables / "errors.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_repetition_table(tables):
    """The seed repeats, in a table rather than only in the prose.

    Every stability claim in the discussion rests on these ranges, and the
    main results table shows only the seed-42 point, so they need a home of
    their own.
    """
    groups = [
        ("W2V（NYT）", ["embed_w2v_nyt", "embed_w2v_nyt_seed7", "embed_w2v_nyt_seed123"]),
        ("W2V（AG News）", ["embed_w2v_ag", "embed_w2v_ag_seed7", "embed_w2v_ag_seed123"]),
        ("BERT，64 词", ["bert", "bert_seed7", "bert_seed123"]),
        ("BERT，512 词", ["bert_len512", "bert_len512_seed7", "bert_len512_seed123"]),
    ]
    rows = []
    for label, names in groups:
        scores = [r["macro_f1"] for r in (load(n) for n in names) if r]
        if scores:
            rows.append((label, scores))
    if not rows:
        return

    width = max(len(scores) for _, scores in rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{带随机性配置的重复运行，每格是一个种子的宏平均 F1。"
        r"词袋与 GloVe 没有随机性，表中未列，其数值是定值。}",
        r"\label{tab:repeats}",
        r"\begin{tabular}{l" + "r" * width + "}",
        r"\toprule",
        "配置 & " + " & ".join([f"种子{i}" for i in range(1, width + 1)])
        + " " + r"\\",
        r"\midrule",
    ]
    for label, scores in rows:
        padded = list(scores) + [""] * (width - len(scores))
        lines.append(f"{tex(label)} & " + " & ".join(f"{s:.4f}" for s in padded)
                     + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tables / "repeats.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_prefix_compare_table(tables):
    """BERT and bag-of-words at the same prefix length.

    This is the measurement that separates "the encoder is stronger" from
    "the encoder saw less": both models read the same first N tokens, so the
    difference at fixed N is attributable to the representation.
    """
    path = RESULTS_DIR / "truncation_curve.json"
    if not path.exists():
        return
    bow = {p["tokens"]: p["macro_f1"]
           for p in json.loads(path.read_text(encoding="utf-8"))["curve"]}
    lengths = [n for n in (64, 128, 256, 512) if n in bow]

    rows = []
    for length in lengths:
        name = "bert.json" if length == 64 else f"bert_len{length}.json"
        bert_path = RESULTS_DIR / name
        if not bert_path.exists():
            continue
        bert = json.loads(bert_path.read_text(encoding="utf-8"))["macro_f1"]
        rows.append((length, bert, bow[length]))
    if not rows:
        return

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{相同前缀长度下 BERT 与词袋的对比。两者读的是同一批前 $N$ 个词，"
        r"因此差值只能归因于表示本身。}",
        r"\label{tab:prefix}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        "前缀词数 & BERT & 词袋 & 差值 \\\\",
        r"\midrule",
    ]
    for length, bert, bow_score in rows:
        delta = round(bert, 4) - round(bow_score, 4)
        lines.append(f"{length} & {bert:.4f} & {bow_score:.4f} & {delta:+.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tables / "prefix_compare.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prefix_macros():
    """Numbers the prose quotes for the matched-length comparison."""
    path = RESULTS_DIR / "truncation_curve.json"
    if not path.exists():
        return {}
    bow = {p["tokens"]: p["macro_f1"]
           for p in json.loads(path.read_text(encoding="utf-8"))["curve"]}
    out = {}
    for length, suffix in ((64, "SixtyFour"), (128, "OneTwentyEight"),
                           (256, "TwoFiftySix"), (512, "FiveTwelve")):
        name = "bert.json" if length == 64 else f"bert_len{length}.json"
        bert_path = RESULTS_DIR / name
        if bert_path.exists() and length in bow:
            bert = json.loads(bert_path.read_text(encoding="utf-8"))["macro_f1"]
            out[f"resultPrefixGap{suffix}"] = f"{round(bert, 4) - round(bow[length], 4):.4f}"
    return out


def _matched_macros():
    """BERT vs bag-of-words at genuinely the same words per document."""
    out = {}
    names = ((64, "Short"), (128, "Mid"), (256, "Long"), (512, "Full"))
    for length, suffix in names:
        bert_path = RESULTS_DIR / ("bert.json" if length == 64 else f"bert_len{length}.json")
        matched_path = RESULTS_DIR / f"matched_{length}.json"
        if not (bert_path.exists() and matched_path.exists()):
            continue
        bert = json.loads(bert_path.read_text(encoding="utf-8"))["macro_f1"]
        matched = json.loads(matched_path.read_text(encoding="utf-8"))["macro_f1"]
        out[f"resultMatchedGap{suffix}"] = f"{round(bert, 4) - round(matched, 4):.4f}"
        out[f"resultMatchedBow{suffix}"] = f"{round(matched, 4):.4f}"
    return out


def _bert_wrong_bow_right():
    """Documents BERT got wrong and bag-of-words got right.

    Computed from the two per-document prediction files rather than typed, so
    the count the prose quotes stays tied to the artefacts it came from.
    """
    import pandas as pd
    bert = RESULTS_DIR / "bert_test_predictions.csv"
    bow = RESULTS_DIR / "bow_binary_test_predictions.csv"
    if not (bert.exists() and bow.exists()):
        return 0
    merged = pd.read_csv(bert).merge(pd.read_csv(bow), on="idx", suffixes=("_b", "_w"))
    return int(((merged["prediction_b"] != merged["label_b"])
                & (merged["prediction_w"] == merged["label_w"])).sum())


def _length_macros():
    """Corpus lengths in the unit the pipeline actually uses.

    An earlier version of the report quoted the median from a whitespace
    split while the length stratification used nltk tokens; the two differ by
    about 14% (672 vs 768) and nothing said which was which.
    """
    import pickle
    import statistics

    from src.config import CACHE_DIR
    tokens = []
    for name in ("train", "val", "test"):
        path = CACHE_DIR / f"tokens_nyt_{name}.pkl"
        if not path.exists():
            return {}
        tokens += pickle.loads(path.read_bytes())
    lengths = sorted(len(t) for t in tokens)
    if not lengths:
        return {}
    def pct(q):
        return lengths[min(int(q * len(lengths)), len(lengths) - 1)]
    median = statistics.median(lengths)
    return {
        "resultMedianTokens": str(int(median)),
        "resultPercentileTokens": str(pct(0.90)),
        "resultMaxTokens": f"{lengths[-1]:,}".replace(",", "{,}"),
        # what share of a median document the 64-token head covers
        "resultHeadShare": f"{64 / median * 100:.1f}",
    }


def _error_macros():
    """Numbers from the error stratifications, for the prose to quote."""
    path = RESULTS_DIR / "error_by_length.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    bins = {row["bin"]: row for row in data["bins"]}
    out = {}
    overall = data["all"]
    out["resultErrAllBow"] = f"{overall['bow_err'] * 100:.1f}"
    out["resultErrAllBert"] = f"{overall['bert_err'] * 100:.1f}"
    out["resultSeenAll"] = f"{overall['seen'] * 100:.1f}"
    for bin_name, suffix in (("<200", "Short"), ("700--1000", "Peak"), (">1000", "Longest")):
        if bin_name in bins:
            row = bins[bin_name]
            out[f"resultByLen{suffix}Bow"] = f"{row['bow_err'] * 100:.1f}"
            out[f"resultByLen{suffix}Bert"] = f"{row['bert_err'] * 100:.1f}"
            out[f"resultSeen{suffix}"] = f"{row['seen'] * 100:.1f}"
            out[f"resultDocs{suffix}"] = f"{row['docs']:,}".replace(",", "{,}")
    return out


def _hedge_macros():
    """Numbers from scripts/hedge_experiments.py, which settled six passages
    that previously ended in a guess."""
    path = RESULTS_DIR / "hedge_experiments.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    bins = d.get("E1_bin_composition", {})
    longest = bins.get(">1000", {})
    if longest:
        n = sum(longest.values())
        out["resultLongestBinSports"] = f"{longest.get('sports', 0) / n * 100:.0f}"
        out["resultLongestBinDocs"] = str(n)
    errs = d.get("E6_bert_sports_errors", {})
    if errs:
        out["resultSportsErrors"] = str(errs.get("n_errors", 0))
        out["resultSportsErrorsBiz"] = str(errs.get("predicted_as", {}).get("business", 0))
        out["resultSportsErrorsPol"] = str(errs.get("predicted_as", {}).get("politics", 0))
    gaps = d.get("E3_train_test_gap", {})
    for key, name in (("glove", "resultGloveTestOverTrain"),
                      ("w2v_nyt_train_only", "resultNytVecTestOverTrain")):
        if key in gaps:
            out[name] = f"{gaps[key]['test_over_train']:.4f}"
    sep = d.get("E4_class_separation", {})
    for key, name in (("glove", "resultSepGlove"), ("w2v_nyt", "resultSepNyt"),
                      ("w2v_ag", "resultSepAg")):
        if key in sep:
            out[name] = f"{sep[key]['between_over_within']:.3f}"
    pos = d.get("E5_word_positions", {})
    if "sports" in pos:
        out["resultSportsDocsWithLateHit"] = f"{pos['sports']['doc_share_with_hits_past_64'] * 100:.1f}"
        out["resultSportsHitsInHead"] = f"{pos['sports']['hit_share_in_first_64'] * 100:.1f}"
        out["resultSportsDocs"] = str(pos["sports"]["docs"])
    curve = d.get("E2_dense_curve", [])
    if curve:
        out["resultDenseBowCurve"] = "、".join(f"{r['bow_macro_f1']:.4f}" for r in curve)
        out["resultDenseBowWords"] = "/".join(f"{r['mean_words']:.0f}" for r in curve)
    return out


def _write_macros(tables, records):
    scores = [records[n]["macro_f1"] for n in MAIN_SIX if records[n]]

    values = {
        "resultMajorityAcc": macro(records["baseline_majority"], "accuracy"),
        "resultMajorityFOne": macro(records["baseline_majority"], "macro_f1"),
        "resultBowAcc": macro(records["bow_binary"], "accuracy"),
        "resultBowFOne": macro(records["bow_binary"], "macro_f1"),
        "resultBowFOneFreq": macro(records["bow_freq"], "macro_f1"),
        "resultBertAcc": macro(records["bert"], "accuracy"),
        "resultBertFOne": macro(records["bert"], "macro_f1"),
        "resultRangeLow": f"{min(scores):.4f}" if scores else "--",
        "resultRangeHigh": f"{max(scores):.4f}" if scores else "--",
    }
    # How far min_df=5 shrinks the vocabulary, as a ratio rather than a phrase.
    small = records.get("bow_binary_mindf5")
    if small and records["bow_binary"]:
        values["resultVocabShrink"] = f"{small['meta']['vocab_size'] / records['bow_binary']['meta']['vocab_size']:.3f}"

    vocab = records["bow_binary"]["meta"]["vocab_size"] if records["bow_binary"] else None
    values["resultVocabSize"] = f"{vocab:,}".replace(",", "{,}") if vocab else "--"

    # The accuracy-minus-macro-F1 gap, which prediction one is about.  Computed
    # rather than typed, so the prose cannot drift from the results.
    gaps = [(n, round(records[n]["accuracy"], 4) - round(records[n]["macro_f1"], 4))
            for n in MAIN_SIX if records[n]]
    if gaps:
        values["resultGapMin"] = f"{min(g for _, g in gaps):.4f}"
        values["resultGapMax"] = f"{max(g for _, g in gaps):.4f}"
        widest = max(gaps, key=lambda item: item[1])[0]
        values["resultGapMaxName"] = dict((n, lab) for n, lab, _ in ORDER).get(widest, widest)

    coverage = {
        name: records[name]["meta"]["token_coverage_train"]
        for name in MAIN_EMBEDDINGS if records[name]
    }
    for name, macro_name in (("embed_glove", "resultGloveCov"),
                             ("embed_w2v_nyt", "resultNytCov"),
                             ("embed_w2v_ag", "resultAgCov")):
        values[macro_name] = f"{coverage[name]:.4f}" if name in coverage else "--"

    embed_scores = [records[n]["macro_f1"] for n in MAIN_EMBEDDINGS if records[n]]
    values["resultEmbedSpread"] = f"{max(embed_scores) - min(embed_scores):.4f}" if embed_scores else "--"

    # The max_length control.  Reported as the gain over the required
    # 64-token run, which is the quantity the argument actually turns on.
    # Defined with a placeholder up front so the report still compiles while
    # the control run is in flight.
    for name in ("resultBertLenAcc", "resultBertLenFOne", "resultBertTruncAccDelta",
                 "resultBertTruncFOneDelta", "resultBestNonBert", "resultBertGapToBest"):
        values.setdefault(name, "--")

    long_run = records.get("bert_len512")
    if long_run and records["bert"]:
        # Deltas are taken at the precision the report prints (4 dp), not on
        # the raw floats, so that a reader subtracting the two displayed
        # numbers gets exactly the delta shown next to them.
        short, long = records["bert"], long_run
        values["resultBertLenAcc"] = f"{long['accuracy']:.4f}"
        values["resultBertLenFOne"] = f"{long['macro_f1']:.4f}"
        values["resultBertTruncAccDelta"] = f"{round(long['accuracy'], 4) - round(short['accuracy'], 4):+.4f}"
        values["resultBertTruncFOneDelta"] = f"{round(long['macro_f1'], 4) - round(short['macro_f1'], 4):+.4f}"
        best_other = max(
            round(records[n]["macro_f1"], 4) for n in MAIN_SIX if records[n] and n != "bert"
        )
        values["resultBestNonBert"] = f"{best_other:.4f}"
        values["resultBertGapToBest"] = f"{round(short['macro_f1'], 4) - best_other:+.4f}"
        # Unsigned companions: the prose says "低 X" / "提高了 X", and a signed
        # macro next to those words reads as a double negative.
        values["resultBertGapToBestAbs"] = f"{abs(round(short['macro_f1'], 4) - best_other):.4f}"
        values["resultBertTruncFOneDeltaUns"] = f"{abs(round(long['macro_f1'], 4) - round(short['macro_f1'], 4)):.4f}"
        values["resultBertTruncAccDeltaUns"] = f"{abs(round(long['accuracy'], 4) - round(short['accuracy'], 4)):.4f}"

    values.update(_repetition_macros(records))
    values.update(_truncation_macros())
    values.update(_diagnostics_macros())
    values.update(_prefix_macros())
    values.update(_matched_macros())
    values.update(_error_macros())
    values.update(_length_macros())
    values.update(_hedge_macros())

    bad = [name for name in values if any(c.isdigit() for c in name)]
    if bad:
        raise ValueError(f"macro names may not contain digits (TeX truncates "
                         f"the name there): {bad}")

    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()]
    (tables / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _repetition_macros(records):
    """Summarise the repeated runs, so the prose can state ranges rather than
    single numbers.  GloVe has no randomness, which the repeats confirm."""
    out = {}
    for stem, prefix in (("embed_w2v_nyt", "resultNytVec"),
                         ("embed_w2v_ag", "resultAgVec")):
        runs = [load(stem), load(f"{stem}_seed7"), load(f"{stem}_seed123")]
        scores = sorted(r["macro_f1"] for r in runs if r)
        if not scores:
            continue
        out[f"{prefix}Worst"] = f"{scores[0]:.4f}"
        out[f"{prefix}Best"] = f"{scores[-1]:.4f}"
        out[f"{prefix}Spread"] = f"{scores[-1] - scores[0]:.4f}"

    glove_runs = [load("embed_glove"), load("embed_glove_seed7")]
    if glove_runs and all(glove_runs):
        out["resultGloveScore"] = f"{glove_runs[0]['macro_f1']:.4f}"
        # A checkable statement, not a claim: the two runs must be identical.
        same = abs(glove_runs[0]["macro_f1"] - glove_runs[1]["macro_f1"]) < 1e-12
        out["resultGloveDeterministic"] = "完全一致" if same else "不一致"
        out["resultGloveRunA"] = f"{glove_runs[0]['macro_f1']:.4f}"
        out["resultGloveRunB"] = f"{glove_runs[1]['macro_f1']:.4f}"

    # BERT is stochastic (weight init, dropout, batch order), so its runs are
    # summarised as a range too.  The 64-token and 512-token groups are kept
    # apart, since the whole argument turns on the gap between them.
    bert_groups = {
        "resultBertSeeds": ["bert", "bert_seed7", "bert_seed123"],
        "resultBertLenSeeds": ["bert_len512", "bert_len512_seed7", "bert_len512_seed123"],
    }
    for prefix, names in bert_groups.items():
        scores = sorted(r["macro_f1"] for r in (load(n) for n in names) if r)
        if not scores:
            continue
        out[f"{prefix}Worst"] = f"{scores[0]:.4f}"
        out[f"{prefix}Best"] = f"{scores[-1]:.4f}"
        out[f"{prefix}Spread"] = f"{scores[-1] - scores[0]:.4f}"
        out[f"{prefix}N"] = str(len(scores))

    # The truncation gain as a cross-seed worst case: the lowest 512-token run
    # against the highest 64-token run.  If this is positive the two ranges do
    # not overlap and the gain does not depend on which seed you got.
    short_runs = [r["macro_f1"] for r in
                  (load(n) for n in ("bert", "bert_seed7", "bert_seed123")) if r]
    long_runs = [r["macro_f1"] for r in
                 (load(n) for n in ("bert_len512", "bert_len512_seed7",
                                    "bert_len512_seed123")) if r]
    if short_runs and long_runs:
        out["resultTruncMinGain"] = f"{min(long_runs) - max(short_runs):.4f}"
        out["resultTruncRangesOverlap"] = "重叠" if min(long_runs) <= max(short_runs) else "不重叠"

    # Gaps the prose quotes directly.
    bow_bin, bow_freq = load("bow_binary"), load("bow_freq")
    if bow_bin and bow_freq:
        out["resultBowPairDiff"] = f"{round(bow_bin['macro_f1'], 4) - round(bow_freq['macro_f1'], 4):.4f}"
        # The validation split ranks the two the other way round, and that is
        # what picked their (differing) C, so the test gap must be read with it.
        out["resultBowValBin"] = f"{bow_bin['meta']['val_macro_f1']:.4f}"
        out["resultBowValFreq"] = f"{bow_freq['meta']['val_macro_f1']:.4f}"

    # Where the residual errors actually sit.  Row-normalised matrices hide
    # class size, and sports is 75% of the test set, so a 1% sports error is
    # larger in documents than a 4% business error.
    def error_split(name):
        """Errors grouped by the TRUE class, one convention for every column.

        An earlier version mixed conventions - one column counted both
        directions of the business/politics confusion, the other only counted
        sports as the true class - which made the two columns incomparable.
        """
        record = records.get(name)
        if not record:
            return None
        matrix = record["confusion_matrix"]
        # row i = true class i; sum its off-diagonal cells
        return (matrix[0][1] + matrix[0][2],
                matrix[1][0] + matrix[1][2],
                matrix[2][0] + matrix[2][1])

    for name, prefix in (("bert", "resultErrBert"), ("bow_binary", "resultErrBow"),
                         ("embed_w2v_ag", "resultErrAgVec")):
        split = error_split(name)
        if split:
            out[f"{prefix}Biz"] = str(split[0])
            out[f"{prefix}Pol"] = str(split[1])
            out[f"{prefix}Sports"] = str(split[2])
            out[f"{prefix}BizPol"] = str(split[0] + split[1])

    # Corpus size, so "orders of magnitude" becomes a checkable ratio.  The
    # token cache is the same one every experiment tokenised.
    import pickle
    from src.config import CACHE_DIR
    token_path = CACHE_DIR / "tokens_nyt_train.pkl"
    if token_path.exists():
        tokens = pickle.loads(token_path.read_bytes())
        total = sum(len(t) for t in tokens)
        out["resultBertWrongBowRight"] = str(_bert_wrong_bow_right())
        # how many of those sit in the two longest length bins
        _b = RESULTS_DIR / "bert_test_predictions.csv"
        _w = RESULTS_DIR / "bow_binary_test_predictions.csv"
        if _b.exists() and _w.exists():
            import pickle
            import pandas as pd
            from src.config import CACHE_DIR
            from src.data import load_split
            _te = load_split("test")
            _len = {int(i): len(t) for i, t in
                    zip(_te["idx"], pickle.loads((CACHE_DIR / "tokens_nyt_test.pkl").read_bytes()))}
            _m = pd.read_csv(_b).merge(pd.read_csv(_w), on="idx", suffixes=("_b", "_w"))
            _m["L"] = _m["idx"].map(_len)
            _c = _m[(_m.prediction_b != _m.label_b) & (_m.prediction_w == _m.label_w)]
            out["resultBertWrongBowRightLong"] = str(int((_c["L"] > 700).sum()))
        out["resultTrainTokens"] = f"{total:,}".replace(",", "{,}")
        out["resultGloveRatio"] = f"{6_000_000_000 // total:,}".replace(",", "{,}")

    if bow_bin:
        per_class = list(bow_bin["per_class_f1"].values())
        gap = max(per_class) - min(per_class)
        out["resultClassGap"] = f"{gap:.4f}"
        # The conclusion says the method spread is smaller than the class gap;
        # give the ratio so the comparison can be checked rather than asserted.
        spread = (max(records[n]["macro_f1"] for n in MAIN_SIX if records[n])
                  - min(records[n]["macro_f1"] for n in MAIN_SIX if records[n]))
        if spread:
            out["resultClassGapRatio"] = f"{gap / spread:.1f}"

    # Test-set class counts: the prose quotes error counts in documents, and the
    # reader needs these to check them.
    meta_path = RESULTS_DIR.parent / "splits" / "meta.json"
    if meta_path.exists():
        counts = json.loads(meta_path.read_text(encoding="utf-8"))["splits"]["test"]["label_counts"]
        for key, name in (("business", "Biz"), ("politics", "Pol"), ("sports", "Spo")):
            if key in counts:
                out[f"resultTest{name}"] = str(counts[key])

    sports = [records[n]["per_class_f1"]["sports"] for n in MAIN_SIX if records[n]]
    if sports:
        out["resultSportsFOneMin"] = f"{min(sports):.4f}"

    nyt_scores = sorted(r["macro_f1"] for r in
                        (load(n) for n in ("embed_w2v_nyt", "embed_w2v_nyt_seed7",
                                           "embed_w2v_nyt_seed123")) if r)
    glove = load("embed_glove")
    if nyt_scores and glove:
        out["resultNytOverGlove"] = f"{nyt_scores[0] - round(glove['macro_f1'], 4):.4f}"

    nyt_main, bow = load("embed_w2v_nyt"), load("bow_binary")
    if nyt_main and bow:
        out["resultNytVec"] = f"{round(nyt_main['macro_f1'], 4):.4f}"
        out["resultNytVecAcc"] = f"{round(nyt_main['accuracy'], 4):.4f}"
        bert = records.get("bert")
        if bert:
            out["resultNytGapToBert"] = f"{round(nyt_main['macro_f1'], 4) - round(bert['macro_f1'], 4):.4f}"
            out["resultBowGapToBert"] = f"{round(bow['macro_f1'], 4) - round(bert['macro_f1'], 4):.4f}"
        out["resultNytOverBow"] = f"{abs(round(nyt_scores[-1], 4) - round(bow['macro_f1'], 4)):.4f}"
        # BoW is deterministic, so the like-for-like comparison is its single
        # value against the Word2Vec run under the same seed, not against the
        # best of three Word2Vec seeds.
        glove_rec = load("embed_glove")
        if glove_rec:
            # BERT-vs-GloVe is a different gap from NYT-Word2Vec-vs-GloVe; the
            # prose used the latter's value for the former.
            out["resultBertOverGlove"] = f"{abs(round(glove_rec['macro_f1'], 4) - round(records['bert']['macro_f1'], 4)):.4f}"
        out["resultNytOverBowSameSeed"] = f"{abs(round(nyt_main['macro_f1'], 4) - round(bow['macro_f1'], 4)):.4f}"

    pairs = [(load("bow_binary"), load("bow_binary_mindf5")),
             (load("bow_freq"), load("bow_freq_mindf5"))]
    deltas = [abs(round(a["macro_f1"], 4) - round(b["macro_f1"], 4))
              for a, b in pairs if a and b]
    if deltas:
        out["resultMinDfMaxDelta"] = f"{max(deltas):.4f}"
    return out


def _truncation_macros():
    """Numbers from the prefix-length sweep used to profile the truncation."""
    path = RESULTS_DIR / "truncation_curve.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    by_length = {point["tokens"]: point for point in data["curve"]}
    out = {"resultTruncFull": f"{data['reference_macro_f1']:.4f}"}
    # Spell the lengths out: a macro name containing a digit is silently cut
    # short by TeX at the digit, so \resultTruncAt64 would expand as
    # \resultTruncAt followed by a stray "64".
    for length, name in ((64, "resultTruncShort"), (512, "resultTruncLong")):
        if length in by_length:
            out[name] = f"{by_length[length]['macro_f1']:.4f}"
    if 64 in by_length:
        out["resultTruncShortLoss"] = f"{data['reference_macro_f1'] - by_length[64]['macro_f1']:.4f}"
    if 64 in by_length and 512 in by_length:
        out["resultTruncOvershoot"] = f"{round(by_length[512]['macro_f1'], 4) - round(data['reference_macro_f1'], 4):.4f}"
        out["resultTruncGain"] = f"{by_length[512]['macro_f1'] - by_length[64]['macro_f1']:.4f}"
    return out


if __name__ == "__main__":
    main()
