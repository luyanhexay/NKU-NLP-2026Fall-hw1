# 自然语言处理 作业一：文本分类

在 NYT 三分类任务上比较六种文本表示：二值词袋、词频词袋、预训练 GloVe 6B、
在 AG News 上训练的 Word2Vec、在 NYT 上训练的 Word2Vec，以及微调的 BERT。
除 BERT 外统一搭配逻辑回归。

## 文件在哪

| 想看什么 | 去哪里 |
| --- | --- |
| 实验报告（成品 PDF，英文） | `report/main.pdf` |
| 报告源码（LaTeX） | `report/main.tex` |
| 代码 | `src/` 公共库，`scripts/` 每个实验一个脚本 |
| 全部实验结果 | `results/`，每个实验一份 JSON |
| 数据划分 | `splits/meta.json`，含种子与三个集合的精确索引 |

报告里的**每一个数字都由 `scripts/collect_results.py` 从 `results/` 生成**，
没有手抄；图表同理。

## 要求的任务

| 任务 | 配置 | 准确率 | 宏平均 F1 |
| --- | --- | --- | --- |
| 参照 | 永远预测体育 | 0.7493 | 0.2856 |
| Task 1 | 二值词袋 | 0.9887 | 0.9744 |
| Task 1 | 词频词袋 | 0.9887 | 0.9728 |
| Task 2 | GloVe 6B（100 维） | 0.9861 | 0.9666 |
| Task 2 | Word2Vec（AG News，100 维） | 0.9844 | 0.9604 |
| Task 2 | Word2Vec（NYT，100 维） | 0.9905 | 0.9745 |
| Task 3 | BERT（max_length=64，3 epochs） | 0.9818 | 0.9623 |

六组共用同一份 80/10/10 划分，均在 NYT 测试集（1,153 篇）上评价。

**最值得看的一条结论**：作业规定的 `max_length=64` 使 BERT 只读到每篇文档的
头部（词元数中位数 768），这是它排名偏低的成因。把截断放宽到 512 子词、其余
不变，宏平均 F1 由 0.9623 升到 0.9823，且三个种子下都高于全部经典配置。

## 额外做的

1. **先写预测、再做实验、最后专门剖析被推翻的两条**——报告第 3、5、6 节。
2. **前缀长度扫描**：把表示固定为无随机性的词袋，只改可见词数，单独量出截断
   的代价（`truncation_curve.py`）。
3. **单位对齐的匹配对比**：BERT 的 `max_length` 数的是子词，词袋那条曲线数的
   是词，实测 64 子词只覆盖 54 个词。按每篇文档各自被覆盖的词数截断词袋，让
   两个模型读同一段文本（`prefix_units.py`、`matched_prefix.py`）。
4. **多种子重复**：带随机性的配置各跑三个种子，以区间而非单点给出，并据此收回
   了"BERT 垫底"这类经不起门槛的说法。
5. **消融与诊断**：`min_df` 剪枝消融；判别词集的重叠、词频、语料的年份分布
   （`topic_diagnostics.py`）。
6. **错误分析**：按文档长度分层的错误率、一篇具体误判样例、以及"BERT 错而词袋
   对"的差分对照（`error_analysis.py`）。

## 怎么跑

需要 Python 3.12 与 [uv](https://docs.astral.sh/uv/)。在项目根目录：

```bash
uv sync                        # PyPI 慢时可用 UV_DEFAULT_INDEX=<镜像> uv sync
uv run python -m scripts.make_splits        # 生成划分，只需一次

uv run python -m scripts.task1_bow
uv run python -m scripts.task2_embed --variant glove
uv run python -m scripts.task2_embed --variant w2v_ag
uv run python -m scripts.task2_embed --variant w2v_nyt
uv run python -m scripts.task3_bert

uv run python -m scripts.truncation_curve   # 以下为上述额外分析的脚本
uv run python -m scripts.prefix_units
uv run python -m scripts.matched_prefix
uv run python -m scripts.error_analysis
uv run python -m scripts.topic_diagnostics

uv run python -m scripts.collect_results    # 从 results/ 生成报告的表与宏
uv run python -m scripts.make_figs
./scripts/build_report.sh                   # 重建 report/main.pdf
```

每个脚本写一份 JSON 到 `results/`。`collect_results.py` 与 `make_figs.py` 只读
这些文件，所以报告可以在不重训的前提下重建。

附录实验（`--min-df 5`、`--max-len 128/256/512`）与重复运行（`--seed 7/123`）
的参数见报告的复现说明一节。

## 数据

`data/nyt.csv`（11,519 篇，含 `text`/`label`）与 `data/ag.csv`（90,000 篇，
含 `text`）由课程提供，已随仓库提交。`glove.6B.zip` 体积 862MB，超出 GitHub
单文件上限，未提交，首次运行 Task 2 的 GloVe 时自动下载。
