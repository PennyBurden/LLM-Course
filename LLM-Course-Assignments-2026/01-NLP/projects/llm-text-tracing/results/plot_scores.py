"""Rebuild the report figures from the recorded scores and family confusion counts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
validation = json.loads((HERE / "validation.json").read_text(encoding="utf-8"))
test = json.loads((HERE / "test_b.json").read_text(encoding="utf-8"))
installed = {font.name for font in font_manager.fontManager.ttflist}
font = next((name for name in ("Noto Sans CJK SC", "Hiragino Sans GB", "Microsoft YaHei",
                               "PingFang SC", "Arial Unicode MS") if name in installed), None)
zh = font is not None
labels = {
    "char-tfidf": "字符基线" if zh else "Character baseline",
    "qwen-head": "Qwen 开头" if zh else "Qwen head",
    "qwen-head-tail": "Qwen 首尾" if zh else "Qwen head + tail",
    "qwen-detect-char-source": "Qwen 检测 + 字符溯源" if zh else "Qwen detect + character source",
}
metrics = [
    ("detect_macro_f1", "检测 Macro-F1" if zh else "Detection macro-F1", "o", "#12618A"),
    ("source_macro_f1", "溯源 Macro-F1" if zh else "Source macro-F1", "^", "#9C4E16"),
    ("joint_score", "联合得分" if zh else "Joint score", "s", "#203A36"),
]
panels = [
    (validation["results"], validation["configuration"]["dev_n"], "固定验证子集" if zh else "Validation subset"),
    (test["results"], test["sample_n"], "独立测试 B 子集" if zh else "Held-out test B subset"),
]
with plt.rc_context({"font.family": font or "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False}):
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5), gridspec_kw={"height_ratios": (4, 2)},
                             layout="constrained")
    for ax, (scores, n, title) in zip(axes, panels, strict=True):
        methods = list(scores)
        for key, metric, marker, color in metrics:
            ax.scatter([scores[method][key] for method in methods], range(len(methods)),
                       label=metric, marker=marker, color=color, s=57, zorder=3)
        ax.set_yticks(range(len(methods)), [labels[m] for m in methods])
        ax.set_ylim(len(methods) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="x", color="#D9DFDF", linewidth=0.8, zorder=0)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.set_title(f"{title}   (n = {n})", loc="left", fontsize=12, pad=7)
    axes[-1].set_xlabel("得分" if zh else "Score")
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False,
                    columnspacing=1.4, handletextpad=0.4)
    fig.savefig(HERE / "scores.png", dpi=180, facecolor="white",
                metadata={"Description": "Point scores from validation.json and test_b.json."})
    plt.close(fig)

    attribution = json.loads((HERE / "data_and_families.json").read_text(encoding="utf-8"))["attribution"]
    matrix = attribution["joint_confusion"]
    families = [row["family"] for row in attribution["by_family"]]
    n = sum(map(sum, matrix))
    fig, ax = plt.subplots(figsize=(7.5, 5.3), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues", vmin=0, interpolation="nearest")
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            color = "white" if image.norm(count) > 0.58 else "#172B3A"
            ax.text(j, i, str(count), ha="center", va="center", color=color, fontsize=9)
    ax.set_yticks(range(len(families)), families)
    ax.set_xticks(range(len(families) + 1), [*families, "检测漏检" if zh else "Missed"],
                  rotation=35, ha="right")
    ax.axvline(len(families) - 0.5, color="#445765", linewidth=1.2)
    ax.set_xlabel("系统输出" if zh else "System output")
    ax.set_ylabel("真实家族" if zh else "True family")
    title = "字符基线对纯机器文本的溯源结果" if zh else "Character baseline on machine-generated texts"
    ax.set_title(f"{title}   (n = {n:,})", loc="left", pad=9)
    fig.colorbar(image, ax=ax, shrink=0.76, label="篇数" if zh else "Texts")
    fig.savefig(HERE / "family_confusion.png", dpi=180, facecolor="white",
                metadata={"Description": "True family by final prediction, including missed machine texts."})
    plt.close(fig)
