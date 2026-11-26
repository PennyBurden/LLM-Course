"""Render the review operating points and the learning curve from archived runs."""
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

from review_experiment import load_scores, summarize


def plot_font():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in ("Noto Sans CJK SC", "Hiragino Sans GB", "Microsoft YaHei",
                                   "PingFang SC", "Arial Unicode MS") if name in fonts), "DejaVu Sans")


def plot_review(folder, family):
    archive = load_scores(folder / "review_scores.json.gz")
    thresholds = archive["thresholds"]
    chinese = family != "DejaVu Sans"
    titles = ("自动判定覆盖率", "自动判定中的准确率") if chinese else ("Automatic coverage", "Accuracy on automatic decisions")
    labels = ["全部" if chinese else "All"] + [f"{t:.2f}" for t in thresholds[1:]]
    with plt.rc_context({"font.family": family, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#263741"}):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.65), sharey=True, layout="constrained")
        for split, symbol, color, style in (("test_a", "o", "#075D83", "-"), ("test_b", "s", "#994A22", "--")):
            data = archive["splits"][split]
            points = [summarize(data, cutoff) for cutoff in thresholds]
            name = split[-1].upper()
            label = f"测试 {name}（{len(data):,} 篇）" if chinese else f"Test {name} (n={len(data):,})"
            series = ([p["auto"] / p["n"] for p in points],
                      [p["auto_correct"] / p["auto"] if p["auto"] else float("nan") for p in points])
            for ax, ys in zip(axes, series, strict=True):
                ax.plot(range(len(thresholds)), ys, color=color, marker=symbol, linestyle=style,
                        label=label, linewidth=2.2, markersize=5.5)
        for ax, title in zip(axes, titles, strict=True):
            ax.set(title=title, xticks=range(len(labels)), xticklabels=labels, ylim=(0, 1.04),
                   xlabel="最低分类分数" if chinese else "Minimum class score")
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#DCE2E5", linewidth=.8)
        axes[1].legend(loc="lower left", frameon=False, fontsize=10)
        fig.savefig(folder / "review_coverage.png", dpi=180, facecolor="white")
        plt.close(fig)


def plot_learning(folder, family):
    curve = json.loads((folder / "learning_curve.json").read_text(encoding="utf-8"))
    trials = curve["trials"]
    sizes = sorted({r["size"] for r in trials})
    chinese = family != "DejaVu Sans"
    with plt.rc_context({"font.family": family, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(6.8, 3.6), layout="constrained")
        for key, zh, en, color, marker, style in (
            ("detect_macro_f1", "三类检测", "Writing class", "#075D83", "o", "-"),
            ("source_macro_f1", "八家族溯源（含漏检）", "Source family (including misses)", "#994A22", "s", "--"),
        ):
            groups = [[r["metrics"][key] for r in trials if r["size"] == size] for size in sizes]
            means = [mean(group) for group in groups]
            err = [[m - min(g) for m, g in zip(means, groups, strict=True)],
                   [max(g) - m for m, g in zip(means, groups, strict=True)]]
            ax.errorbar(sizes, means, yerr=err, label=zh if chinese else en, color=color, marker=marker,
                        linestyle=style, capsize=4, linewidth=2, elinewidth=1.4)
        ax.set_xscale("log", base=2)
        ax.set(xticks=sizes, xticklabels=[f"{n:,}" for n in sizes], ylim=(0, 1), ylabel="Macro-F1",
               xlabel="训练文本数（每档翻倍）" if chinese else "Training texts (doubling at each step)",
               title=(f"训练量与验证集表现（{curve['validation_n']:,} 篇）" if chinese
                      else f"Training size and validation performance (n={curve['validation_n']:,})"))
        ax.text(0.02, 0.94, "点为三个种子的均值，误差线为最小值至最大值" if chinese else "Three seeds: mean and min–max",
                transform=ax.transAxes, fontsize=9, color="#33454F")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DCE2E5", linewidth=.8)
        ax.legend(loc="lower right", frameon=False)
        fig.savefig(folder / "learning_curve.png", dpi=180, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    family = plot_font()
    plot_review(root, family)
    plot_learning(root, family)
