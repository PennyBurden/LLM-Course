"""Render retrieval recall on the fixed, answerable QA turns at each candidate depth."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "qa_evaluation.json").read_text(encoding="utf-8"))["methods"]
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in ("Noto Sans CJK SC", "Hiragino Sans GB", "Microsoft YaHei",
                                     "PingFang SC", "Arial Unicode MS") if name in fonts), "DejaVu Sans")
    chinese = family != "DejaVu Sans"
    n = data["scoped_evidence"]["metrics"]["answerable"]
    with plt.rc_context({"font.family": family, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(6.2, 3.7), layout="constrained")
        conditions = [("single_turn_tfidf", "单轮字符检索" if chinese else "Single-turn retrieval", "#975313", "s", "--"),
                      ("scoped_evidence", "带范围校验的检索" if chinese else "Scope-aware retrieval", "#12608A", "o", "-")]
        for key, label, color, marker, style in conditions:
            curve = data[key]["metrics"]["recall_at_k"]
            ax.plot(range(1, 6), [curve[str(k)] for k in range(1, 6)], label=label, color=color,
                    marker=marker, linestyle=style, markersize=6, linewidth=2)
        ax.set(xticks=range(1, 6), ylim=(0, 1.03),
               xlabel="候选文档数 K" if chinese else "Number of candidate documents (K)",
               ylabel="含正确来源的问题占比" if chinese else "Questions with correct source retrieved",
               title=f"正确来源进入前 K 个候选的比例（{n} 个可答问题）" if chinese else f"Retrieval recall at K ({n} answerable turns)")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="lower right", frameon=False)
        fig.savefig(HERE / "qa_recall.png", dpi=170, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
