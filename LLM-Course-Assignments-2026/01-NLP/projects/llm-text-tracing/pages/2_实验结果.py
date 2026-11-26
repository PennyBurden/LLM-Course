import json

import pandas as pd
import streamlit as st

from review_experiment import load_scores, random_review, summarize
from tracing.data import ROOT


RESULTS = ROOT / "results"
POSITIONS = {"front": "前部", "middle": "中部", "end": "后部", "absent": "移除证据"}
st.set_page_config(page_title="检测与溯源实验结果", page_icon="📊")
st.title("实验结果")
review_tab, training_tab, context_tab = st.tabs(["低置信度复核", "训练样本量", "长文证据定位"])


def percent(part, whole):
    return f"{part / whole:.1%}" if whole else "—"


with review_tab:
    st.subheader("哪些检测结果值得交给人复核？")
    st.write("冻结字符检测器。分数取三个写作类别中的最高值，低于门槛的文本转入人工复核。")
    data = load_scores(RESULTS / "review_scores.json.gz")
    left, right = st.columns(2)
    split = left.radio("完整测试集", list(data["splits"]), horizontal=True,
                       format_func=lambda s: f"测试 {s[-1].upper()}")
    threshold = right.select_slider("自动判定最低分数", data["thresholds"], value=0.7,
                                    format_func=lambda t: f"{t:.2f}" if t else "全部")
    rows = data["splits"][split]
    counts = summarize(rows, threshold)
    review_n = counts["n"] - counts["auto"]
    metrics = st.columns(3)
    metrics[0].metric("自动判定覆盖率", percent(counts["auto"], counts["n"]))
    metrics[1].metric("自动判定中的准确率", percent(counts["auto_correct"], counts["auto"]))
    metrics[2].metric("待人工复核", f"{review_n:,} 篇")
    mean, low, high = random_review(counts)
    st.write(f"全部 {counts['n']:,} 篇中有 {counts['errors']:,} 篇原判错误。"
             f"按当前门槛送审，队列包含 {counts['review_errors']:,} 篇原判错误。"
             f"随机抽同样多篇，预计包含 {mean:,.0f} 篇，95% 随机范围为 {low:,} 至 {high:,} 篇。")
    st.table(pd.DataFrame([
        {"真实类别": data["labels"][c["class"]], "总篇数": c["n"], "自动判定": c["auto"],
         "自动判对": c["auto_correct"], "自动判定中的准确率": percent(c["auto_correct"], c["auto"])}
        for c in counts["by_class"]
    ]).set_index("真实类别"))
    with st.expander("查看仍被自动判错的例子"):
        examples = []
        for label, name in enumerate(data["labels"]):
            selected = max((r for r in rows if r[1] == label and r[2] != label and r[3] >= threshold),
                           key=lambda r: (r[3], -r[0]), default=None)
            if selected:
                record_id, _, predicted, score = selected
                examples.append({"真实类别": name, "样本 ID": record_id,
                                 "模型判定": data["labels"][predicted], "分数": f"{score:.4f}"})
        if examples:
            st.table(pd.DataFrame(examples).set_index("真实类别"))
            st.caption("每个真实类别取分数最高的一篇误判，同分选编号最小者。")
        else:
            st.write("这一门槛下，自动判定的文本中没有误判。")
    st.image(str(RESULTS / "review_coverage.png"), use_container_width=True)

with training_tab:
    st.subheader("增加标注样本，哪项能力变化更大？")
    curve = json.loads((RESULTS / "learning_curve.json").read_text(encoding="utf-8"))
    st.write(f"在相同的 {curve['validation_n']:,} 篇验证文本上比较五档训练量。每档用三组抽样分别训练，"
             "词表和两个分类器都从该次训练样本重新拟合。")
    st.image(str(RESULTS / "learning_curve.png"), use_container_width=True,
             caption="每个点是三组抽样的均值，误差线从最小值延伸到最大值。")
    sizes = sorted({r["size"] for r in curve["trials"]})
    size = st.select_slider("查看该训练量的三次结果", sizes, value=1536, format_func=lambda n: f"{n:,} 篇")
    st.table(pd.DataFrame([
        {"抽样种子": str(r["seed"]), "检测 Macro-F1": f"{r['metrics']['detect_macro_f1']:.4f}",
         "溯源 Macro-F1": f"{r['metrics']['source_macro_f1']:.4f}",
         "联合得分": f"{r['metrics']['joint_score']:.4f}"}
        for r in curve["trials"] if r["size"] == size
    ]).set_index("抽样种子"))
    st.caption("溯源分数包含检测环节漏掉的机器文本。联合得分给检测 80% 权重、溯源 20% 权重。")

with context_tab:
    st.subheader("移动或删除证据，问答系统会怎样？")
    sweep = json.loads((RESULTS / "context_sweep.json").read_text(encoding="utf-8"))
    trials = sweep["trials"]
    budgets = sorted({r["budget"] for r in trials})
    st.write("两个自编档案各有一个目标记录和两条相似记录。表中分子为答对的次数。移除证据后，正确行为是拒答。")
    matrix = []
    for budget in budgets:
        line = {"材料目标长度": f"{budget:,} token"}
        for key, name in POSITIONS.items():
            cell = [r for r in trials if r["budget"] == budget and r["position"] == key]
            line[name] = f"{sum(r['correct'] for r in cell)} / {len(cell)}"
        matrix.append(line)
    st.table(pd.DataFrame(matrix).set_index("材料目标长度"))
    left, right = st.columns(2)
    topic = left.selectbox("原始档案", list(sweep["cases"]), format_func={"archive": "批次档案", "instrument": "仪器校准"}.get)
    length = right.selectbox("材料目标长度", budgets, index=len(budgets)-1, format_func=lambda n: f"{n:,} token")
    position = st.radio("目标记录", list(POSITIONS), index=3, format_func=POSITIONS.get, horizontal=True)
    row = next(r for r in trials if (r["case"], r["budget"], r["position"]) == (topic, length, position))
    st.markdown(f"**问题**　{sweep['cases'][topic]['question']}")
    st.write(f"预期回答　{row['expected_answer']}")
    feedback = st.success if row["correct"] else st.warning
    feedback(f"系统实答　{row['answer']}")
    citation = row["citation"]
    location = "本次没有引用。"
    if citation.startswith("note-"):
        _, start, end = citation.split("-")
        location = f"引文在原文第 {int(start) + 1:,} 至 {int(end):,} 个字符。"
    st.caption(f"材料实长 {row['note_tokens']:,} token，提示读入 {row['prompt_tokens']:,} token。{location}")
    if row["verified"] and not row["correct"]:
        st.write("这句话可以在原文中找到，但它说的是另一条记录。逐字核对引文仍会漏掉对象错误。")
    with st.expander("查看材料中保留的两条相似记录"):
        for sentence in sweep["cases"][topic]["distractors"]:
            st.write(sentence)
