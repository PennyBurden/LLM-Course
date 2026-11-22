import os
from pathlib import Path

import streamlit as st

from predict import Predictor
from tracing.data import FAMILIES, LABELS
from tracing.models import COMBINED
from tracing.segments import CHANGE, MEAN, METHODS

st.set_page_config(page_title="大模型生成文本检测与溯源", page_icon="🔎")
st.title("大模型生成文本检测与溯源")
st.page_link("pages/1_Evidence_QA.py", label="进入溯源证据问答")
st.write("输入一段文本，查看检测类别。系统会为判为纯机器生成的文本给出候选模型家族。")
if "TRACE_RUN_DIR" in os.environ:
    model_dirs = [Path(os.environ["TRACE_RUN_DIR"])]
else:
    model_dirs = [Path("runs/main"), Path("runs/segments")]
available = {p.stem: p for directory in model_dirs for p in sorted(directory.glob("*.joblib"))}
if not available:
    st.info("请先在项目根目录运行 README 中的实验命令，再刷新页面。")
    st.stop()

names = {"char-tfidf": "字符基线", "qwen-head": "Qwen 开头窗口",
         "qwen-head-tail": "Qwen 首尾窗口", COMBINED: "Qwen 检测 + 字符溯源",
         MEAN: "Qwen 四段均值", CHANGE: "Qwen 四段均值 + 相邻差异"}
options = list(available)
method = st.selectbox("检测模型", options, index=options.index(COMBINED) if COMBINED in options else 0,
                      format_func=lambda key: names.get(key, key))
source = st.text_area("待检测文本", height=180, placeholder="粘贴一段中文或英文文本……")


@st.cache_resource
def load_model(path, modified, weights_dir):
    return Predictor(path, weights_dir)


if st.button("检测文本", type="primary"):
    if not source.strip():
        st.warning("请先输入文本。")
    else:
        file = available[method]
        with st.spinner("正在分析文本"):
            predictor = load_model(file, file.stat().st_mtime_ns, os.environ.get("QWEN_MODEL_DIR"))
            if method in METHODS:
                row, spans, distances = predictor.predict_with_segments(source, 1)
            else:
                row = predictor.predict([source], [1])[1]
        st.subheader(LABELS[row["label"]])
        if row["label"] == 1:
            st.write(f"候选模型家族　**{FAMILIES[row['family']]}**")
        st.caption("这是基于课程数据训练的统计预测，个别文本的作者身份仍需其他证据核实。")
        st.json(row)
        if method in METHODS:
            with st.expander("查看采样片段与相邻差异", expanded=True):
                st.caption("引号内是模型读取的原文。余弦距离越大，两段的表示差异越大。话题或语言变化也会影响这个值。")
                for i, (start, end) in enumerate(spans):
                    st.markdown(f"**片段 {i + 1}** · 原文第 {start + 1} 至 {end} 字符")
                    st.text(f"「{source[start:end]}」")
                    if i > 0:
                        st.caption(f"与片段 {i} 的余弦距离　{distances[i - 1]:.3f}")
