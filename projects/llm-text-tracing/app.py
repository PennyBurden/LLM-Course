import os
from pathlib import Path

import streamlit as st

from predict import Predictor
from tracing.data import FAMILIES, LABELS
from tracing.models import COMBINED

st.set_page_config(page_title="大模型生成文本检测与溯源", page_icon="🔎")
st.title("大模型生成文本检测与溯源")
st.write("输入一段文本，查看检测类别。系统会为判为纯机器生成的文本给出候选模型家族。")
model_dir = Path(os.environ.get("TRACE_RUN_DIR", "runs/main"))
available = {p.stem: p for p in sorted(model_dir.glob("*.joblib"))}
if not available:
    st.info("请先在项目根目录运行 README 中的实验命令，再刷新页面。")
    st.stop()

names = {"char-tfidf": "字符基线", "qwen-head": "Qwen 开头窗口",
         "qwen-head-tail": "Qwen 首尾窗口", COMBINED: "Qwen 检测 + 字符溯源"}
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
            row = predictor.predict([source], [1])[1]
        st.subheader(LABELS[row["label"]])
        if row["label"] == 1:
            st.write(f"候选模型家族　**{FAMILIES[row['family']]}**")
        st.caption("这是基于课程数据训练的统计预测，个别文本的作者身份仍需其他证据核实。")
        st.json(row)
