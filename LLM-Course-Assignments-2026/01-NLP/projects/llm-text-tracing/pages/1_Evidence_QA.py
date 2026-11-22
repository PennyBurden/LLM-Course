import os
from pathlib import Path

import streamlit as st

from qa.engine import answer
from qa.index import EvidenceIndex
from rag_qa import DEFAULT_INDEX

st.set_page_config(page_title="溯源证据问答", page_icon="📖")
st.title("溯源证据问答")
st.write("可以查询公开案例的标注，比较测试 A、B 的实验结果，或继续追问上一条证据。")
path = Path(os.environ.get("TRACE_QA_INDEX", DEFAULT_INDEX))
if not (path / "manifest.json").is_file():
    st.info("先在项目根目录运行 python rag_qa.py build --data data/dataset.zip，然后刷新页面。")
    st.stop()


@st.cache_resource
def load_index(directory, modified):
    return EvidenceIndex(directory)


@st.cache_resource
def load_generator(directory):
    from qa.generator import EvidenceGenerator
    return EvidenceGenerator(directory)


def show(row):
    with st.chat_message("user"):
        st.text(row["question"])
    with st.chat_message("assistant"):
        st.text(row["answer"])
        if row["verified"]:
            with st.expander(f"已核对引用 · {row['title']}", expanded=True):
                st.caption(f"证据编号　{row['citation']}")
                st.text(row["excerpt"])
                if row["url"]:
                    st.link_button("打开原始资料", row["url"])


turns = st.session_state.setdefault("qa_turns", [])
with st.expander("补充自己的材料"):
    notes = st.text_area("材料全文将随本轮问题传入模型", key="qa_notes", height=140)
    st.caption("问题、材料、最近四轮对话和答案共支持 49,152 个 token。")
if turns and st.button("清空本次对话"):
    turns.clear()
for turn in turns:
    show(turn)
question = st.chat_input("例如：测试 A 上字符基线和首尾方案的联合得分是多少？")
if question:
    try:
        with st.spinner("正在检索证据并核对引用"):
            evidence = load_index(path, (path / "manifest.json").stat().st_mtime_ns)
            row = answer(evidence, load_generator(os.environ.get("QWEN_MODEL_DIR")), question, turns, notes)
        turns.append(row)
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))
