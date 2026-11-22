# 证据检索复测

新增 24 轮上，正确数 14 → 20，引用微平均 F1 为 0.529 → 0.800。两次有引用的误答和两次错误拒答保留在逐题结果中。原 31 轮用于开发。两套题都属于项目自编。条件、时限和自查边界见[协议](contract.md)。

[新增问题](challenge.json) · [旧版输出](runs/evidence-v2/previous.json) · [新版输出](runs/evidence-v2/current.json) · [开发题输出](runs/evidence-v2/dev.json) · [配对计时](runs/evidence-v2/timing.json)

在项目根目录用安装了课程依赖的 Python 执行，先准备上一版项目和同一份本地索引与模型。课程 Fork 中上一版项目为提交 `4a85aa1` 下的 `LLM-Course-Assignments-2026/01-NLP/projects/llm-text-tracing`。以下命令假设其内容在 `../previous-project`。

```bash
python research/evaluate.py --project ../previous-project --questions research/challenge.json --index runs/qa-index --model-dir models/Qwen2.5-0.5B --out runs/replay-old.json
python research/evaluate.py --project . --questions research/challenge.json --index runs/qa-index --model-dir models/Qwen2.5-0.5B --out runs/replay-new.json
python research/time_retrieval.py --previous ../previous-project --out runs/replay-timing.json
```

候选实现 SHA-256：`cb6c1f6587d91775746ca12f5bdb761b2d6807ab5280feb4764e45cfe4d8c503`（问答），`60f81012dbd9426c089b0fbe6e4d84a689b05f8cf558241ce64e5a546a9a6256`（索引）。挑战题与评分器的哈希见账本或逐题文件。