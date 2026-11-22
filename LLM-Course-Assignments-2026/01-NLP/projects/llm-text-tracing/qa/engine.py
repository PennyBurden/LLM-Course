"""Resolve conversational scope, retrieve evidence, then verify a generated quotation."""
import re

from .index import clean

REFUSAL = "现有资料没有足够证据回答这个问题。"
CASE = re.compile(r"(?:案例|样本|编号|记录|case|id)\s*[-#：:]?\s*(\d+)", re.I)
SPLIT = re.compile(r"(?:测试(?:集)?|测评|评测|换成|那么|那|改看|换到|在)\s*([AB])(?=[^a-z]|$)", re.I)
FOLLOW = re.compile(r"^(那|那么|它|这条|这个|上面|刚才|再问|其中|换成|还有|这个数|如果换)")
UNRECORDED = re.compile(r"真实姓名|作者姓名|学号|身份证|手机号|家庭住址|精确版本|具体版本号|排行榜|榜单|全量.{0,8}(得分|分数|准确率)|训练.{0,6}(费用|花费|电费)")
ALIASES = {"总分": " 联合得分", "总成绩": " 联合得分", "分数": " 得分", "误伤": " 误报",
           "误判成协作": " 误报", "找出": " 检出", "占比": " 权重", "头尾": " 首尾",
           "验证集": " 验证子集", "模型厂商": " 家族", "多少种": " 清单", "窗口": " 采样"}


def scope_of(text):
    match = SPLIT.search(text)
    if match:
        return match.group(1).upper()
    if "验证" in text:
        return "DEV"
    return ""


def note_hits(question, text):
    """Keep the original character span so the model's quote can be verified after decoding."""
    normalized = clean(question).lower()
    query = {normalized[i:i + 2] for i in range(len(normalized) - 1)}
    hits, seen = [], set()
    for match in re.finditer(r"[^。！？!?\n]+[。！？!?]?", text):
        for offset in range(0, len(match[0]), 400):
            part = match[0][offset:offset + 400]
            answer = part.strip()
            if len(answer) < 2 or answer in seen:
                continue
            seen.add(answer)
            value = clean(answer).lower()
            grams = {value[i:i + 2] for i in range(len(value) - 1)}
            common = len(grams & query)
            if common < 2:
                continue
            start = match.start() + offset + len(part) - len(part.lstrip())
            end = start + len(answer)
            score = common / max(1, (len(query) * len(grams)) ** 0.5)
            hits.append({"id": f"note-{start}-{end}", "kind": "note", "scope": "",
                         "title": f"补充材料第 {start + 1} 至 {end} 字符", "answer": answer,
                         "body": answer, "url": "", "score": score})
    return sorted(hits, key=lambda h: -h["score"])[:5]


def retrieve(index, question, history, notes, improved):
    if not improved:
        return index.ranked(question, False)
    previous = history[-1] if history else {}
    case, scope = CASE.search(question), scope_of(question)
    follow = bool(previous and FOLLOW.search(question))
    if notes:
        local = note_hits(question, notes)
        if local and local[0]["score"] >= 0.17:
            return local
    if UNRECORDED.search(question):
        return []
    key = "case-" + str(int(case.group(1))) if case else ""
    if not key and follow and not scope and previous.get("citation", "").startswith("case-"):
        key = previous["citation"]
    if key:
        hit = index.lookup(key)
        return [hit] if hit else []
    query = question
    if follow:
        prior_question = previous["question"]
        scope = scope or scope_of(prior_question)
        # Keep the subject of the previous turn, remove its old A/B scope.
        query += " " + SPLIT.sub("", prior_question)
    for word, expanded in ALIASES.items():
        if word in query:
            query += expanded
    kind = "case" if re.search(r"(找|举|搜索|检索).{0,5}(案例|原文|例子)", question) else "fact"
    hits = index.ranked(query, True, kind=kind, scope=scope)
    if scope:
        # Require the requested experiment and metric to occur in the same evidence card.
        dimensions = (("四段", "相邻差异"), ("误报", "误伤"), ("漏检", "漏掉"),
                      ("联合得分", "总分", "总成绩"), ("召回", "检出", "找出"))
        for terms in dimensions:
            if any(term in question for term in terms):
                hits = [h for h in hits if any(term in h["title"] + h["answer"] for term in terms)]
    return hits if hits and hits[0]["score"] >= 0.095 else []


def answer(index, generator, question, history=(), notes="", improved=True):
    if not question.strip():
        raise ValueError("请输入问题")
    hits = retrieve(index, question, history, notes, improved)
    result = {"question": question, "answer": REFUSAL, "citation": "", "title": "", "url": "",
              "excerpt": "", "prompt_tokens": 0, "retrieved": [p["id"] for p in hits], "verified": False}
    if not hits:
        return result
    passages = hits[:3]
    sources = "\n".join(f"〔{p['id']}〕{p['title']}。{p['answer']}" for p in passages)
    dialogue = "\n".join("用户：" + t["question"] + "\n回答：" + t["answer"] for t in history[-4:]) if improved else ""
    prompt = ("以下是有关生成文本检测与溯源的资料。请摘录回答问题的原句。\n"
              + ("补充材料全文：\n" + notes + "\n" if notes else "")
              + "备选证据：\n" + sources + "\n" + dialogue + "\n问题：" + question + "\n答：")
    selected, quote, count = generator.choose(prompt, passages)
    hit = passages[selected]
    if hit["kind"] == "note":
        _, lo, hi = hit["id"].split("-")
        verified = notes[int(lo):int(hi)] == quote
    else:
        original = index.lookup(hit["id"])
        verified = original is not None and original["answer"] == quote
    result["prompt_tokens"] = count
    if verified:
        result.update(answer=quote, citation=hit["id"], title=hit["title"], url=hit["url"],
                      excerpt=hit["body"], verified=True)
    return result
