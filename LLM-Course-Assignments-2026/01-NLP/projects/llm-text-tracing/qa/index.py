"""Build a persistent vector index over annotated training cases and original experiment records."""
import hashlib
import json
import re
import sqlite3
import tempfile
import unicodedata
from contextlib import closing
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from tracing.data import FAMILIES, LABELS, ROOT, read_corpus

CASE_COUNT = 5120
COURSE_EVIDENCE = "https://github.com/PennyBurden/LLM-Course/blob/1c9857dee52ea0aa8689c3408779b5c0a40821c9/projects/llm-text-tracing/"
CASE_SOURCE = "https://github.com/ASCII-LAB/CCKS26-Task6-LLM-Generated-Text-Detect-Trace/tree/09833974f03ca83a52f43830f93913c9a5b26f20"
METHOD_NAMES = {"human-mix-machine": "人机混合写作", "machine-modify-human": "机器修改人工文本",
                "machine-continue-human (short prefix)": "机器续写人类短前缀",
                "machine-continue-human (long prefix)": "机器续写人类长前缀"}


def clean(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def chunks(text, width=480, overlap=60):
    for start in range(0, len(text), width - overlap):
        yield text[start:start + width]
        if start + width >= len(text):
            break


def case_answer(row):
    label = row["label"]
    answer = f"训练案例 {row['id']} 的公开标注为{LABELS[label]}。"
    if label == 1:
        return answer + f"生成模型家族为 {FAMILIES[row['family']]}。"
    if label == 2 and row.get("method"):
        return answer + f"协作方式为{METHOD_NAMES.get(row['method'], row['method'])}，未标注机器家族。"
    return answer + "未标注机器家族。"


def documents(data):
    cards = json.loads((ROOT / "qa/cards.json").read_text(encoding="utf-8"))
    for key, title, answer, path, scope in cards:
        yield (key, "fact", scope, title, answer, COURSE_EVIDENCE + path, answer)
    train, _ = read_corpus(data)
    ranked = sorted(train, key=lambda i: hashlib.sha256(f"nlp-cases-v1/{i}".encode()).digest())
    seen = set()
    for i in ranked:
        row = train[i]
        original = clean(row["text"])
        digest = hashlib.sha256(original.casefold().encode()).digest()
        if not original or digest in seen:
            continue
        seen.add(digest)
        title = f"训练案例 {i} 的来源标注和写作方式"
        yield (f"case-{i}", "case", "", title, case_answer(row), CASE_SOURCE, original)
        if len(seen) == CASE_COUNT:
            return
    raise ValueError(f"训练集只有 {len(seen)} 条去重文本")


def build(data, target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"{target} 已有索引，请另选新目录")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".qa-") as tmp:
        stage = Path(tmp) / "index"
        stage.mkdir()
        bodies, titles, ids, case_ids = [], [], [], []
        chinese_originals = 0
        with closing(sqlite3.connect(stage / "evidence.sqlite")) as db:
            db.executescript("""
                CREATE TABLE docs(id TEXT PRIMARY KEY, kind TEXT, scope TEXT, title TEXT, answer TEXT, url TEXT);
                CREATE TABLE chunks(n INTEGER PRIMARY KEY, doc TEXT REFERENCES docs(id), body TEXT);
            """)
            for key, kind, scope, title, answer, url, body in documents(data):
                db.execute("INSERT INTO docs VALUES(?,?,?,?,?,?)", (key, kind, scope, title, answer, url))
                if kind == "case":
                    case_ids.append(key)
                    chinese_originals += int(len(re.findall(r"[\u4e00-\u9fff]", body[:500])) >= 24)
                ids.append(key)
                for part in chunks(body):
                    db.execute("INSERT INTO chunks VALUES(?,?,?)", (len(bodies), key, part))
                    bodies.append(title + " " + answer + " " + part)
                    titles.append(title)
            db.commit()
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=45000,
                                     sublinear_tf=True, dtype=np.float32)
        matrix = vectorizer.fit_transform(bodies)
        sparse.save_npz(stage / "content.npz", matrix)
        sparse.save_npz(stage / "titles.npz", vectorizer.transform(titles))
        joblib.dump(vectorizer, stage / "tfidf.joblib")
        stats = {"case_documents": len(case_ids), "fact_documents": len(ids) - len(case_ids),
                 "unique_documents": len(ids), "chunks": len(bodies), "vector_dimensions": matrix.shape[1],
                 "case_ids_sha256": hashlib.sha256("\n".join(sorted(case_ids)).encode()).hexdigest(),
                 "case_originals_with_24_han_in_first_500": chinese_originals,
                 "chunk_characters": 480, "overlap_characters": 60}
        (stage / "manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
        stage.rename(target)
    return stats


class EvidenceIndex:
    def __init__(self, path):
        path = Path(path)
        self.vectorizer = joblib.load(path / "tfidf.joblib")
        self.content = sparse.load_npz(path / "content.npz")
        self.titles = sparse.load_npz(path / "titles.npz")
        with closing(sqlite3.connect(path / "evidence.sqlite")) as db:
            db.row_factory = sqlite3.Row
            self.docs = {r["id"]: dict(r) for r in db.execute("SELECT * FROM docs")}
            self.parts = [dict(r) for r in db.execute("SELECT doc, body FROM chunks ORDER BY n")]
        if self.content.shape != self.titles.shape or self.content.shape[0] != len(self.parts):
            raise ValueError("文本与向量索引未对齐，请重建索引")
        self.first = {}
        fact_rows = []
        for i, part in enumerate(self.parts):
            self.first.setdefault(part["doc"], part["body"])
            if self.docs[part["doc"]]["kind"] == "fact":
                fact_rows.append(i)
        self.fact_rows = np.asarray(fact_rows, dtype=int)
        self.fact_content = self.content[self.fact_rows]
        self.fact_titles = self.titles[self.fact_rows]

    def lookup(self, key):
        doc = self.docs.get(key)
        return dict(doc, body=self.first[key], score=1.0) if doc else None

    def ranked(self, query, improved, kind=None, scope=None, limit=5):
        q = self.vectorizer.transform([clean(query)])
        content, titles = (self.fact_content, self.fact_titles) if kind == "fact" else (self.content, self.titles)
        scores = (content @ q.T).toarray().ravel()
        if improved:
            scores = 0.60 * scores + 0.40 * (titles @ q.T).toarray().ravel()
        seen, hits = set(), []
        for idx in np.argsort(-scores, kind="stable"):
            if scores[idx] <= 0:
                break
            part = self.parts[self.fact_rows[idx] if kind == "fact" else idx]
            key = part["doc"]
            doc = self.docs[key]
            if key in seen or (kind and doc["kind"] != kind) or (scope and doc["scope"] != scope):
                continue
            hits.append(dict(doc, score=float(scores[idx]), body=part["body"]))
            seen.add(key)
            if len(hits) == limit:
                break
        return hits
