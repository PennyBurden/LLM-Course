from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

COMBINED = "qwen-detect-char-source"


def text_windows(texts):
    # TF-IDF on contiguous characters. The newline marks a cut between windows.
    return [text if len(text) <= 800 else text[:400] + "\n" + text[-400:] for text in texts]


def text_vectors(train_texts):
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(1, 2), max_features=20000,
        min_df=2, sublinear_tf=True, dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(text_windows(train_texts))
    return vectorizer, matrix


def qwen_vectors(windows, method):
    head, tail = windows[:, 0], windows[:, 1]
    if method == "qwen-head":
        return normalize(head)
    if method in ("qwen-head-tail", COMBINED):
        # Normalize the windows separately so one cannot dominate by vector norm.
        return normalize(np.concatenate([normalize(head), normalize(tail)], axis=1))
    raise ValueError(f"未知特征方式 {method}")


def fit_linear_head(x, y):
    with threadpool_limits(limits=1):
        return LogisticRegression(C=8, max_iter=500, random_state=42).fit(x, y)


@dataclass
class Classifiers:
    detector: LogisticRegression
    source: LogisticRegression

    @classmethod
    def fit(cls, x, rows):
        labels = np.asarray([r["label"] for r in rows])
        families = np.asarray([r.get("family", -1) for r in rows])
        machine = labels == 1
        if set(labels) != set(range(3)) or set(families[machine]) != set(range(8)):
            raise ValueError("训练需要三个检测类别和八个机器家族")
        detector = fit_linear_head(x, labels)
        source = fit_linear_head(x[machine], families[machine])
        return cls(detector, source)

    def predict(self, x, ids, source_x=None):
        labels = self.detector.predict(x)
        families = np.full(len(labels), -1, dtype=int)
        machine = labels == 1
        if machine.any():
            family_features = x if source_x is None else source_x
            families[machine] = self.source.predict(family_features[machine])
        return {
            i: {"id": i, "label": int(label), "family": int(family)}
            for i, label, family in zip(ids, labels, families, strict=True)
        }
