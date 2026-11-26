"""Score every official test text with the frozen character detector before opening its labels."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from scipy.stats import hypergeom

from heldout import read_answers
from tracing.data import LABELS, read_release_file, validate_gold, validate_text
from tracing.models import text_windows


THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_OUTPUT = Path("results/review_scores.json.gz")


def score_split(data, gold, split, bundle):
    inputs = read_release_file(data, f"{split}.jsonl")
    ids = sorted(inputs)
    predictions, scores = [], []
    detector = bundle["heads"].detector
    for start in range(0, len(ids), 256):
        records = [inputs[i] for i in ids[start:start + 256]]
        for row in records:
            validate_text(row)
        matrix = bundle["vectorizer"].transform(text_windows([row["text"] for row in records]))
        probabilities = detector.predict_proba(matrix)
        predictions.extend(detector.classes_[probabilities.argmax(axis=1)].tolist())
        scores.extend(probabilities.max(axis=1).tolist())
    labels = read_answers(gold, split, inputs, ids)
    rows = []
    for i, prediction, score in zip(ids, predictions, scores, strict=True):
        validate_gold(labels[i])
        rows.append([i, labels[i]["label"], prediction, score])
    return rows


def summarize(rows, threshold):
    """Counts refer to the three writing classes. An abstention sends one text to review."""
    values = np.asarray(rows)
    truth, guess, confidence = values[:, 1], values[:, 2], values[:, 3]
    automatic = confidence >= threshold
    correct = truth == guess
    by_class = []
    for label in range(len(LABELS)):
        members = truth == label
        accepted = automatic & members
        by_class.append({"class": label, "n": int(members.sum()), "auto": int(accepted.sum()),
                         "auto_correct": int((accepted & correct).sum())})
    return {"n": len(values), "auto": int(automatic.sum()), "auto_correct": int((automatic & correct).sum()),
            "errors": int((~correct).sum()), "review_errors": int((~automatic & ~correct).sum()),
            "by_class": by_class}


def load_scores(path=DEFAULT_OUTPUT):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def random_review(counts):
    """Errors captured by a uniformly random review queue of the same size, without replacement."""
    draws = counts["n"] - counts["auto"]
    mean = draws * counts["errors"] / counts["n"]
    low, high = hypergeom.ppf([0.025, 0.975], counts["n"], counts["errors"], draws)
    return mean, int(low), int(high)


def run(args):
    if args.out.resolve() in {p.resolve() for p in (args.data, args.gold, args.model)}:
        raise ValueError("逐篇结果需要使用独立的输出文件")
    bundle = joblib.load(args.model)
    if bundle["method"] != "char-tfidf" or set(bundle["heads"].detector.classes_) != set(range(len(LABELS))):
        raise ValueError("本实验需要三分类字符基线的已训练模型")
    results = {"model": bundle["method"], "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
               "columns": ["id", "true_label", "predicted_label", "max_class_score"],
               "labels": LABELS, "thresholds": THRESHOLDS, "splits": {}}
    for split in ("test_a", "test_b"):
        rows = score_split(args.data, args.gold, split, bundle)
        results["splits"][split] = rows
        print(split, summarize(rows, 0.7), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as temp:
        staged = Path(temp) / "scores.gz"
        payload = json.dumps(results, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        staged.write_bytes(gzip.compress(payload, mtime=0))
        staged.replace(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--gold", type=Path, default=Path("data/test_labeled.zip"))
    parser.add_argument("--model", type=Path, default=Path("runs/main/char-tfidf.joblib"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())
