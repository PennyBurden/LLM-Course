"""Check the fixed split and analyze eight-family attribution on its full validation set."""
import argparse
import hashlib
import unicodedata
from collections import Counter
from pathlib import Path

import joblib
from sklearn.metrics import f1_score

from experiment import write_json
from tracing.data import FAMILIES, read_corpus, read_release_file
from tracing.metrics import evaluate
from tracing.models import text_windows

RELEASE_BLOB = "76e6496cd39489e81e56598b7263283319016300"


def release_identity(path):
    size = path.stat().st_size
    blob = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            blob.update(block)
            checksum.update(block)
    if blob.hexdigest() != RELEASE_BLOB:
        raise ValueError("压缩包与项目固定的官方 Git 版本不一致")
    return {"git_blob": blob.hexdigest(), "sha256": checksum.hexdigest(), "bytes": size}


def split_stats(rows):
    texts = set()
    labels, families = Counter(), Counter()
    for r in rows.values():
        normalized = " ".join(unicodedata.normalize("NFKC", r["text"]).casefold().split())
        texts.add(hashlib.sha256(normalized.encode("utf-8")).digest())
        labels[r["label"]] += 1
        if r["label"] == 1:
            families[r["family"]] += 1
    return {"records": len(rows), "labels": dict(sorted(labels.items())),
            "machine_families": dict(sorted(families.items())),
            "duplicate_normalized_texts": len(rows) - len(texts)}, texts


def compare_fields(a, b, fields):
    if a.keys() != b.keys():
        raise ValueError("官方数据和课程划分的 ID 不一致")
    for i, row in a.items():
        if any(row.get(key) != b[i].get(key) for key in fields):
            raise ValueError(f"id={i} 的字段不一致")
    return len(a)


def family_analysis(dev, model_file):
    bundle = joblib.load(model_file)
    vectorizer, heads = bundle["vectorizer"], bundle["heads"]
    predictions, true_families, candidates = {}, [], []
    candidate_matrix = [[0] * 8 for _ in FAMILIES]
    joint_matrix = [[0] * 9 for _ in FAMILIES]
    ids = sorted(dev)
    for start in range(0, len(ids), 256):
        batch = ids[start:start + 256]
        x = vectorizer.transform(text_windows([dev[i]["text"] for i in batch]))
        predictions.update(heads.predict(x, batch))
        machine = [k for k, i in enumerate(batch) if dev[i]["label"] == 1]
        if not machine:
            continue
        for k, candidate in zip(machine, heads.source.predict(x[machine]), strict=True):
            i = batch[k]
            family = dev[i]["family"]
            predicted = predictions[i]["family"]
            true_families.append(family)
            candidates.append(int(candidate))
            candidate_matrix[family][candidate] += 1
            joint_matrix[family][predicted if predicted != -1 else 8] += 1
    scores = evaluate(dev, predictions)
    candidate_f1 = f1_score(true_families, candidates, labels=list(range(8)), average=None, zero_division=0)
    return {
        "method": "char-tfidf", "full_dev_scores": scores,
        "candidate_macro_f1": float(candidate_f1.mean()),
        "by_family": [{"id": i, "family": name, "n": sum(joint_matrix[i]),
                       "missed_machine": joint_matrix[i][-1], "correct_joint": joint_matrix[i][i],
                       "candidate_f1": float(candidate_f1[i]), "joint_source_f1": scores["family_f1"][i]}
                      for i, name in enumerate(FAMILIES)],
        "candidate_confusion": candidate_matrix, "joint_confusion": joint_matrix,
        "joint_columns": [*FAMILIES, "检测漏检"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="固定版本官方 dataset.zip 或固定划分目录")
    parser.add_argument("--compare-split", type=Path, help="按 ID 核对的本地课程划分目录")
    parser.add_argument("--run", type=Path, default=Path("runs/main"))
    parser.add_argument("--out", type=Path, default=Path("results/data_and_families.json"))
    args = parser.parse_args()
    identity = release_identity(args.data) if args.data.suffix == ".zip" else None
    train, dev = read_corpus(args.data)
    split = {}
    split["train"], train_hashes = split_stats(train)
    split["dev"], dev_hashes = split_stats(dev)
    split["overlapping_normalized_texts"] = len(train_hashes & dev_hashes)
    if split["overlapping_normalized_texts"]:
        raise ValueError("训练与验证存在规范化全文交集")
    matches = {}
    if args.compare_split:
        local_train, local_dev = read_corpus(args.compare_split)
        fields = ("text", "label", "family", "method")
        matches["train"] = compare_fields(train, local_train, fields)
        matches["dev"] = compare_fields(dev, local_dev, fields)
        for name in ("test_a", "test_b"):
            filename = name + ".jsonl"
            matches[name] = compare_fields(read_release_file(args.data, filename),
                                           read_release_file(args.compare_split, filename), ("text",))
    report = {"release": identity, "split": split, "matched_local_records": matches,
              "attribution": family_analysis(dev, args.run / "char-tfidf.joblib")}
    write_json(args.out, report)
    print(args.out, "dev", len(dev), "机器样本", report["attribution"]["full_dev_scores"]["n_machine"])


if __name__ == "__main__":
    main()
