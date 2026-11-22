"""Compare four-window averaging against adjacent-window changes on dev and test A."""
import argparse
import json
import tempfile
from pathlib import Path

import joblib

from experiment import id_hash, write_json
from heldout import read_answers, sample_ids
from tracing.data import balanced_sample, read_corpus, read_release_file, validate_text
from tracing.metrics import evaluate
from tracing.models import COMBINED, Classifiers, fit_linear_head, qwen_vectors, text_windows
from tracing.qwen import MODEL_ID, REVISION, QwenEncoder
from tracing.segments import CHANGE, COUNT, MEAN, METHODS, WIDTH, segment_vectors


def predict_all(rows, encoder, baseline, head_tail, segment_heads, batch_size):
    ids = [row["id"] for row in rows]
    texts = [row["text"] for row in rows]
    char_x = baseline["vectorizer"].transform(text_windows(texts))
    standard = qwen_vectors(encoder.encode(texts, batch_size), COMBINED)
    predicted = {
        "char-tfidf": baseline["heads"].predict(char_x, ids),
        COMBINED: head_tail["heads"].predict(standard, ids, source_x=char_x),
    }
    windows, _ = encoder.encode_segments(texts, batch_size)
    for method, heads in segment_heads.items():
        predicted[method] = heads.predict(segment_vectors(windows, method), ids, source_x=char_x)
    return predicted


def score_split(out, split, gold, predictions):
    metrics = {}
    for method, records in predictions.items():
        metrics[method] = evaluate(gold, records)
        with (out / f"{split}-{method}-predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in records.values():
                stream.write(json.dumps(row) + "\n")
    control, change = predictions[MEAN], predictions[CHANGE]
    corrected = regressed = 0
    for i, row in gold.items():
        correct_before = control[i]["label"] == row["label"]
        correct_after = change[i]["label"] == row["label"]
        corrected += int(correct_after and not correct_before)
        regressed += int(correct_before and not correct_after)
    return {"n": len(gold), "ids_sha256": id_hash(gold.values()), "results": metrics,
            "change_vs_mean_detection": {"corrected": corrected, "regressed": regressed}}


def experiment(args, out):
    config = json.loads((args.run / "summary.json").read_text())["configuration"]
    train_pool, dev_pool = read_corpus(args.data)
    train = balanced_sample(train_pool, config["train_per_family"])
    dev = balanced_sample(dev_pool, config["dev_per_family"])
    del train_pool, dev_pool
    if (id_hash(train), id_hash(dev)) != (config["train_ids_sha256"], config["dev_ids_sha256"]):
        raise ValueError("本次训练与验证 ID 和已有基线不同")
    baseline = joblib.load(args.run / "char-tfidf.joblib")
    head_tail = joblib.load(args.run / f"{COMBINED}.joblib")
    if config.get("revision") != REVISION:
        raise ValueError("首尾对照与四段实验需要相同的 Qwen 发布版本")
    encoder = QwenEncoder(args.model_dir, args.device)

    print(f"训练 {len(train)} 篇，四段共享一次编码", flush=True)
    windows, _ = encoder.encode_segments([r["text"] for r in train], args.batch_size)
    labels = [row["label"] for row in train]
    segment_heads = {}
    for method in METHODS:
        detector = fit_linear_head(segment_vectors(windows, method), labels)
        heads = Classifiers(detector, baseline["heads"].source)
        segment_heads[method] = heads
        joblib.dump({"method": method, "heads": heads, "vectorizer": baseline["vectorizer"]},
                    out / f"{method}.joblib")
    del windows

    print(f"验证 {len(dev)} 篇", flush=True)
    dev_pred = predict_all(dev, encoder, baseline, head_tail, segment_heads, args.batch_size)
    inputs = read_release_file(args.test_input or args.data, "test_a.jsonl")
    ids = sample_ids(inputs, "test_a", 768)
    for i in ids:
        validate_text(inputs[i])
    print(f"测试 A {len(ids)} 篇，四种方案完成预测后读取答案", flush=True)
    test_pred = predict_all([inputs[i] for i in ids], encoder, baseline, head_tail,
                            segment_heads, args.batch_size)
    answers = read_answers(args.gold, "test_a", inputs, ids)
    results = {
        "configuration": {
            "model": MODEL_ID, "revision": REVISION, "device": str(encoder.device),
            "dtype": str(encoder.dtype), "train_n": len(train), "train_ids_sha256": id_hash(train),
            "segments": COUNT, "tokens_per_segment": WIDTH, "max_tokens_per_text": COUNT * WIDTH,
            "batch_size": args.batch_size, "test_a_total": len(inputs),
            "test_a_selection": "smallest SHA256(20261122/test_a/{id}), then sort selected IDs",
            "detector_dimensions": {name: heads.detector.n_features_in_ for name, heads in segment_heads.items()},
            "shared_family_head": "char-tfidf", "logistic_C": segment_heads[MEAN].detector.C,
        },
        "dev": score_split(out, "dev", {r["id"]: r for r in dev}, dev_pred),
        "test_a": score_split(out, "test_a", answers, test_pred),
    }
    write_json(out / "summary.json", results)
    for split in ("dev", "test_a"):
        for name, item in results[split]["results"].items():
            print(f"{split:7} {name:24} 协作 F1 {item['class_f1'][2]:.4f}  联合 {item['joint_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="官方 dataset.zip 或固定划分目录")
    parser.add_argument("--test-input", type=Path, help="单独提供的测试 A 输入（默认从 --data 读取）")
    parser.add_argument("--gold", type=Path, required=True, help="官方 test_labeled.zip 或测试 A 答案")
    parser.add_argument("--run", type=Path, default=Path("runs/main"), help="已完成四种基线的目录")
    parser.add_argument("--out", type=Path, default=Path("runs/segments"), help="新建的模型与逐条预测目录")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("输出目录已存在，请通过 --out 选择新目录")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent, prefix=".segments-") as directory:
        staging = Path(directory) / "result"
        staging.mkdir()
        experiment(args, staging)
        staging.rename(args.out)


if __name__ == "__main__":
    main()
