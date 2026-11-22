"""Train, evaluate and export models on the same reproducible benchmark."""
import argparse
import hashlib
import json
import platform
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn

from tracing.data import balanced_sample, read_corpus
from tracing.metrics import evaluate
from tracing.models import COMBINED, Classifiers, qwen_vectors, text_vectors, text_windows


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as staging:
        temporary = Path(staging) / "result.json"
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)


def record(out, name, heads, dev, dev_x, vectorizer=None, source_x=None):
    predictions = heads.predict(dev_x, [row["id"] for row in dev], source_x)
    metrics = evaluate({row["id"]: row for row in dev}, predictions)
    write_json(out / f"{name}-metrics.json", metrics)
    with (out / f"{name}-predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in predictions.values():
            stream.write(json.dumps(row) + "\n")
    joblib.dump({"method": name, "heads": heads, "vectorizer": vectorizer}, out / f"{name}.joblib")
    print(f"{name:16} 联合得分 {metrics['joint_score']:.4f}", flush=True)
    return metrics


def id_hash(rows):
    return hashlib.sha256("".join(f"{r['id']}\n" for r in rows).encode("ascii")).hexdigest()


def train_and_evaluate(args, out):
    train_pool, dev_pool = read_corpus(args.data)
    train = balanced_sample(train_pool, args.train_per_family)
    dev = balanced_sample(dev_pool, args.dev_per_family)
    train_text = [row["text"] for row in train]
    dev_text = [row["text"] for row in dev]
    settings = {
        "train_pool": len(train_pool), "dev_pool": len(dev_pool),
        "train_n": len(train), "dev_n": len(dev),
        "train_per_family": args.train_per_family, "dev_per_family": args.dev_per_family,
        "train_ids_sha256": id_hash(train), "dev_ids_sha256": id_hash(dev),
        "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
    }
    start = perf_counter()
    vec, train_x = text_vectors(train_text)
    dev_x = vec.transform(text_windows(dev_text))
    baseline = Classifiers.fit(train_x, train)
    scores = {"char-tfidf": record(out, "char-tfidf", baseline, dev, dev_x, vec)}
    settings["baseline_seconds"] = round(perf_counter() - start, 2)
    if not args.baseline_only:
        import torch
        import transformers
        from tracing.qwen import MODEL_ID, REVISION, WINDOW, QwenEncoder

        start = perf_counter()
        encoder = QwenEncoder(args.model_dir, args.device)
        train_features = encoder.encode(train_text, args.batch_size)
        dev_features = encoder.encode(dev_text, args.batch_size)
        for name in ("qwen-head", "qwen-head-tail"):
            heads = Classifiers.fit(qwen_vectors(train_features, name), train)
            qwen_dev = qwen_vectors(dev_features, name)
            scores[name] = record(out, name, heads, dev, qwen_dev)
            if name == "qwen-head-tail":
                combined = Classifiers(heads.detector, baseline.source)
                scores[COMBINED] = record(out, COMBINED, combined, dev, qwen_dev, vec, dev_x)
        settings.update(model=MODEL_ID, revision=REVISION, tokens_per_window=WINDOW,
                        device=str(encoder.device), dtype=str(encoder.dtype),
                        torch=torch.__version__, transformers=transformers.__version__,
                        batch_size=args.batch_size, qwen_seconds=round(perf_counter() - start, 2))
    write_json(out / "summary.json", {"configuration": settings, "results": scores})


def run(args):
    target = args.out or Path("runs/main")
    if target.exists() and args.out is None:
        target = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if target.exists():
        raise FileExistsError(f"{target} 已有实验记录，请另选 --out 目录")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".training-") as staging:
        result = Path(staging) / "result"
        result.mkdir()
        train_and_evaluate(args, result)
        result.rename(target)
    print(f"模型、预测与评分已保存到 {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="官方 dataset.zip、原始 train.jsonl 或固定划分目录")
    parser.add_argument("--out", type=Path, help="新实验目录，默认首次使用 runs/main，后续按时间命名")
    parser.add_argument("--train-per-family", type=int, default=64)
    parser.add_argument("--dev-per-family", type=int, default=16)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--model-dir", type=Path, help="已下载的 Qwen2.5-0.5B 本地目录")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=12)
    run(parser.parse_args())
