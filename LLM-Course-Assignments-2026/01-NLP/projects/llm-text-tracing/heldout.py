"""Evaluate frozen models on the same ID-only sample of an official test split."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from predict import Predictor
from tracing.data import read_release_file, validate_text
from tracing.metrics import evaluate
from tracing.models import COMBINED


def sample_ids(inputs, split, size):
    if not 0 <= size <= len(inputs):
        raise ValueError(f"--size 需在 0 到 {len(inputs)} 之间，0 表示全量")
    def rank(i):
        return hashlib.sha256(f"20261122/{split}/{i}".encode("ascii")).digest()
    return sorted(sorted(inputs, key=rank)[:size or len(inputs)])


def read_answers(source, split, inputs, ids):
    gold = read_release_file(source, f"{split}_labeled.jsonl")
    if gold.keys() != inputs.keys():
        raise ValueError("输入文件和答案文件的完整 ID 集合不同")
    for i in ids:
        if "text" in gold[i] and gold[i]["text"] != inputs[i]["text"]:
            raise ValueError(f"id={i} 的答案文本和输入文本不一致，请检查 split")
    return {i: gold[i] for i in ids}


def run(args):
    protected = {args.input.resolve(), args.gold.resolve()}
    protected.update((args.run / f"{name}.joblib").resolve() for name in ("char-tfidf", COMBINED))
    if args.out.resolve() in protected:
        raise ValueError("评分记录需要使用独立的输出文件")
    inputs = read_release_file(args.input, f"{args.split}.jsonl")
    ids = sample_ids(inputs, args.split, args.size)
    texts = []
    for i in ids:
        validate_text(inputs[i])
        texts.append(inputs[i]["text"])
    predictions = {}
    for method in ("char-tfidf", COMBINED):
        classifier = Predictor(args.run / f"{method}.joblib", args.model_dir)
        predictions[method] = classifier.predict(texts, ids)
    # Open the answer file only after both models have finished inference.
    selected_gold = read_answers(args.gold, args.split, inputs, ids)
    result = {
        "split": args.split, "total_in_split": len(inputs), "sample_n": len(ids),
        "ids_sha256": hashlib.sha256("".join(f"{i}\n" for i in ids).encode("ascii")).hexdigest(),
        "results": {method: evaluate(selected_gold, rows) for method, rows in predictions.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.out.parent, encoding="utf-8", delete=False) as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        staged = stream.name
    os.replace(staged, args.out)
    for name, metrics in result["results"].items():
        print(f"{name:24} {metrics['joint_score']:.4f}")
    print(f"评分已保存到 {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="官方 dataset.zip、输入目录或单个测试 JSONL")
    parser.add_argument("--gold", type=Path, required=True, help="test_labeled.zip、答案目录或同一 split 的答案 JSONL")
    parser.add_argument("--split", choices=["test_a", "test_b"], default="test_b")
    parser.add_argument("--run", type=Path, default=Path("runs/main"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--out", type=Path, default=Path("runs/test_score.json"))
    run(parser.parse_args())
