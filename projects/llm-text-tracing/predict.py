"""Predict official JSONL using a model produced by experiment.py."""
import argparse
import json
from pathlib import Path

import joblib

from tracing.data import read_jsonl, validate_text
from tracing.models import qwen_vectors, text_windows


class Predictor:
    def __init__(self, model_file, model_dir=None, device=None):
        bundle = joblib.load(model_file)
        self.method = bundle["method"]
        self.vectorizer = bundle["vectorizer"]
        self.heads = bundle["heads"]
        self.encoder = None
        if self.method.startswith("qwen-"):
            from tracing.qwen import QwenEncoder
            self.encoder = QwenEncoder(model_dir, device)

    def predict(self, texts, ids):
        char_x = self.vectorizer.transform(text_windows(texts)) if self.vectorizer is not None else None
        if self.encoder is None:
            return self.heads.predict(char_x, ids)
        x = qwen_vectors(self.encoder.encode(texts), self.method)
        return self.heads.predict(x, ids, source_x=char_x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--input", type=Path, help="每行含 id 和 text 的 JSONL")
    parser.add_argument("--output", type=Path, help="可选的 JSONL 输出路径")
    args = parser.parse_args()
    if args.input and args.output and args.input.resolve() == args.output.resolve():
        parser.error("输入和输出必须使用不同的文件")
    data = read_jsonl(args.input) if args.input else {1: {"id": 1, "text": args.text}}
    for row in data.values():
        validate_text(row)
    model = Predictor(args.model, args.model_dir)
    records = list(data.values())
    predicted = []
    for start in range(0, len(records), 64):
        batch = records[start:start + 64]
        predicted.extend(model.predict([r["text"] for r in batch], [r["id"] for r in batch]).values())
    lines = "".join(json.dumps(r) + "\n" for r in predicted)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(lines, encoding="utf-8")
    else:
        print(lines, end="")


if __name__ == "__main__":
    main()
