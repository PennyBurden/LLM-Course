"""Predict official JSONL using a model produced by experiment.py."""
import argparse
import json
from pathlib import Path

import joblib

from tracing.data import read_jsonl, validate_text
from tracing.models import qwen_vectors, text_windows
from tracing.segments import METHODS, adjacent_distances, segment_vectors


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

    def _char_features(self, texts):
        if self.vectorizer is not None:
            return self.vectorizer.transform(text_windows(texts))

    def predict(self, texts, ids):
        char_x = self._char_features(texts)
        if self.encoder is None:
            return self.heads.predict(char_x, ids)
        if self.method in METHODS:
            windows, _ = self.encoder.encode_segments(texts)
            x = segment_vectors(windows, self.method)
        else:
            x = qwen_vectors(self.encoder.encode(texts), self.method)
        return self.heads.predict(x, ids, source_x=char_x)

    def predict_with_segments(self, text, record_id):
        """Predict and describe the same four-window forward pass used by the CLI."""
        if self.method not in METHODS:
            raise ValueError("当前模型未使用四段采样")
        windows, spans = self.encoder.encode_segments([text], with_offsets=True)
        rows = self.heads.predict(segment_vectors(windows, self.method), [record_id],
                                 source_x=self._char_features([text]))
        return rows[record_id], spans[0], adjacent_distances(windows[0])


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
