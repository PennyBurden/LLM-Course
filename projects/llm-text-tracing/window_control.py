"""Compare two positions with the same Qwen token budget and head dimensions."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from experiment import id_hash, write_json
from tracing.data import ROOT, balanced_sample, read_corpus
from tracing.metrics import evaluate
from tracing.models import Classifiers, fit_linear_head, qwen_vectors, text_windows
from tracing.qwen import MODEL_ID, REVISION, QwenEncoder

WIDTH = 32


def position_features(encoder, rows, batch_size=12):
    """Encode 3 windows once, then reuse the first in both 2-window conditions."""
    batches = []
    different_positions = 0
    for start in range(0, len(rows), batch_size):
        texts = [r["text"] for r in rows[start:start + batch_size]]
        sequences = encoder.tokenizer(texts, add_special_tokens=False, verbose=False)["input_ids"]
        windows = []
        for seq in sequences:
            if not seq:
                raise ValueError("分词后文本为空")
            following = seq[WIDTH:2 * WIDTH]
            tail = seq[-WIDTH:] if len(seq) > 2 * WIDTH else following
            different_positions += int(following != tail)
            windows.extend((seq[:WIDTH], following, tail))
        batches.append(encoder.pool(windows).reshape(len(texts), 3, -1))
    return np.concatenate(batches), different_positions


@torch.inference_mode()
def check_pooling(encoder):
    sequences = encoder.tokenizer(["细雨。", "测试注意力掩码。" * 6], add_special_tokens=False)["input_ids"]
    observed = encoder.pool([*sequences, []])
    relative_errors = []
    for seq, actual in zip(sequences, observed[:2], strict=True):
        ids = torch.tensor([seq], device=encoder.device)
        hidden = encoder.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               use_cache=False).last_hidden_state.float()
        expected = hidden[0].mean(0).cpu().numpy()
        relative_errors.append(float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)))
    frozen = all(not p.requires_grad for p in encoder.model.parameters())
    empty_is_zero = bool(np.all(observed[-1] == 0))
    if max(relative_errors) > 0.01 or not (frozen and empty_is_zero and not encoder.model.training):
        raise RuntimeError("真实模型的冻结状态或掩码池化检查失败")
    return {"frozen_parameters": frozen, "eval_mode": not encoder.model.training,
            "padding_test_token_lengths": list(map(len, sequences)), "empty_window_is_zero": empty_is_zero,
            "relative_l2_errors_against_unpadded_means": relative_errors, "relative_l2_tolerance": 0.01}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, default=Path("runs/main"))
    p.add_argument("--model-dir", type=Path, help="可选的 Qwen2.5-0.5B 发布权重目录")
    p.add_argument("--out", type=Path, default=Path("results/window_control.json"))
    args = p.parse_args()

    config = json.loads((args.run / "summary.json").read_text())["configuration"]
    train_pool, dev_pool = read_corpus(args.data)
    train = balanced_sample(train_pool, config["train_per_family"])
    dev = balanced_sample(dev_pool, config["dev_per_family"])
    if (id_hash(train), id_hash(dev)) != (config["train_ids_sha256"], config["dev_ids_sha256"]):
        raise ValueError("对照实验必须使用原实验的训练和验证 ID")
    weights_file = (args.model_dir or ROOT / "models/Qwen2.5-0.5B") / "model.safetensors"
    if not weights_file.is_file() and args.model_dir is None:
        from huggingface_hub import hf_hub_download
        weights_file = Path(hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION))
    with weights_file.open("rb") as stream:
        weights_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if weights_sha != "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342":
        raise ValueError("本地权重与本次实验记录的 Qwen2.5-0.5B 发布版本不一致")
    encoder = QwenEncoder(args.model_dir)
    checks = check_pooling(encoder)
    x_train, changed_train = position_features(encoder, train)
    x_dev, changed_dev = position_features(encoder, dev)
    baseline = joblib.load(args.run / "char-tfidf.joblib")
    family_x = baseline["vectorizer"].transform(text_windows([r["text"] for r in dev]))
    labels = [r["label"] for r in train]
    gold = {r["id"]: r for r in dev}
    scores, predictions = {}, {}
    for method, second in (("first32_next32", 1), ("first32_last32", 2)):
        training = qwen_vectors(x_train[:, [0, second]], "qwen-head-tail")
        testing = qwen_vectors(x_dev[:, [0, second]], "qwen-head-tail")
        model = Classifiers(fit_linear_head(training, labels), baseline["heads"].source)
        predictions[method] = model.predict(testing, gold, source_x=family_x)
        scores[method] = evaluate(gold, predictions[method])
    first, last = predictions.values()
    corrected = sum(first[i]["label"] != r["label"] == last[i]["label"] for i, r in gold.items())
    regressed = sum(first[i]["label"] == r["label"] != last[i]["label"] for i, r in gold.items())
    write_json(args.out, {
        "model": MODEL_ID, "revision": REVISION, "weights_sha256": weights_sha,
        "device": str(encoder.device), "dtype": str(encoder.dtype), "pooling_check": checks,
        "max_tokens_per_text": 2 * WIDTH, "detector_dimensions": 2 * encoder.model.config.hidden_size,
        "train_n": len(train), "dev_n": len(dev), "train_ids_sha256": id_hash(train),
        "dev_ids_sha256": id_hash(dev), "train_with_different_second_window": changed_train,
        "dev_with_different_second_window": changed_dev, "detection_corrected": corrected,
        "detection_regressed": regressed, "results": scores,
    })
    for name, score in scores.items():
        print(name, "detect", round(score["detect_macro_f1"], 4), "joint", round(score["joint_score"], 4))


if __name__ == "__main__":
    main()
