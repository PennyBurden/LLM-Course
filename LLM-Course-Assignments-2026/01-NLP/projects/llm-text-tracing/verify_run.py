"""Replay the command-line and Streamlit entry points and cross-check the saved course results."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiment import write_json
from tracing.data import ROOT, read_jsonl, read_release_file
from tracing.models import COMBINED


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score_by_counts(gold, predictions):
    if gold.keys() != predictions.keys():
        raise ValueError("真值与预测 ID 集合不一致")
    det_pairs, family_pairs = [], []
    for i, truth in gold.items():
        row = predictions[i]
        label, family = row["label"], row["family"]
        if type(label) is not int or label not in range(3) or type(family) is not int:
            raise ValueError(f"id={i} 的分类字段格式错误")
        if (label == 1 and family not in range(8)) or (label != 1 and family != -1):
            raise ValueError(f"id={i} 的家族字段与检测结果不符")
        det_pairs.append((truth["label"], label))
        if truth["label"] == 1:
            family_pairs.append((truth["family"], family))

    def macro(pairs, n):
        actual = Counter(a for a, _ in pairs)
        predicted = Counter(b for _, b in pairs)
        correct = Counter(a for a, b in pairs if a == b)
        f1 = [2 * correct[k] / (actual[k] + predicted[k]) if actual[k] + predicted[k] else 0
              for k in range(n)]
        return sum(f1) / n

    det, source = macro(det_pairs, 3), macro(family_pairs, 8)
    return {"detect_macro_f1": det, "source_macro_f1": source, "joint_score": 0.8 * det + 0.2 * source}


def compare_scores(independent, stored):
    for key, value in independent.items():
        if not math.isclose(value, stored[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{key} 的独立复算结果与已存记录不同")


def run_cli(model_file, *arguments):
    command = [sys.executable, str(ROOT / "predict.py"), "--model", str(model_file), *map(str, arguments)]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    return proc.stdout


def check_demo(run_dir, text, expected):
    from streamlit.testing.v1 import AppTest

    os.environ["TRACE_RUN_DIR"] = str(run_dir.resolve())
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    if app.exception or app.selectbox[0].value != COMBINED:
        raise RuntimeError("演示页面未能加载默认组合模型")
    options = list(app.selectbox[0].options)
    if len(options) != 4:
        raise ValueError("页面模型选择器未显示四种实验方案")
    app.button[0].click().run()
    if app.exception or not app.warning or app.warning[0].value != "请先输入文本。":
        raise RuntimeError("空输入提示未通过")
    app.text_area[0].input(text)
    app.button[0].click().run()
    if app.exception:
        raise RuntimeError(str(app.exception))
    actual = json.loads(app.json[0].value)
    if actual != expected or actual["label"] != 1:
        raise ValueError("页面预测与命令行输出不一致")
    return {"available_models": options, "empty_input_check": True, "single_prediction": actual}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="官方 dataset.zip 或本地固定划分目录")
    p.add_argument("--gold", type=Path, default=Path("data/test_labeled.zip"))
    p.add_argument("--reference", type=Path, default=Path("runs/main"))
    p.add_argument("--repeat", type=Path, default=Path("runs/team-verified"))
    p.add_argument("--out", type=Path, default=Path("results/reproduction.json"))
    args = p.parse_args()
    if args.reference.resolve() == args.repeat.resolve():
        raise ValueError("请分别指定首次训练和独立重训的输出目录")
    old = load_json(args.reference / "summary.json")
    new = load_json(args.repeat / "summary.json")
    published = load_json(ROOT / "results/validation.json")
    if new["results"] != old["results"] or new["results"] != published["results"]:
        raise ValueError("重训指标与归档值不符")
    fields = ("train_ids_sha256", "dev_ids_sha256", "model", "revision", "tokens_per_window")
    for k in fields:
        if new["configuration"][k] != old["configuration"][k]:
            raise ValueError(f"两次实验的 {k} 不一致")
    if args.data.is_dir():
        dev_gold = read_jsonl(args.data / "dev_gold.jsonl")
    else:
        from tracing.data import read_corpus
        _, dev_gold = read_corpus(args.data)
    validation = {}
    for method, scores in new["results"].items():
        filename = f"{method}-predictions.jsonl"
        first, repeat = read_jsonl(args.reference / filename), read_jsonl(args.repeat / filename)
        if repeat != first:
            raise ValueError(f"{method} 两次验证预测存在差异")
        gold = {i: dev_gold[i] for i in repeat}
        compare_scores(score_by_counts(gold, repeat), scores)
        validation[method] = {"matched_records": len(repeat), "joint_score": scores["joint_score"]}

    official_run = load_json(args.repeat / "test_b.json")
    if official_run != load_json(ROOT / "results/test_b.json"):
        raise ValueError("独立测试 B 的复跑记录与归档结果不符")
    inputs = read_release_file(args.data, "test_b.jsonl")
    size = official_run["sample_n"]
    ids = sorted(sorted(inputs, key=lambda i: hashlib.sha256(f"20261122/test_b/{i}".encode()).digest())[:size])
    digest = hashlib.sha256("".join(f"{i}\n" for i in ids).encode()).hexdigest()
    if digest != official_run["ids_sha256"]:
        raise ValueError("抽样 ID 与独立测试记录不一致")
    predictions = {}
    with tempfile.TemporaryDirectory() as temp:
        batch_file = Path(temp) / "inputs.jsonl"
        # Deliberately reverse ID order and pass only the two official input fields.
        with batch_file.open("w", encoding="utf-8") as stream:
            for i in reversed(ids):
                stream.write(json.dumps({"id": i, "text": inputs[i]["text"]}, ensure_ascii=False) + "\n")
        for method in ("char-tfidf", COMBINED):
            output = args.repeat / f"test_b-{method}.jsonl"
            run_cli(args.repeat / f"{method}.joblib", "--input", batch_file, "--output", output)
            predictions[method] = read_jsonl(output)
    # Ground truth is read after the input has been sampled and both outputs written.
    answers = read_release_file(args.gold, "test_b_labeled.jsonl")
    if inputs.keys() != answers.keys():
        raise ValueError("测试 B 输入和官方答案的完整 ID 集合不同")
    truth = {i: answers[i] for i in ids}
    heldout = {}
    for method, rows in predictions.items():
        independent = score_by_counts(truth, rows)
        compare_scores(independent, official_run["results"][method])
        heldout[method] = {"matched_records": len(rows), "count_based_scores": independent}

    machine_id = next(i for i in ids if predictions[COMBINED][i]["label"] == 1)
    sample_text = inputs[machine_id]["text"]
    single = json.loads(run_cli(args.repeat / f"{COMBINED}.joblib", "--text", sample_text))
    if single != {**predictions[COMBINED][machine_id], "id": 1}:
        raise ValueError("单条与批量推理输出不同")
    demo = check_demo(args.repeat, sample_text, single)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_scoring.py")
    outcome = unittest.TextTestRunner(verbosity=1).run(suite)
    if not outcome.wasSuccessful() or outcome.testsRun < 1:
        raise RuntimeError("评分边界测试失败或未运行")
    write_json(args.out, {"training_id_hash": new["configuration"]["train_ids_sha256"],
                          "dev_id_hash": new["configuration"]["dev_ids_sha256"],
                          "validation": validation, "test_b_id_hash": digest, "test_b": heldout,
                          "test_b_single_demo_id": machine_id, "web_demo": demo,
                          "scoring_tests_passed": outcome.testsRun})
    print(f"已复核四种验证预测、错序批量测试、单条推理与网页演示，记录在 {args.out}")


if __name__ == "__main__":
    main()
