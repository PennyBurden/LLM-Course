"""Joint CCKS score. Missing the machine class also costs source recall."""
from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .data import validate_gold


def evaluate(gold, predictions):
    if not gold or gold.keys() != predictions.keys():
        raise ValueError("评分要求真值与预测的 ID 完全匹配")
    det_true, det_pred, src_true, src_pred = [], [], [], []
    methods = {}
    missed = 0
    for i, truth in gold.items():
        validate_gold(truth)
        prediction = predictions[i]
        validate_gold(prediction)
        label, family = prediction["label"], prediction.get("family")
        if label != 1 and (type(family) is not int or family != -1):
            raise ValueError(f"id={i} 的非机器预测必须使用 family=-1")
        det_true.append(truth["label"])
        det_pred.append(label)
        if truth["label"] == 1:
            src_true.append(truth["family"])
            src_pred.append(family)
            missed += int(label != 1)
        if truth["label"] == 2 and truth.get("method"):
            counts = methods.setdefault(truth["method"], Counter())
            counts["total"] += 1
            counts["to_human"] += int(label == 0)
            counts["to_machine"] += int(label == 1)
    det_f1 = f1_score(det_true, det_pred, labels=list(range(3)), average=None, zero_division=0)
    src_f1 = f1_score(src_true, src_pred, labels=list(range(8)), average=None, zero_division=0)
    det, src = float(np.mean(det_f1)), float(np.mean(src_f1))
    return {
        "n": len(gold), "n_machine": len(src_true), "machine_missed": missed,
        "detect_macro_f1": det, "source_macro_f1": src, "joint_score": 0.8 * det + 0.2 * src,
        "class_f1": det_f1.tolist(), "family_f1": src_f1.tolist(),
        "detect_confusion": confusion_matrix(det_true, det_pred, labels=list(range(3))).tolist(),
        "collaboration_errors": methods,
    }
