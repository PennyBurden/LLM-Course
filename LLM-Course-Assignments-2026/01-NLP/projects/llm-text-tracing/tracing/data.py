"""Read the official training release and reproduce the fixed course split."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["纯人类撰写", "纯机器生成", "人机协作"]
FAMILIES = ["OpenAI", "Alibaba", "DeepSeek", "ByteDance", "Moonshot", "Google", "Anthropic", "xAI"]


def read_rows(lines):
    rows = {}
    for line_no, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
            record_id = row["id"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"第 {line_no} 行缺少有效的 JSON 对象或 id") from exc
        if type(record_id) is not int or record_id in rows:
            raise ValueError(f"第 {line_no} 行的 id 重复或不是整数")
        rows[record_id] = row
    if not rows:
        raise ValueError("输入文件为空")
    return rows


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return read_rows(stream)


def read_release_file(source, filename):
    source = Path(source)
    if source.is_dir():
        return read_jsonl(source / filename)
    if source.suffix != ".zip":
        return read_jsonl(source)
    with ZipFile(source) as archive:
        files = [n for n in archive.namelist() if Path(n).name == filename]
        if len(files) != 1:
            raise ValueError(f"数据包需要包含一份 {filename}")
        with archive.open(files[0]) as stream:
            return read_rows(stream)


def validate_gold(row):
    label = row.get("label")
    if type(label) is not int or label not in range(3):
        raise ValueError(f"id={row['id']} 的 label 无效")
    if label == 1:
        family = row.get("family")
        if type(family) is not int or family not in range(8):
            raise ValueError(f"id={row['id']} 缺少有效的机器家族")


def validate_text(row):
    if not isinstance(row.get("text"), str) or not row["text"].strip():
        raise ValueError(f"id={row['id']} 的 text 为空或格式错误")


def read_corpus(source):
    source = Path(source)
    if source.is_dir():
        train = read_jsonl(source / "train.jsonl")
        dev = read_jsonl(source / "dev.jsonl")
        gold = read_jsonl(source / "dev_gold.jsonl")
        if dev.keys() != gold.keys():
            raise ValueError("dev 与 dev_gold 的 ID 不匹配")
        dev = {i: {**gold[i], "text": sample["text"]} for i, sample in dev.items()}
    else:
        original = read_release_file(source, "train.jsonl")
        dev_ids = {int(line) for line in (ROOT / "splits/dev_ids.txt").read_text().splitlines()}
        if len(dev_ids) != 19760 or set(original) != set(range(1, 98801)):
            raise ValueError("原始 train.jsonl 的 ID 与 CCKS 98,800 条训练样本不相符")
        train = {i: row for i, row in original.items() if i not in dev_ids}
        dev = {i: original[i] for i in dev_ids}
    if train.keys() & dev.keys():
        raise ValueError("训练和验证 ID 有交集")
    for split in (train, dev):
        for row in split.values():
            validate_text(row)
            validate_gold(row)
    return train, dev


def balanced_sample(rows, per_family):
    """Use equal detection class sizes and equal counts for each machine family."""
    if per_family <= 0:
        raise ValueError("每家族样本量必须为正数")
    groups = defaultdict(list)
    for row in rows.values():
        group = (row["label"], row.get("family") if row["label"] == 1 else -1)
        groups[group].append(row)
    result = []
    for label in range(3):
        for family in (range(8) if label == 1 else [-1]):
            eligible = groups[label, family]
            count = per_family if label == 1 else 8 * per_family
            if len(eligible) < count:
                raise ValueError(f"分层 {(label, family)} 的样本不足 {count} 条")
            # The ID is used for sampling and alignment only.
            ranked = sorted(eligible, key=lambda r: hashlib.sha256(
                f"20261119/{r['id']}".encode("ascii")).digest())
            result.extend(ranked[:count])
    return sorted(result, key=lambda row: row["id"])
