"""Pair the full-index and fact-only searches on the same warm index and verify ties."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa.index import EvidenceIndex


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True, help="基线项目根目录")
    p.add_argument("--index", type=Path, default=ROOT / "runs/qa-index")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("previous_index", args.previous / "qa/index.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    new = EvidenceIndex(args.index)
    # Identical fitted artifacts and SQLite rows. The old method only uses the shared fields.
    old = previous.EvidenceIndex.__new__(previous.EvidenceIndex)
    old.__dict__.update(vars(new))
    inputs = [(q, None) for d in json.loads((ROOT / "qa/benchmark.json").read_text()) for q, _, _ in d]
    inputs += [("联合得分", scope) for scope in ("A", "B", "DEV")]
    times = {"previous": [], "current": []}
    for i, (q, scope) in enumerate(inputs * 3):
        outputs = {}
        for name, index in ([("previous", old), ("current", new)] if i % 2 == 0 else
                            [("current", new), ("previous", old)]):
            start = perf_counter_ns()
            hits = index.ranked(q, True, kind="fact", scope=scope)
            elapsed = (perf_counter_ns() - start) / 1e6
            outputs[name] = [(h["id"], h["score"], h["body"]) for h in hits]
            if i >= len(inputs):
                times[name].append(elapsed)
        if outputs["previous"] != outputs["current"]:
            raise ValueError(f"不同的候选、顺序或分数：{q}")
    result = {"distinct_queries_and_scopes": len(inputs), "timed_pairs": len(times["current"]),
              "all_rankings_identical": True, "rows_before": len(new.parts), "rows_after": len(new.fact_rows),
              "milliseconds": {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
                               for k, v in times.items()}, "paired_ms": times}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(result["milliseconds"])


if __name__ == "__main__":
    main()
