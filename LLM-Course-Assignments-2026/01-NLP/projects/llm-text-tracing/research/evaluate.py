"""Run an archived or current project against fixed Chinese QA turns with the original scorer."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("project", "questions", "index", "model-dir", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project.resolve()))
    from qa.engine import answer
    from qa.generator import EvidenceGenerator
    from qa.index import EvidenceIndex
    from rag_eval import measure

    index = EvidenceIndex(args.index)
    model = EvidenceGenerator(args.model_dir)
    raw = args.questions.read_bytes()
    turns = []
    for d, dialogue in enumerate(json.loads(raw), 1):
        history = []
        for i, (q, expected, terms) in enumerate(dialogue, 1):
            row = answer(index, model, q, history)
            history.append(row)
            correct = (bool(row["citation"]) and all(x in row["answer"] for x in terms)
                       if expected else not row["citation"])
            turns.append({"turn": f"{d}.{i}", "question": q, "expected": expected, "required": terms,
                          "correct": correct, **{k: row[k] for k in
                          ("answer", "citation", "verified", "retrieved", "prompt_tokens")}})
    result = {"input_sha256": hashlib.sha256(raw).hexdigest(), "metrics": measure(turns), "turns": turns}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["metrics"], flush=True)


if __name__ == "__main__":
    main()
