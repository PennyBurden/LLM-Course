"""Evaluate the same constrained Qwen generator with two retrieval policies."""
import argparse
import hashlib
import json
from pathlib import Path

from experiment import write_json
from qa.engine import answer
from qa.index import EvidenceIndex
from tracing.data import ROOT


def measure(turns):
    n = len(turns)
    answerable = [t for t in turns if t["expected"]]
    answered = [t for t in turns if t["citation"]]
    correct_citations = sum(t["citation"] == t["expected"] for t in answered)
    # One gold citation per answerable turn and one returned citation per answer.
    unsupported = sum(not t["verified"] or t["citation"] != t["expected"] for t in answered)
    return {
        "turns": n, "answerable": len(answerable), "answered": len(answered),
        "correct_answers": sum(t["correct"] for t in turns),
        "answer_accuracy": sum(t["correct"] for t in turns) / n,
        "correct_citations": correct_citations,
        "citation_micro_f1": 2 * correct_citations / (len(answered) + len(answerable)),
        "unsupported_answers": unsupported, "unsupported_per_answer": unsupported / len(answered) if answered else None,
        "verified_quotes": sum(t["verified"] for t in answered),
        "correct_refusals": sum(not t["expected"] and not t["citation"] for t in turns),
        "recall_at_k": {str(k): sum(t["expected"] in t["retrieved"][:k] for t in answerable) / len(answerable)
                        for k in (1, 2, 3, 4, 5)},
    }


def main():
    from qa.generator import EvidenceGenerator
    from tracing.qwen import MODEL_ID, REVISION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("runs/qa-index"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/qa-evaluation.json"))
    args = parser.parse_args()
    evidence = EvidenceIndex(args.index)
    generator = EvidenceGenerator(args.model_dir)
    benchmark = (ROOT / "qa/benchmark.json").read_bytes()
    dialogues = json.loads(benchmark)
    output = {"model": MODEL_ID, "revision": REVISION, "device": str(generator.device),
              "index": json.loads((args.index / "manifest.json").read_text()),
              "benchmark_sha256": hashlib.sha256(benchmark).hexdigest(), "dialogues": len(dialogues), "methods": {}}
    for name, improved in [("single_turn_tfidf", False), ("scoped_evidence", True)]:
        turns = []
        for d, dialogue in enumerate(dialogues, 1):
            history = []
            for i, (question, expected, required) in enumerate(dialogue, 1):
                result = answer(evidence, generator, question, history, improved=improved)
                history.append(result)
                correct = (bool(result["citation"]) and all(s in result["answer"] for s in required)
                           if expected else not result["citation"])
                turns.append({"turn": f"{d}.{i}", "question": question, "expected": expected,
                              "required": required, "correct": correct,
                              **{k: result[k] for k in ("answer", "citation", "verified", "retrieved", "prompt_tokens")}})
        metrics = measure(turns)
        output["methods"][name] = {"metrics": metrics, "turns": turns}
        print(name, metrics, flush=True)
    write_json(args.out, output)


if __name__ == "__main__":
    main()
