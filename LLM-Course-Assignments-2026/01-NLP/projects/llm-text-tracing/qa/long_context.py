"""Send a document exceeding 32,768 tokens through the same QA path as the demo."""
import argparse
from pathlib import Path
from time import perf_counter

from experiment import write_json
from qa.engine import answer
from qa.generator import CONTEXT_LIMIT, EvidenceGenerator
from qa.index import EvidenceIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("runs/qa-index"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/qa-long-context.json"))
    args = parser.parse_args()
    model = EvidenceGenerator(args.model_dir)
    lines = []
    while len(model.tokens("\n".join(lines))) < 33000:
        first = len(lines) + 1
        lines.extend(f"归档页 {i} 记录本周的常规校对，交接员按页码核对材料。" for i in range(first, first + 32))
    target = "秋季复核批次的档案代号是松塔427。"
    lines.insert(len(lines) // 2, target)
    lines.append("春季资料入库时使用的档案代号是白桦113。")
    notes = "\n".join(lines)
    question = "补充材料中，秋季复核批次的档案代号是什么？"
    char = notes.index(target)
    doc_tokens = len(model.tokens(notes))
    print(f"实测材料 {doc_tokens} 个 token，证据位于文中，开始完整前向计算。", flush=True)
    start = perf_counter()
    row = answer(EvidenceIndex(args.index), model, question, notes=notes)
    expected_id = f"note-{char}-{char + len(target)}"
    result = {"fixture": "synthetic numbered archive pages with a middle needle and a final distractor",
              "model": "Qwen/Qwen2.5-0.5B", "rope": "YaRN factor 2", "native_tokens": 32768,
              "application_limit": CONTEXT_LIMIT, "device": str(model.device), "backend": model.backend_for(row["prompt_tokens"]),
              "note_tokens": doc_tokens,
              "evidence_starts_after_tokens": len(model.tokens(notes[:char])), "expected_citation": expected_id,
              "expected_answer": target, "answer": row["answer"], "citation": row["citation"],
              "verified_quote": row["verified"], "prefilled_prompt_tokens": row["prompt_tokens"],
              "seconds": round(perf_counter() - start, 2),
              "passed": row["citation"] == expected_id and row["answer"] == target and row["prompt_tokens"] > 32768}
    write_json(args.out, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
