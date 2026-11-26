"""Move or remove a known record and replay each archive through the deployed QA path."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from experiment import write_json
from qa.engine import REFUSAL, answer
from qa.index import EvidenceIndex


POSITIONS = {"front": 0.10, "middle": 0.50, "end": 0.90, "absent": None}
BUDGETS = (4096, 33000)
CASES = {
    "archive": {
        "question": "补充材料中，秋季复核批次的档案代号是什么？",
        "target": "秋季复核批次的档案代号是松塔427。",
        "distractors": ["春季复核批次的档案代号是白桦113。", "秋季初审批次的档案代号是枫叶205。"],
    },
    "instrument": {
        "question": "补充材料中，西库温控仪的校准编号是什么？",
        "target": "西库温控仪的校准编号是云杉806。",
        "distractors": ["东库温控仪的校准编号是青石214。", "西库照度计的校准编号是银杏539。"],
    },
}
FILLER = (
    "装订前已复查页脚和目录顺序，交接员在清单上签收。",
    "值班人员查看封条，记录当天交接和归还情况。",
    "资料室完成一次例行盘点，纸本记录按页码排列。",
    "本页只登记装订与移交事项，下一页继续记录。",
)


def archive_pages(model, budget):
    pages, count = [], 0
    while count < budget:
        i = len(pages)
        line = f"第 {i + 1} 页　{FILLER[i % len(FILLER)]}"
        pages.append(line)
        count += len(model.tokens(line + "\n"))
    return pages


def material(pages, case, position):
    insertions = {int(len(pages) * fraction): line
                  for fraction, line in zip((0.30, 0.70), case["distractors"], strict=True)}
    if position is not None:
        insertions[int(len(pages) * position)] = case["target"]
    lines = []
    for index, page in enumerate(pages):
        if index in insertions:
            lines.append(insertions[index])
        lines.append(page)
    return "\n".join(lines)


def run(args):
    from qa.generator import EvidenceGenerator
    from tracing.qwen import MODEL_ID, REVISION

    index = EvidenceIndex(args.index)
    model = EvidenceGenerator(args.model_dir)
    rows = []
    for budget in BUDGETS:
        pages = archive_pages(model, budget)
        for case_id, case in CASES.items():
            for position, fraction in POSITIONS.items():
                notes = material(pages, case, fraction)
                note_tokens = len(model.tokens(notes))
                char = notes.find(case["target"])
                assert (char >= 0) == (fraction is not None)
                expected = f"note-{char}-{char + len(case['target'])}" if char >= 0 else ""
                start = perf_counter()
                turn = answer(index, model, case["question"], notes=notes)
                seconds = round(perf_counter() - start, 2)
                if expected:
                    correct = turn["verified"] and turn["citation"] == expected and turn["answer"] == case["target"]
                else:
                    correct = not turn["citation"] and turn["answer"] == REFUSAL
                if budget > 32768 and (note_tokens <= 32768 or (turn["citation"] and turn["prompt_tokens"] <= 32768)):
                    raise RuntimeError("长文条件未达到 32,768 token")
                rows.append({
                    "case": case_id, "budget": budget, "position": position, "note_tokens": note_tokens,
                    "notes_sha256": hashlib.sha256(notes.encode()).hexdigest(),
                    "target_start_tokens": len(model.tokens(notes[:char])) if char >= 0 else None,
                    "expected_citation": expected, "expected_answer": case["target"] if expected else REFUSAL,
                    **{key: turn[key] for key in ("answer", "citation", "verified", "retrieved", "prompt_tokens")},
                    "backend": model.backend_for(turn["prompt_tokens"]) if turn["prompt_tokens"] else None,
                    "correct": correct, "seconds": seconds,
                })
                print(budget, case_id, position, correct, turn["answer"], flush=True)
    write_json(args.out, {
        "model": MODEL_ID, "revision": REVISION, "device": str(model.device), "cases": CASES,
        "positions": POSITIONS, "index": json.loads((args.index / "manifest.json").read_text()),
        "trials": rows,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("runs/qa-index"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/context_sweep.json"))
    run(parser.parse_args())
