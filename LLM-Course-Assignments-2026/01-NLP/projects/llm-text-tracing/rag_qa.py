"""Build the local evidence database or ask a Chinese domain question."""
import argparse
import json
from pathlib import Path

from qa.engine import answer
from qa.index import EvidenceIndex, build

DEFAULT_INDEX = Path("runs/qa-index")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build", help="清洗训练样本、分块并写入向量索引")
    make.add_argument("--data", type=Path, required=True)
    make.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ask = commands.add_parser("ask", help="在索引中检索并引用证据回答")
    ask.add_argument("question")
    ask.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ask.add_argument("--model-dir", type=Path)
    ask.add_argument("--notes", type=Path, help="UTF-8 补充材料，全文加入模型上下文")
    ask.add_argument("--history", type=Path, help="此前 ask 输出组成的 JSON 数组")
    args = parser.parse_args()
    if args.command == "build":
        output = build(args.data, args.index)
    else:
        from qa.generator import EvidenceGenerator
        notes = args.notes.read_text(encoding="utf-8") if args.notes else ""
        history = json.loads(args.history.read_text(encoding="utf-8")) if args.history else []
        output = answer(EvidenceIndex(args.index), EvidenceGenerator(args.model_dir),
                        args.question, history, notes)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
