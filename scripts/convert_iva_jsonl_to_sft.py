#!/usr/bin/env python3
"""Convert IVA synthetic JSONL dialogues into the repository's assistant-only SFT text format."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/sft/batch02")
    parser.add_argument("--output", default="data/raw/approved/iva_synthetic_batch02.txt")
    parser.add_argument("--expected-dialogues", type=int, default=20000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("part*.jsonl"))
    if len(files) != 20:
        raise SystemExit(f"Expected 20 JSONL parts, found {len(files)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    pairs = 0
    with output.open("w", encoding="utf-8") as dst:
        for path in files:
            with path.open(encoding="utf-8") as src:
                for line_no, line in enumerate(src, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
                    messages = record.get("messages")
                    if not isinstance(messages, list) or not messages:
                        raise SystemExit(f"{path}:{line_no}: missing messages")
                    count += 1
                    pending_user = None
                    blocks = []
                    for message in messages:
                        role = message.get("role")
                        content = str(message.get("content", "")).strip()
                        if not content:
                            continue
                        if role == "user":
                            if pending_user is not None:
                                raise SystemExit(f"{path}:{line_no}: user message without assistant reply")
                            pending_user = content
                        elif role == "tool":
                            if pending_user is not None:
                                raise SystemExit(f"{path}:{line_no}: tool result before assistant tool call")
                            tool_name = str(message.get("name", "tool"))
                            pending_user = f"[Результат инструмента {tool_name}]\\n{content}"
                        elif role == "assistant":
                            if pending_user is None:
                                raise SystemExit(f"{path}:{line_no}: assistant message without user/tool context")
                            blocks.append(f"### Пользователь:\\n{pending_user}\\n### IVA:\\n{content}")
                            pending_user = None
                        else:
                            raise SystemExit(f"{path}:{line_no}: unsupported role {role!r}")
                    if pending_user is not None:
                        raise SystemExit(f"{path}:{line_no}: final user/tool message has no assistant response")
                    if not blocks:
                        raise SystemExit(f"{path}:{line_no}: no usable user-assistant pairs")
                    pairs += len(blocks)
                    dst.write("\\n\\n".join(blocks) + "\\n\\n")

    if count != args.expected_dialogues:
        raise SystemExit(f"Expected {args.expected_dialogues} dialogues, converted {count}")
    if pairs < count:
        raise SystemExit("Unexpectedly low assistant-pair count")
    print(json.dumps({"dialogues": count, "assistant_pairs": pairs, "output": str(output), "bytes": output.stat().st_size}, ensure_ascii=False))

if __name__ == "__main__":
    main()
