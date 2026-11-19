#!/usr/bin/env python3
"""Build a reviewed IVA SFT corpus from conversation JSONL files.

Input format: one JSON object per line:
{"id":"...","language":"ru","messages":[
 {"role":"user","content":"..."},
 {"role":"assistant","content":"..."},
 {"role":"assistant","tool_call":{"name":"calculator","arguments":{"expression":"2+2"}}},
 {"role":"tool","name":"calculator","content":"4"},
 {"role":"assistant","content":"Получается 4."}
]}
Tool calls are represented as assistant targets; tool results are inserted into the
following user-side context. Only the allow-listed IVA tools are accepted.
This script does not download data, invent examples, or approve source rights.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ALLOWED_TOOLS = {"calculator", "current_time", "web_search"}
PII_PATTERNS = [
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)"),
]
MAX_DIALOGUE_CHARS = 30000


def clean_text(value):
    if not isinstance(value, str):
        raise ValueError("message content must be a string")
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        raise ValueError("empty message")
    if len(value) > MAX_DIALOGUE_CHARS:
        raise ValueError("message too long")
    if any(pattern.search(value) for pattern in PII_PATTERNS):
        raise ValueError("possible personal data")
    return value


def validate_dialogue(obj):
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
        raise ValueError("expected object with messages array")
    if obj.get("language", "ru") != "ru":
        raise ValueError("only Russian examples are accepted in this build")
    messages = obj["messages"]
    if len(messages) < 2:
        raise ValueError("dialogue must contain at least two messages")
    rendered = []
    pending_tool = None
    saw_user = False
    saw_assistant = False
    has_tool = False

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        role = message.get("role")
        if role == "user":
            content = clean_text(message.get("content", ""))
            rendered.append(("user", content))
            saw_user = True
        elif role == "assistant" and "tool_call" in message:
            call = message["tool_call"]
            if not isinstance(call, dict) or call.get("name") not in ALLOWED_TOOLS:
                raise ValueError("unknown or malformed tool call")
            args = call.get("arguments", {})
            if not isinstance(args, dict):
                raise ValueError("tool arguments must be an object")
            # Compact stable JSON is important: the model must learn a parseable format.
            payload = json.dumps(
                {"name": call["name"], "arguments": args},
                ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            rendered.append(("assistant", "<tool_call>" + payload + "</tool_call>"))
            pending_tool = call["name"]
            saw_assistant = True
            has_tool = True
        elif role == "tool":
            name = message.get("name")
            if not pending_tool or name != pending_tool:
                raise ValueError("tool result does not match preceding tool call")
            content = clean_text(message.get("content", ""))
            rendered.append(("tool", f"<tool_result name=\"{name}\">{content}</tool_result>"))
            pending_tool = None
        elif role == "assistant":
            content = clean_text(message.get("content", ""))
            rendered.append(("assistant", content))
            saw_assistant = True
        else:
            raise ValueError(f"unsupported message role: {role!r}")

    if pending_tool:
        raise ValueError("tool call has no matching tool result")
    if not saw_user or not saw_assistant:
        raise ValueError("dialogue needs user and assistant messages")
    # Keep the conversation in the existing IVA SFT parser's two-role format.
    turns = []
    previous_role = None
    for role, content in rendered:
        if role == "tool":
            role = "user"
        if role == "user":
            turns.append(f"### Пользователь:\n{content}")
        else:
            turns.append(f"### IVA:\n{content}")
        previous_role = role
    if not any(line.startswith("### IVA:") for line in turns):
        raise ValueError("no assistant target")
    return "\n".join(turns), has_tool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/sft/incoming")
    parser.add_argument("--output", default="data/sft/iva_sft_120k.txt")
    parser.add_argument("--target-dialogues", type=int, default=120000)
    parser.add_argument("--minimum-tool-dialogues", type=int, default=20000)
    args = parser.parse_args()
    if args.target_dialogues < 1 or args.minimum_tool_dialogues < 0:
        parser.error("dialogue targets must be positive")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files found in {input_dir}")

    seen = set()
    accepted = []
    rejected = {}
    tool_count = 0
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    rendered, has_tool = validate_dialogue(obj)
                    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
                    if digest in seen:
                        raise ValueError("duplicate dialogue")
                    seen.add(digest)
                    accepted.append(rendered)
                    tool_count += int(has_tool)
                except Exception as exc:
                    reason = str(exc)[:120] or type(exc).__name__
                    rejected[reason] = rejected.get(reason, 0) + 1

    if len(accepted) < args.target_dialogues:
        raise SystemExit(
            f"STOP: accepted {len(accepted):,}/{args.target_dialogues:,} unique dialogues; "
            "no output corpus written."
        )
    if tool_count < args.minimum_tool_dialogues:
        raise SystemExit(
            f"STOP: only {tool_count:,} tool-use dialogues; "
            f"minimum is {args.minimum_tool_dialogues:,}; no output corpus written."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exact target, deterministic source ordering.
    output.write_text("\n\n".join(accepted[:args.target_dialogues]) + "\n", encoding="utf-8")
    report = {
        "output": str(output),
        "dialogues_written": args.target_dialogues,
        "tool_use_dialogues_seen": tool_count,
        "unique_dialogues_accepted": len(accepted),
        "rejected_by_reason": rejected,
        "tool_allowlist": sorted(ALLOWED_TOOLS),
        "training_approval": "not_granted_by_this_script",
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
