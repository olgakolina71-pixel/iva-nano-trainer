#!/usr/bin/env python3
"""Generate IVA's 20,000-dialogue Russian synthetic SFT corpus with Gemini.

The API key is read only from GEMINI_API_KEY. Progress is appended to JSONL so
the run can be resumed safely after quota limits or an interrupted session.
"""
import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL_DEFAULT = "gemini-2.5-flash"
TOTAL_DEFAULT = 20_000
TOOLS = {"calculator", "current_time", "web_search"}
CATEGORIES = [
    ("general", 7000),
    ("long_context", 3000),
    ("teaching", 2500),
    ("coding_math", 2000),
    ("tool_use", 3000),
    ("tool_recovery", 1000),
    ("reasoning", 1500),
]
SCHEMA = """Return ONLY a JSON array of dialogue objects. Each object:
{"id":"ru-000001","language":"ru","category":"general","messages":[
 {"role":"user","content":"..."},
 {"role":"assistant","content":"..."}
]}
Allowed roles: user, assistant, tool. A tool call is:
{"role":"assistant","tool_call":{"name":"calculator","arguments":{"expression":"12*8"}}}
A matching result is:
{"role":"tool","name":"calculator","content":"96"}
Allowed tool names ONLY: calculator, current_time, web_search.
For web_search arguments use {"query":"..."}; for current_time use {}; for calculator use {"expression":"..."}.
No other tools, no fake API results, no personal data, no real names/phone/email/address.
Make every dialogue distinct, natural Russian, useful, and internally consistent.
Usually 4-10 messages per dialogue; vary length and style. Avoid repetitive openings.
The assistant should be helpful, honest about uncertainty, and never claim actions it did not perform.
For non-tool dialogues do not include tool calls. Tool examples must show a sensible call, result, and explanation.
Do not put markdown fences around JSON."""

CATEGORY_GUIDANCE = {
    "general": "Everyday helpful conversation: planning, explanations, recommendations, clarification, casual but respectful chat.",
    "long_context": "A multi-turn conversation where IVA remembers constraints stated earlier and applies them consistently. Include a correction or changed requirement.",
    "teaching": "Teach a concept step by step, adapt to the user's level, check understanding, and correct a misconception kindly.",
    "coding_math": "Programming, debugging, arithmetic, or school-level mathematics. Explain reasoning clearly; code must be safe and runnable.",
    "tool_use": "Use one of IVA's actual tools only when genuinely useful. Include assistant tool_call, matching tool result, then a natural final answer.",
    "tool_recovery": "A tool returns an error, empty result, or insufficient information. IVA must acknowledge the limitation and recover safely without inventing results.",
    "reasoning": "Solve a practical logic or planning problem, state assumptions, compare options, and ask a clarification only if essential.",
}

def request_gemini(api_key, model, prompt, max_output_tokens, timeout):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.85,
            "responseMimeType": "application/json",
            "maxOutputTokens": max_output_tokens,
        },
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    if not text:
        raise ValueError("Gemini returned no text (possibly blocked or token-limited)")
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Gemini response is not a JSON array")
    return parsed

def validate(obj, allowed_ids):
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
        raise ValueError("bad dialogue structure")
    if obj.get("language") != "ru" or obj.get("category") not in CATEGORY_GUIDANCE:
        raise ValueError("bad language/category")
    if not 2 <= len(obj["messages"]) <= 24:
        raise ValueError("bad message count")
    saw_user = saw_assistant = False
    pending = None
    for m in obj["messages"]:
        if not isinstance(m, dict):
            raise ValueError("message is not an object")
        role = m.get("role")
        if role == "assistant" and "tool_call" in m:
            call = m["tool_call"]
            if not isinstance(call, dict) or call.get("name") not in TOOLS:
                raise ValueError("unknown tool")
            if not isinstance(call.get("arguments", {}), dict):
                raise ValueError("bad tool arguments")
            if pending:
                raise ValueError("nested tool call before previous result")
            pending = call["name"]
            saw_assistant = True
        elif role in ("user", "assistant"):
            if not isinstance(m.get("content"), str) or not m["content"].strip():
                raise ValueError("empty message")
            if role == "user":
                saw_user = True
            else:
                saw_assistant = True
        elif role == "tool":
            if not pending or m.get("name") != pending or not isinstance(m.get("content"), str):
                raise ValueError("unmatched tool result")
            pending = None
        else:
            raise ValueError("unsupported role")
        for value in m.values():
            if isinstance(value, str) and (re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", value)
                or re.search(r"(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)", value)):
                raise ValueError("possible personal data")
    if pending or not saw_user or not saw_assistant:
        raise ValueError("incomplete dialogue/tool cycle")
    if obj.get("id") in allowed_ids:
        raise ValueError("duplicate id")
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/sft/incoming/iva_20k_gemini.jsonl")
    ap.add_argument("--total", type=int, default=TOTAL_DEFAULT)
    ap.add_argument("--per-request", type=int, default=20)
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", MODEL_DEFAULT))
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--pause", type=float, default=2.0)
    args = ap.parse_args()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY as an environment secret. Never paste it into chat or commit it.")
    if args.total < 1 or args.per_request < 1 or args.per_request > 40:
        raise SystemExit("--total must be positive and --per-request must be 1..40")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if out.exists():
        existing = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {x.get("id") for x in existing}
    counts = {}
    for x in existing:
        counts[x.get("category")] = counts.get(x.get("category"), 0) + 1
    if len(existing) >= args.total:
        print(f"Already have {len(existing)} dialogues in {out}")
        return

    targets = dict(CATEGORIES)
    # Scale category quotas proportionally if --total differs from 20,000.
    if args.total != TOTAL_DEFAULT:
        targets = {k: round(v * args.total / TOTAL_DEFAULT) for k, v in CATEGORIES}
        targets["general"] += args.total - sum(targets.values())

    failures = 0
    with out.open("a", encoding="utf-8") as stream:
        while len(existing) < args.total:
            category = max(
                (k for k, _ in CATEGORIES if counts.get(k, 0) < targets.get(k, 0)),
                key=lambda k: targets.get(k, 0) - counts.get(k, 0),
                default="general"
            )
            need = min(args.per_request, args.total - len(existing),
                       targets.get(category, args.total) - counts.get(category, 0))
            if need <= 0:
                break
            start = len(existing) + 1
            prompt = (
                SCHEMA + "\n\nCATEGORY: " + category + "\nGUIDANCE: " + CATEGORY_GUIDANCE[category]
                + f"\nGenerate exactly {need} dialogues. Use IDs ru-{start:06d} onward."
                + "\nDo not reuse these existing IDs: " + ", ".join(sorted(ids)[-30:])
            )
            try:
                batch = request_gemini(key, args.model, prompt, args.max_output_tokens, args.timeout)
                accepted = []
                for item in batch:
                    try:
                        item = validate(item, ids)
                        if item.get("category") != category:
                            raise ValueError("wrong category")
                        ids.add(item["id"])
                        accepted.append(item)
                    except ValueError:
                        continue
                if len(accepted) != need:
                    raise ValueError(f"requested {need}, validated {len(accepted)}; retrying this category")
                for item in accepted:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                stream.flush()
                existing.extend(accepted)
                counts[category] = counts.get(category, 0) + len(accepted)
                failures = 0
                print(f"Progress: {len(existing)}/{args.total}; {category}: {counts[category]}/{targets[category]}")
                time.sleep(args.pause)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                failures += 1
                wait = min(300, 5 * (2 ** min(failures, 6)))
                print(f"Request failed ({exc}); waiting {wait}s. Progress is saved.")
                if failures >= 10:
                    raise SystemExit("Stopped after 10 consecutive failures. Rerun to resume.")
                time.sleep(wait)

    print(json.dumps({"output": str(out), "dialogues": len(existing), "category_counts": counts,
                      "model": args.model, "training_approval": "not_granted"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
