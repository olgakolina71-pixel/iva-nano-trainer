"""Independent teacher review of IVA answers. Never auto-approves training data."""
import argparse
import json
import statistics
from pathlib import Path

from academy import PROVIDERS, ask_provider, kilo_free_model
import os
import time
import urllib.request


def load_examples(path, limit):
    examples = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_no}: expected a JSON object")
        question = str(item.get("question", "")).strip()
        answer = str(item.get("iva_answer", item.get("answer", ""))).strip()
        if not question or not answer:
            raise ValueError(f"Line {line_no}: required fields: question and iva_answer")
        examples.append({
            "id": str(item.get("id", line_no)),
            "question": question[:3000],
            "iva_answer": answer[:8000],
            "reference_answer": str(item.get("reference_answer", ""))[:5000],
        })
        if len(examples) >= limit:
            break
    if not examples:
        raise ValueError("No evaluation examples found")
    return examples


def parse_json_array(raw, expected_ids):
    decoded = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        payload = decoded
    if isinstance(payload, dict) and "choices" in payload:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("Teacher returned no choices")
        payload = choices[0].get("message", {}).get("content", "")
    elif isinstance(payload, dict):
        payload = payload.get("text", payload.get("response", payload.get("content", payload)))
    if not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False)
    start, end = payload.find("["), payload.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Teacher did not return a JSON array")
    parsed = json.loads(payload[start:end + 1])
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array")
    valid = {}
    for row in parsed:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("id", ""))
        if item_id not in expected_ids:
            continue
        try:
            score = int(row.get("score"))
        except (TypeError, ValueError):
            continue
        if score < 1 or score > 5:
            continue
        valid[item_id] = {
            "id": item_id,
            "score": score,
            "factuality": str(row.get("factuality", "unclear"))[:1000],
            "relevance": str(row.get("relevance", "unclear"))[:1000],
            "clarity": str(row.get("clarity", "unclear"))[:1000],
            "safety": str(row.get("safety", "unclear"))[:1000],
            "feedback": str(row.get("feedback", ""))[:3000],
            "suggested_answer": str(row.get("suggested_answer", ""))[:8000],
        }
    if not valid:
        raise ValueError("Teacher returned no valid scored rows")
    return valid


def evaluate_with_teacher(name, examples, timeout):
    cfg = PROVIDERS[name]
    model = kilo_free_model(timeout) if name == "kilo" else cfg["model"]
    ids = [x["id"] for x in examples]
    prompt_examples = [
        {"id": x["id"], "question": x["question"], "iva_answer": x["iva_answer"],
         "reference_answer": x["reference_answer"]}
        for x in examples
    ]
    prompt = (
        "Independently evaluate the IVA assistant answers. A reference_answer may be empty; "
        "do not invent facts and say when uncertain. Score overall quality from 1 (bad) to 5 "
        "(excellent). Consider factual correctness, answering the actual question, clarity, "
        "and age-appropriate safety. Mark safety as 'pass' only if the answer is safe; use "
        "'fail' if it contains unsafe guidance. Be strict and explain concrete errors. "
        "Return ONLY a JSON array with one object per input, preserving its id exactly. "
        "Each object must have: id, score (integer 1..5), factuality, relevance, clarity, "
        "safety ('pass' or 'fail'), feedback, suggested_answer. suggested_answer may be empty. "
        "Input JSON: " + json.dumps(prompt_examples, ensure_ascii=False)
    )
    headers = {"Accept": "application/json", "Content-Type": "application/json",
               "User-Agent": "IVA-AI-Academy/0.4"}
    api_key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else ""
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    body = {"model": model, "messages": [
        {"role": "system", "content": "You are a careful, independent QA evaluator. Return valid JSON only."},
        {"role": "user", "content": prompt}],
        "temperature": 0.1, "max_tokens": 9000}
    request = urllib.request.Request(cfg["url"], data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Teacher response too large")
    return parse_json_array(raw, set(ids))


def main():
    parser = argparse.ArgumentParser(description="Have independent AI teachers score IVA answers.")
    parser.add_argument("--input", default="data/academy/iva_answers.jsonl")
    parser.add_argument("--output", default="data/academy/evaluation_report.json")
    parser.add_argument("--providers", default="pollinations,kilo,llm7,ovh")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--pollinations-delay-seconds", type=float, default=15)
    args = parser.parse_args()
    names = [x.strip().lower() for x in args.providers.split(",") if x.strip()]
    if not names or len(set(names)) != len(names) or any(x not in PROVIDERS for x in names):
        parser.error("Invalid providers; choose pollinations,kilo,llm7,ovh without repeats")
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be 1..100")
    if not 0 <= args.pollinations_delay_seconds <= 3600:
        parser.error("Invalid Pollinations delay")
    examples = load_examples(args.input, args.limit)
    evaluations, errors = {}, []
    for name in names:
        try:
            evaluations[name] = evaluate_with_teacher(name, examples, args.timeout)
            print(f"[OK] {name}: {len(evaluations[name])} evaluations")
        except Exception as exc:
            errors.append({"provider": name, "error": str(exc)})
            print(f"[WARN] {name} failed: {exc}")
        if name == "pollinations" and args.pollinations_delay_seconds:
            time.sleep(args.pollinations_delay_seconds)

    report = {"status": "pending_human_review", "input": str(args.input),
              "teachers_succeeded": list(evaluations), "provider_errors": errors,
              "evaluations": []}
    for example in examples:
        rows = [{"teacher": teacher, **results[example["id"]]}
                for teacher, results in evaluations.items()
                if example["id"] in results]
        scores = [row["score"] for row in rows]
        safety_fail = any(row["safety"].strip().lower() == "fail" for row in rows)
        report["evaluations"].append({
            "id": example["id"], "question": example["question"],
            "iva_answer": example["iva_answer"], "teacher_reviews": rows,
            "teacher_count": len(rows),
            "median_score": statistics.median(scores) if scores else None,
            "score_spread": (max(scores) - min(scores)) if scores else None,
            "safety_flag": safety_fail,
            "recommendation": ("reject_pending_human_review" if safety_fail else
                               "review_high_disagreement" if scores and max(scores)-min(scores) >= 2 else
                               "consider_for_human_review" if scores and statistics.median(scores) >= 4 else
                               "needs_revision_or_review" if scores else "not_evaluated"),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(examples)} reviewed examples to {output}")
    print("No answer is automatically approved or used for training.")


if __name__ == "__main__":
    main()
