"""IVA AI Academy: collect quarantined dialogue candidates from four providers."""
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

PROVIDERS = {
    "pollinations": {
        "url": "https://text.pollinations.ai/openai",
        "model": "openai",
        "key_env": "",
    },
    "kilo": {
        "url": "https://ai-gateway.kilo.ai/api/v1/chat/completions",
        "model": "minimax/minimax-m2.7:free",
        "key_env": "KILO_API_KEY",
    },
    "llm7": {
        "url": "https://api.llm7.io/v1/chat/completions",
        "model": "default",
        "key_env": "LLM7_API_KEY",
    },
    "ovh": {
        "url": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions",
        "model": "gpt-oss-20b",
        "key_env": "OVH_AI_ENDPOINTS_ACCESS_TOKEN",
    },
}


def extract_pairs(raw, count):
    decoded = raw.decode("utf-8", errors="replace")
    try:
        result = json.loads(decoded)
    except json.JSONDecodeError:
        result = decoded

    if isinstance(result, dict) and "choices" in result:
        choices = result.get("choices") or []
        if not choices:
            raise ValueError("Provider returned no choices")
        result = choices[0].get("message", {}).get("content", "")
    elif isinstance(result, dict):
        result = result.get("text", result.get("response", result.get("content", result)))
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)

    start, end = result.find("["), result.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Provider did not return a JSON array")
    parsed = json.loads(result[start:end + 1])
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array")

    valid, seen = [], set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer or question.casefold() in seen:
            continue
        if len(question) > 2000 or len(answer) > 6000:
            continue
        seen.add(question.casefold())
        valid.append({"question": question, "answer": answer})
        if len(valid) >= count:
            break
    if not valid:
        raise ValueError("No valid question-answer pairs returned")
    return valid


def ask_provider(name, count, timeout):
    cfg = PROVIDERS[name]
    prompt = (
        f"Create exactly {count} original Russian question-answer pairs for a helpful assistant. "
        "Use varied everyday, educational, science, programming and creative topics. "
        "Keep answers factual and age-appropriate. Do not include personal data. "
        "Return ONLY a JSON array of objects with string fields question and answer."
    )
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "IVA-AI-Academy/0.3",
    }
    api_key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else ""
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "You generate clean supervised dialogue examples. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
        "max_tokens": 5000,
    }
    request = urllib.request.Request(
        cfg["url"],
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Provider response too large")
    return extract_pairs(raw, count)


def main():
    parser = argparse.ArgumentParser(
        description="Collect quarantined IVA dialogue candidates from four AI providers."
    )
    parser.add_argument(
        "--providers",
        default="pollinations,kilo,llm7,ovh",
        help="Comma-separated provider names: pollinations,kilo,llm7,ovh",
    )
    parser.add_argument("--count", type=int, default=20, help="Total target number of pairs")
    parser.add_argument(
        "--pollinations-delay-seconds",
        type=float,
        default=15.0,
        help="Wait only after the Pollinations request (default: 15 seconds)",
    )
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output", default="data/academy/candidates.txt")
    args = parser.parse_args()

    names = [part.strip().lower() for part in args.providers.split(",") if part.strip()]
    if not names or any(name not in PROVIDERS for name in names):
        parser.error("Use one or more of: pollinations,kilo,llm7,ovh")
    if len(set(names)) != len(names):
        parser.error("Provider names must not repeat")
    if not 1 <= args.count <= 100:
        parser.error("--count must be between 1 and 100")
    if not 0 <= args.pollinations_delay_seconds <= 3600:
        parser.error("--pollinations-delay-seconds must be between 0 and 3600")

    base, remainder = divmod(args.count, len(names))
    allocations = [base + (1 if i < remainder else 0) for i in range(len(names))]
    collected, successes, errors = [], [], []

    for name, allocation in zip(names, allocations):
        if allocation == 0:
            continue
        try:
            pairs = ask_provider(name, allocation, args.timeout)
            for pair in pairs:
                pair["provider"] = name
            collected.extend(pairs)
            successes.append(name)
            print(f"[OK] {name}: {len(pairs)} pairs")
        except Exception as exc:
            errors.append({"provider": name, "error": str(exc)})
            print(f"[WARN] {name} failed: {exc}")

        # The requested cooldown applies only to Pollinations, not to the other services.
        if name == "pollinations" and args.pollinations_delay_seconds:
            print(f"Pollinations cooldown: {args.pollinations_delay_seconds:g} seconds")
            time.sleep(args.pollinations_delay_seconds)

    if not collected:
        raise RuntimeError("All providers failed; no candidate data was written.")

    blocks = [
        f"### Пользователь:\n{pair['question']}\n### IVA:\n{pair['answer']}"
        for pair in collected
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    manifest = {
        "datasets": [{
            "id": "iva_ai_academy_candidates",
            "path": str(output.resolve()),
            "license": "provider_terms_review_required",
            "attribution": "AI-generated; review each provider's terms before training",
            "source_url": ";".join(PROVIDERS[name]["url"] for name in successes),
            "providers_succeeded": successes,
            "provider_errors": errors,
            "use": "pending_review",
            "content_type": "supervised_dialogue_text",
            "rights_status": "pending_review",
            "records": len(blocks),
        }]
    }
    (output.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Created {len(blocks)} candidate pairs: {output}")
    print("Status: pending_review. Nothing is automatically approved or used for training.")


if __name__ == "__main__":
    main()
