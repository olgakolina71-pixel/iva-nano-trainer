"""IVA AI Academy candidate generator."""
import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["get", "openai"], required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", default="default")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--output", default="data/academy/candidates.txt")
    p.add_argument("--api-key-env", default="IVA_TEACHER_API_KEY")
    args = p.parse_args()
    if not args.endpoint.startswith("https://"):
        p.error("Only HTTPS endpoints are allowed")
    if not 1 <= args.count <= 100:
        p.error("--count must be between 1 and 100")

    prompt = (
        "Create exactly " + str(args.count) +
        " original Russian question-answer pairs for a helpful assistant. "
        "Use safe topics, avoid personal data, return only a JSON array "
        "with question and answer fields."
    )
    key = os.environ.get(args.api_key_env, "")
    headers = {"Accept": "application/json, text/plain"}
    if key:
        headers["Authorization"] = "Bearer " + key

    if args.mode == "get":
        parts = urllib.parse.urlsplit(args.endpoint)
        query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        query.append(("prompt", prompt))
        if args.model != "default":
            query.append(("model", args.model))
        url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path,
                                       urllib.parse.urlencode(query), parts.fragment))
        request = urllib.request.Request(url, headers=headers)
    else:
        payload = {"model": args.model, "messages": [{"role": "user", "content": prompt}]}
        headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            args.endpoint, data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST"
        )

    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Provider response too large")
    result = json.loads(raw.decode("utf-8"))
    if isinstance(result, dict) and "choices" in result:
        result = result["choices"][0]["message"]["content"]
    elif isinstance(result, dict):
        result = result.get("text", result.get("response", result.get("content", result)))
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)
    start, end = result.find("["), result.rfind("]")
    pairs = json.loads(result[start:end + 1])
    blocks = []
    for pair in pairs[:args.count]:
        question = str(pair.get("question", "")).strip()
        answer = str(pair.get("answer", "")).strip()
        if question and answer:
            blocks.append("### Пользователь:\n" + question + "\n### IVA:\n" + answer)
    if not blocks:
        raise ValueError("No valid question-answer pairs returned")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    manifest = {
        "datasets": [{
            "id": "iva_ai_academy_candidates",
            "path": str(output.resolve()),
            "license": "provider_terms_review_required",
            "attribution": "AI-generated; review provider terms before training",
            "source_url": args.endpoint,
            "use": "pending_review",
            "content_type": "supervised_dialogue_text",
            "rights_status": "pending_review",
            "records": len(blocks)
        }]
    }
    (output.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Created", len(blocks), "candidate pairs:", output)
    print("Quarantined: review quality and provider terms before training.")


if __name__ == "__main__":
    main()
