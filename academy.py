"""IVA AI Academy candidate generator with configurable endpoint fallback."""
import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate quarantined IVA SFT candidates.")
    parser.add_argument("--mode", choices=["get", "openai"], required=True)
    parser.add_argument("--endpoint", action="append", required=True,
                        help="HTTPS endpoint; repeat this option to add fallback servers")
    parser.add_argument("--model", default="default")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", default="data/academy/candidates.txt")
    parser.add_argument("--api-key-env", default="IVA_TEACHER_API_KEY")
    args = parser.parse_args()

    if any(not endpoint.startswith("https://") for endpoint in args.endpoint):
        parser.error("Only HTTPS endpoints are allowed")
    if not 1 <= args.count <= 100:
        parser.error("--count must be between 1 and 100")

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

    pairs = None
    errors = []
    for endpoint in args.endpoint:
        try:
            if args.mode == "get":
                if "{prompt}" in endpoint:
                    url = endpoint.replace("{prompt}", urllib.parse.quote(prompt, safe=""))
                    if args.model != "default":
                        separator = "&" if "?" in url else "?"
                        url += separator + urllib.parse.urlencode({"model": args.model})
                else:
                    parts = urllib.parse.urlsplit(endpoint)
                    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
                    query.append(("prompt", prompt))
                    if args.model != "default":
                        query.append(("model", args.model))
                    url = urllib.parse.urlunsplit((
                        parts.scheme, parts.netloc, parts.path,
                        urllib.parse.urlencode(query), parts.fragment
                    ))
                request = urllib.request.Request(url, headers=headers)
            else:
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}]
                }
                post_headers = dict(headers)
                post_headers["Content-Type"] = "application/json"
                request = urllib.request.Request(
                    endpoint, data=json.dumps(payload).encode("utf-8"),
                    headers=post_headers, method="POST"
                )

            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("Provider response too large")
            decoded = raw.decode("utf-8", errors="replace")
            try:
                result = json.loads(decoded)
            except json.JSONDecodeError:
                result = decoded
            if isinstance(result, dict) and "choices" in result:
                result = result["choices"][0]["message"]["content"]
            elif isinstance(result, dict):
                result = result.get("text", result.get("response", result.get("content", result)))
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            start, end = result.find("["), result.rfind("]")
            if start < 0 or end < start:
                raise ValueError("Endpoint did not return a JSON array")
            parsed = json.loads(result[start:end + 1])
            if not isinstance(parsed, list):
                raise ValueError("Expected a JSON array")
            valid = []
            seen = set()
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
                if len(valid) >= args.count:
                    break
            if not valid:
                raise ValueError("No valid question-answer pairs returned")
            pairs = valid
            break
        except Exception as exc:
            errors.append(endpoint + ": " + str(exc))

    if pairs is None:
        raise RuntimeError("All AI endpoints failed: " + " | ".join(errors))

    blocks = [
        "### Пользователь:\\n" + pair["question"] +
        "\\n### IVA:\\n" + pair["answer"]
        for pair in pairs
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\\n\\n".join(blocks) + "\\n", encoding="utf-8")
    manifest = {
        "datasets": [{
            "id": "iva_ai_academy_candidates",
            "path": str(output.resolve()),
            "license": "provider_terms_review_required",
            "attribution": "AI-generated; review provider terms before training",
            "source_url": ";".join(args.endpoint),
            "use": "pending_review",
            "content_type": "supervised_dialogue_text",
            "rights_status": "pending_review",
            "records": len(blocks)
        }]
    }
    (output.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    print("Created", len(blocks), "candidate pairs:", output)
    print("Quarantined: review content quality and provider terms before training.")


if __name__ == "__main__":
    main()
