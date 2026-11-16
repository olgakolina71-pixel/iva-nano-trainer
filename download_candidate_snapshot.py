"""Download one registry-approved candidate snapshot into quarantine.

This downloads source files only. It does not parse, approve, or train on them.
Only registry entries with status="candidate" are selectable; "hold" sources are
intentionally blocked. Large corpora are kept as short-lived Actions artifacts,
not committed into Git history.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REGISTRY = Path("data/dataset_candidates.json")
OUT = Path("quarantine/huggingface")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    matches = [d for d in registry["datasets"] if d["id"] == args.dataset_id]
    if len(matches) != 1:
        raise SystemExit(f"Unknown dataset id: {args.dataset_id}")
    item = matches[0]
    if item.get("status") != "candidate":
        raise SystemExit(
            f"Blocked: {args.dataset_id} has status={item.get('status')!r}; "
            "only candidate entries can be downloaded by this workflow."
        )

    target = OUT / args.dataset_id
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=item["hf_id"],
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    metadata = {
        "id": item["id"],
        "hf_id": item["hf_id"],
        "source_url": item["url"],
        "declared_license": item.get("declared_license"),
        "provenance_claim": item.get("provenance_claim"),
        "registry_status": item["status"],
        "training_enabled": False,
        "rights_status": "dataset_card_claim_unverified",
        "downloaded_files": sorted(
            str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()
        ),
        "warning": (
            "Quarantined source snapshot only. License metadata and provenance "
            "have not been independently verified. Do not train until rights, "
            "privacy, schema, and quality reviews are complete."
        ),
    }
    (target / "iva_source_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Downloaded {item['hf_id']} into {target}")
    print(f"Files: {len(metadata['downloaded_files'])}; training_enabled=false")


if __name__ == "__main__":
    main()
