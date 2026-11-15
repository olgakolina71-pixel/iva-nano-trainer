"""Prepare a reviewed dialogue dataset for IVA supervised fine-tuning.

The source manifest must explicitly mark the chosen entry as training and
rights_status=reviewed_approved. This script never upgrades rights status itself.
"""
import argparse
import json
import shutil
from pathlib import Path


def prepare(manifest_path, dataset_id, source_root, output_dir):
    manifest_path = Path(manifest_path)
    source_root = Path(source_root).resolve()
    output_dir = Path(output_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [item for item in payload.get("datasets", []) if item.get("id") == dataset_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one dataset entry for {dataset_id!r}")
    entry = matches[0]
    if entry.get("use") != "training":
        raise ValueError("Dataset is not approved for training; manifest use must be 'training'")
    if entry.get("rights_status") != "reviewed_approved":
        raise ValueError("Rights review is not recorded; set rights_status to 'reviewed_approved' only after review")
    if not entry.get("license") or not entry.get("attribution"):
        raise ValueError("Dataset license and attribution are required")

    source = Path(entry["path"]).resolve()
    if source_root not in source.parents:
        raise ValueError(f"Source must be inside {source_root}")
    if not source.is_file():
        raise FileNotFoundError(source)

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{dataset_id}.txt"
    shutil.copyfile(source, target)
    output_manifest = {
        "datasets": [{
            "id": dataset_id,
            "path": str(target.resolve()),
            "license": entry["license"],
            "attribution": entry["attribution"],
            "source_url": entry.get("source_url", ""),
            "use": "training",
            "content_type": "supervised_dialogue_text",
            "rights_status": "reviewed_approved",
        }]
    }
    manifest_out = output_dir / "manifest.json"
    manifest_out.write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"text": str(target), "manifest": str(manifest_out), "bytes": target.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description="Prepare a rights-reviewed dialogue source for IVA SFT.")
    parser.add_argument("--manifest", default="data/hf_manifest.json")
    parser.add_argument("--dataset-id", default="small_life_dialogues_cc0")
    parser.add_argument("--source-root", default="data/raw/hf")
    parser.add_argument("--output-dir", default="data/sft")
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.dataset_id, args.source_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
