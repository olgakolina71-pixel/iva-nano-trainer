"""Download approved IVA dataset snapshots from the Cloudflare Worker proxy.

This preserves source CSV files and metadata. It deliberately does not convert
auxiliary CSVs into general pretraining text; schemas must be reviewed first.
"""
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_WORKER_URL = "https://iva-ai-studio.olgakolina71.workers.dev"
EXPECTED_LICENSE = "CC BY-NC-SA 4.0"
ALLOWED_DATASETS = {
    "assoc_safe",
    "kartaslovsent",
    "open_semantics_hierarchy",
    "orfo_letter_matrix",
}
MAX_DATASET_BYTES = 100 * 1024 * 1024


class ImportErrorSafe(RuntimeError):
    pass


def _request(url: str, timeout: int = 60):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "IVA-Nano-Trainer/0.1", "Accept": "application/json,text/csv"},
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise ImportErrorSafe(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ImportErrorSafe(f"Could not reach {url}: {exc.reason}") from exc


def fetch_catalog(worker_url: str):
    with _request(worker_url.rstrip("/") + "/api/datasets") as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload.get("datasets"), list):
        raise ImportErrorSafe("Worker returned an invalid dataset catalog")
    return payload["datasets"]


def _validate_csv(raw: bytes, dataset_id: str):
    if not raw:
        raise ImportErrorSafe(f"{dataset_id}: empty CSV")
    if b"\x00" in raw[:4096]:
        raise ImportErrorSafe(f"{dataset_id}: binary data returned instead of CSV")
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ImportErrorSafe(f"{dataset_id}: source is not valid UTF-8") from exc
    reader = csv.reader(io.StringIO(decoded))
    try:
        first = next(reader)
    except StopIteration:
        raise ImportErrorSafe(f"{dataset_id}: CSV has no header/records")
    if not first or len(first) > 500:
        raise ImportErrorSafe(f"{dataset_id}: unexpected CSV structure")
    return sum(1 for _ in reader) + 1


def import_datasets(worker_url: str, output_dir: str, selected=None):
    worker_url = worker_url.rstrip("/")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    catalog = fetch_catalog(worker_url)
    selected_ids = set(selected or ALLOWED_DATASETS)
    unknown = selected_ids - ALLOWED_DATASETS
    if unknown:
        raise ImportErrorSafe("Dataset IDs are not allowlisted: " + ", ".join(sorted(unknown)))

    entries = {item.get("dataset_id"): item for item in catalog}
    imported = []
    for dataset_id in sorted(selected_ids):
        item = entries.get(dataset_id)
        if not item:
            raise ImportErrorSafe(f"{dataset_id}: missing from Worker catalog")
        if item.get("usage_status") != "approved_noncommercial_only":
            raise ImportErrorSafe(f"{dataset_id}: usage status does not permit import")
        if item.get("license") != EXPECTED_LICENSE:
            raise ImportErrorSafe(f"{dataset_id}: unexpected or missing license")
        if not item.get("attribution"):
            raise ImportErrorSafe(f"{dataset_id}: attribution is missing")
        if item.get("downloadable") is not True:
            raise ImportErrorSafe(f"{dataset_id}: Worker does not mark this dataset downloadable")

        url = f"{worker_url}/api/datasets/{dataset_id}/raw"
        with _request(url, timeout=120) as response:
            license_header = response.headers.get("x-dataset-license", "")
            attribution_header = response.headers.get("x-dataset-attribution", "")
            if license_header != EXPECTED_LICENSE or attribution_header != item["attribution"]:
                raise ImportErrorSafe(f"{dataset_id}: license/attribution headers do not match catalog")
            chunks = []
            size = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_DATASET_BYTES:
                    raise ImportErrorSafe(f"{dataset_id}: exceeds {MAX_DATASET_BYTES} byte safety limit")
                chunks.append(chunk)
        raw = b"".join(chunks)
        rows = _validate_csv(raw, dataset_id)
        digest = hashlib.sha256(raw).hexdigest()
        target = output / f"{dataset_id}.csv"
        fd, temp_name = tempfile.mkstemp(prefix=dataset_id + ".", suffix=".tmp", dir=output)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

        imported.append({
            "dataset_id": dataset_id,
            "path": str(target.as_posix()),
            "source_url": item.get("source_url"),
            "license": EXPECTED_LICENSE,
            "usage_status": item["usage_status"],
            "attribution": item["attribution"],
            "bytes": size,
            "rows_including_header": rows,
            "sha256": digest,
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "training_role": "auxiliary_only_pending_schema_review",
        })

    manifest = {
        "worker_url": worker_url,
        "license_notice": "CC BY-NC-SA 4.0 is non-commercial and share-alike; check all terms before redistribution or use.",
        "datasets": imported,
    }
    manifest_path = output / "import_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import approved IVA datasets through the Cloudflare Worker.")
    parser.add_argument("--worker-url", default=os.environ.get("IVA_WORKER_URL", DEFAULT_WORKER_URL))
    parser.add_argument("--output-dir", default="data/imported")
    parser.add_argument("--only", nargs="*", choices=sorted(ALLOWED_DATASETS))
    args = parser.parse_args()
    result = import_datasets(args.worker_url, args.output_dir, args.only)
    print(json.dumps({
        "imported": [x["dataset_id"] for x in result["datasets"]],
        "manifest": str(Path(args.output_dir) / "import_manifest.json"),
        "total_bytes": sum(x["bytes"] for x in result["datasets"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
