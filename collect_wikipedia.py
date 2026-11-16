"""Collect Russian Wikipedia article extracts into a quarantined IVA corpus.

Nothing collected here is automatically approved for training. Review each page's
content and licensing before changing its manifest entry to use="training".
"""
import argparse
import json
import re
from pathlib import Path

from iva_nano.search import collect_wikipedia_pages

SAFE_ID = re.compile(r"[^a-z0-9_-]+")


def collect(queries, output_dir="data/wikipedia_candidates", limit=5):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"datasets": []}
    entries = manifest.setdefault("datasets", [])
    known = {str(item.get("id")) for item in entries}
    added = []

    for query in queries:
        pages = collect_wikipedia_pages(query, limit=limit)
        for page in pages:
            source_id = f"wikipedia_ru_{page['page_id']}_{page['revision_id']}"
            if source_id in known:
                continue
            filename = f"{source_id}.txt"
            target = root / filename
            if target.exists():
                # Do not silently overwrite content whose provenance may differ.
                continue
            target.write_text(page["extract"].strip() + "\n", encoding="utf-8")
            title = page["title"]
            attribution = (
                f"Wikipedia contributors, «{title}», revision {page['revision_id']}, "
                f"{page['url']}; text under {page['license']}. "
                "Preserve attribution and license notice when redistributing."
            )
            entry = {
                "id": source_id,
                "path": str(target.as_posix()),
                "license": page["license"],
                "license_url": page["license_url"],
                "attribution": attribution,
                "source_title": title,
                "source_url": page["url"],
                "page_id": page["page_id"],
                "revision_id": page["revision_id"],
                "revision_timestamp": page["revision_timestamp"],
                "use": "pending_rights_review",
                "rights_status": "pending_human_review",
                "collection_method": "MediaWiki API search result extract; max 12000 chars",
            }
            entries.append(entry)
            known.add(source_id)
            added.append(entry)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Добавлено статей: {len(added)}")
    print(f"Всего в карантинном корпусе: {len(entries)}")
    print(f"Манифест: {manifest_path}")
    print("Все новые материалы остаются pending_rights_review; обучение ими не запускается.")
    return added


def main():
    parser = argparse.ArgumentParser(
        description="Collect Russian Wikipedia extracts for later human-reviewed IVA training."
    )
    parser.add_argument("queries", nargs="+", help="One or more Russian search topics")
    parser.add_argument("--output-dir", default="data/wikipedia_candidates")
    parser.add_argument("--limit", type=int, default=5, help="Pages per query (1-10)")
    args = parser.parse_args()
    if not 1 <= args.limit <= 10:
        parser.error("--limit must be between 1 and 10")
    collect(args.queries, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
