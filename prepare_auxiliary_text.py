"""Convert reviewed KartaSlov lexical datasets into attributed Russian training lines.

This creates task-oriented synthetic sentences, not a general Russian text corpus.
It intentionally excludes semantic hierarchy and letter-frequency matrix files.
"""
import argparse
import csv
import json
from pathlib import Path

LICENSE = "CC BY-NC-SA 4.0"
ATTRIBUTION = "KartaSlov / D.I. Kulagin, https://github.com/dkulagin/kartaslov"
SOURCES = {
    "assoc_safe": "https://github.com/dkulagin/kartaslov/tree/master/dataset/assoc",
    "kartaslovsent": "https://github.com/dkulagin/kartaslov/tree/master/dataset/kartaslovsent",
}


def read_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        except csv.Error:
            dialect = csv.excel
            dialect.delimiter = ";"
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"{path}: missing CSV header")
        for row in reader:
            yield {str(k).strip(): (v or "").strip() for k, v in row.items() if k}


def association_lines(path):
    count = 0
    for row in read_rows(path):
        word, assoc = row.get("word", ""), row.get("assoc", "")
        safe = row.get("is_safe", "1")
        if not word or not assoc or safe not in {"1", "1.0", "true", "True"}:
            continue
        yield f"Слово «{word}» вызывает ассоциацию «{assoc}»."
        yield f"К слову «{word}» люди подбирают ассоциацию «{assoc}»."
        count += 1
    if count == 0:
        raise ValueError(f"{path}: no usable safe word-association rows; expected word, assoc, is_safe")


def sentiment_lines(path):
    labels = {"PSTV": "положительную", "NGTV": "отрицательную", "NEUT": "нейтральную"}
    count = 0
    for row in read_rows(path):
        term, tag = row.get("term", ""), row.get("tag", "").upper()
        if not term or tag not in labels:
            continue
        yield f"Слово или выражение «{term}» имеет {labels[tag]} эмоциональную окраску."
        yield f"В тональном словаре выражение «{term}» отмечено как {labels[tag]}."
        count += 1
    if count == 0:
        raise ValueError(f"{path}: no usable sentiment rows; expected term and tag")


def prepare(assoc_csv, sentiment_csv, output_dir, manifest_path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    text_path = out / "kartaslov_auxiliary.txt"
    lines = []
    if assoc_csv:
        lines.extend(association_lines(assoc_csv))
    if sentiment_csv:
        lines.extend(sentiment_lines(sentiment_csv))
    if not lines:
        raise ValueError("No source datasets selected")
    # Stable de-duplication preserves source order.
    lines = list(dict.fromkeys(lines))
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "datasets": [{
            "id": "kartaslov_auxiliary",
            "path": str(text_path.resolve()),
            "license": LICENSE,
            "attribution": ATTRIBUTION,
            "source_urls": [url for key, url in SOURCES.items()
                            if (key == "assoc_safe" and assoc_csv) or (key == "kartaslovsent" and sentiment_csv)],
            "use": "training",
            "content_type": "synthetic_task_sentences_from_lexical_annotations",
            "limitations": "Auxiliary lexical examples only; not a general-purpose Russian conversation corpus."
        }]
    }
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"sentences": len(lines), "text": str(text_path), "manifest": str(manifest_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assoc-csv")
    parser.add_argument("--sentiment-csv")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--manifest", default="data/manifest.json")
    args = parser.parse_args()
    result = prepare(args.assoc_csv, args.sentiment_csv, args.output_dir, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
