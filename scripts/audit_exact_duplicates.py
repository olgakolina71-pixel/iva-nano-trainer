#!/usr/bin/env python3
"""Exact duplicate audit for newline-delimited corpus files.

Uses the system sort utility so the full corpus is not held in RAM.
Only aggregate counts are written; source records are never emitted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def audit(path: Path) -> dict[str, int | str]:
    sort_bin = shutil.which("sort")
    if not sort_bin:
        raise RuntimeError("GNU sort is required for the bounded-memory duplicate audit")

    total = unique = duplicate_occurrences = duplicate_groups = max_repeat = 0
    previous: bytes | None = None
    current_count = 0

    with tempfile.TemporaryDirectory(prefix="iva-duplicate-sort-") as temp_dir:
        process = subprocess.Popen(
            [sort_bin, "--parallel=2", "-T", temp_dir, "--", str(path)],
            env={**os.environ, "LC_ALL": "C"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        try:
            for raw_line in process.stdout:
                record = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
                total += 1
                if previous is None:
                    previous = record
                    current_count = 1
                elif record == previous:
                    current_count += 1
                else:
                    unique += 1
                    if current_count > 1:
                        duplicate_groups += 1
                        duplicate_occurrences += current_count - 1
                        max_repeat = max(max_repeat, current_count)
                    previous = record
                    current_count = 1

            unique += int(previous is not None)
            if current_count > 1:
                duplicate_groups += 1
                duplicate_occurrences += current_count - 1
                max_repeat = max(max_repeat, current_count)
        finally:
            process.stdout.close()

        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"sort failed ({return_code}): {stderr[-2000:]}")

    return {
        "source_file": path.name,
        "total_records": total,
        "unique_records": unique,
        "repeated_occurrences": duplicate_occurrences,
        "duplicate_groups": duplicate_groups,
        "max_repeat": max_repeat,
        "method": "external_sort_exact_byte_line_comparison",
        "raw_records_emitted": 0,
        "training_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"Source file not found: {args.source}")

    report = audit(args.source)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))



if __name__ == "__main__":
    main()
