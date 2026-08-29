#!/usr/bin/env python3
"""Export a safe, answer-free public manifest from the frozen internal manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_ROWS = 300
EXPECTED_DATASETS = Counter({
    "evalplus/humanevalplus": 75,
    "evalplus/mbppplus": 75,
    "TIGER-Lab/MMLU-Pro": 150,
})
SOURCE_KEYS = {
    "dataset",
    "revision",
    "split",
    "row_index",
    "evalplus_task_id",
    "question_id",
    "category",
    "upstream_source",
    "raw_sha256",
    "evalplus_release_sha256",
    "prompt_source",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def canonical_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def export_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    required = {
        "task_id",
        "family",
        "language",
        "max_output_tokens",
        "protocol_version",
        "prompt",
        "task_sha256",
        "source",
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(f"row {index} missing fields: {sorted(missing)}")
    if row["family"] not in {"public_code", "public_reasoning"}:
        raise ValueError(f"row {index} is outside the completed public core")
    source = {key: row["source"][key] for key in SOURCE_KEYS if key in row["source"]}
    return {
        "index": index,
        "task_id": row["task_id"],
        "family": row["family"],
        "language": row["language"],
        "max_output_tokens": row["max_output_tokens"],
        "protocol_version": row["protocol_version"],
        "historical_task_sha256": row["task_sha256"],
        "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="frozen internal 300-row JSONL manifest")
    parser.add_argument("--output", type=Path, required=True, help="answer-free public JSONL manifest")
    args = parser.parse_args()

    source_rows = sorted(read_jsonl(args.source), key=lambda row: str(row.get("task_id", "")))
    if len(source_rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} source rows, got {len(source_rows)}")
    if len({row.get("task_id") for row in source_rows}) != EXPECTED_ROWS:
        raise ValueError("source task IDs are not unique")
    datasets = Counter(row.get("source", {}).get("dataset") for row in source_rows)
    if datasets != EXPECTED_DATASETS:
        raise ValueError(f"source dataset counts differ: {datasets}")

    public_rows = [export_row(row, index) for index, row in enumerate(source_rows, 1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in public_rows:
            handle.write(canonical_line(row))
    print(f"wrote {len(public_rows)} answer-free rows to {args.output}")


if __name__ == "__main__":
    main()
