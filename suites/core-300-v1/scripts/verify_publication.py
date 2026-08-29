#!/usr/bin/env python3
"""Verify count, identity, source distribution, hashes, and publication hygiene."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest.jsonl"
SOURCES = ROOT / "sources.json"
EXPECTED_TASK_SET_SHA256 = "b0beacae2973ff00a30af0b36a699d1e12bfaf29d386aad8b599ed252f2344d6"
EXPECTED_PUBLIC_MANIFEST_SHA256 = "703a6e5a9073f2b434a175eeefbb48a7ec4024027c6cf9e7f29846d3c6d74eec"
EXPECTED_SOURCES_SHA256 = "d5711e24fdf6df1f06614d59aee68e9e4418524caca4b0cb80201281aaa91b20"
EXPECTED_DATASETS = Counter({
    "evalplus/humanevalplus": 75,
    "evalplus/mbppplus": 75,
    "TIGER-Lab/MMLU-Pro": 150,
})
EXPECTED_TOP_KEYS = {
    "index",
    "task_id",
    "family",
    "language",
    "max_output_tokens",
    "protocol_version",
    "historical_task_sha256",
    "prompt_sha256",
    "source",
}
EXPECTED_SOURCE_KEYS = {
    "evalplus/humanevalplus": {
        "dataset", "revision", "split", "row_index", "evalplus_task_id", "raw_sha256",
        "evalplus_release_sha256", "prompt_source",
    },
    "evalplus/mbppplus": {
        "dataset", "revision", "split", "row_index", "evalplus_task_id", "raw_sha256",
        "evalplus_release_sha256", "prompt_source",
    },
    "TIGER-Lab/MMLU-Pro": {
        "dataset", "revision", "split", "row_index", "question_id", "category", "upstream_source",
        "raw_sha256",
    },
}
BANNED_KEYS = {
    "answer",
    "answers",
    "arm_order",
    "choices",
    "gold",
    "prompt",
    "scorer",
    "selection_index",
    "artifact_path",
    "quantize_command",
}
BANNED_TEXT = (
    "/Users/", "/home/", "/private/", "api_key", "authorization:", "bearer ", "password", "sk-",
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def walk(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in BANNED_KEYS:
                raise ValueError(f"banned field {path}.{key}")
            walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk(child, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for fragment in BANNED_TEXT:
            if fragment.lower() in lowered:
                raise ValueError(f"banned text fragment at {path}: {fragment}")
        if WINDOWS_ABSOLUTE_PATH.match(value):
            raise ValueError(f"absolute Windows path at {path}")


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 300:
        raise ValueError(f"expected 300 rows, got {len(rows)}")
    if [row.get("index") for row in rows] != list(range(1, 301)):
        raise ValueError("index must be the contiguous range 1..300")
    if len({row.get("task_id") for row in rows}) != 300:
        raise ValueError("task IDs are not unique")
    if len({row.get("historical_task_sha256") for row in rows}) != 300:
        raise ValueError("historical task hashes are not unique")
    if len({row.get("prompt_sha256") for row in rows}) != 300:
        raise ValueError("prompt hashes are not unique")
    if [row.get("task_id") for row in rows] != sorted(row.get("task_id") for row in rows):
        raise ValueError("public manifest must use task-ID order, not execution order")

    datasets = Counter()
    for row in rows:
        if set(row) != EXPECTED_TOP_KEYS:
            difference = sorted(set(row) ^ EXPECTED_TOP_KEYS)
            raise ValueError(f"unexpected top-level fields in {row.get('task_id')}: {difference}")
        if row["family"] not in {"public_code", "public_reasoning"}:
            raise ValueError(f"unexpected family in {row['task_id']}")
        if row["max_output_tokens"] != 8192 or row["protocol_version"] != "qwen38-q8-q4-v1":
            raise ValueError(f"protocol mismatch in {row['task_id']}")
        for field in ("historical_task_sha256", "prompt_sha256"):
            if not HEX64.fullmatch(row[field]):
                raise ValueError(f"invalid {field} in {row['task_id']}")
        dataset = row["source"]["dataset"]
        if set(row["source"]) != EXPECTED_SOURCE_KEYS.get(dataset):
            raise ValueError(f"unexpected source fields in {row['task_id']}")
        datasets[dataset] += 1
        walk(row)
    if datasets != EXPECTED_DATASETS:
        raise ValueError(f"dataset counts differ: {datasets}")

    sources_sha256 = hashlib.sha256(SOURCES.read_bytes()).hexdigest()
    if sources_sha256 != EXPECTED_SOURCES_SHA256:
        raise ValueError(f"sources registry hash mismatch: expected {EXPECTED_SOURCES_SHA256}, got {sources_sha256}")
    registry = json.loads(SOURCES.read_text(encoding="utf-8"))
    by_repo = {source["repo_id"]: source for source in registry["sources"].values()}
    for row in rows:
        source = row["source"]
        registered = by_repo.get(source["dataset"])
        if not registered:
            raise ValueError(f"unregistered dataset in {row['task_id']}")
        if source["revision"] != registered["revision"] or source["raw_sha256"] != registered["raw_sha256"]:
            raise ValueError(f"source identity mismatch in {row['task_id']}")

    task_set = hashlib.sha256(canonical_bytes([row["historical_task_sha256"] for row in rows])).hexdigest()
    if task_set != EXPECTED_TASK_SET_SHA256:
        raise ValueError(f"task-set hash mismatch: expected {EXPECTED_TASK_SET_SHA256}, got {task_set}")
    manifest_sha256 = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if manifest_sha256 != EXPECTED_PUBLIC_MANIFEST_SHA256:
        raise ValueError(
            f"public manifest hash mismatch: expected {EXPECTED_PUBLIC_MANIFEST_SHA256}, got {manifest_sha256}"
        )
    print(json.dumps({
        "status": "ok",
        "rows": len(rows),
        "datasets": dict(sorted(datasets.items())),
        "task_set_sha256": task_set,
        "public_manifest_sha256": manifest_sha256,
        "sources_sha256": sources_sha256,
        "publication_hygiene": "no embedded prompt, answer, scorer, execution order, local path, or credential found",
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
