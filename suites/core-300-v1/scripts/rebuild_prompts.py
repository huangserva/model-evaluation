#!/usr/bin/env python3
"""Rebuild the exact 300 prompts from pinned upstream snapshots, without answers."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CODE_PREFIX = (
    "Solve the following Python problem. Think carefully, then output exactly one complete Python code block. "
    "Do not include tests, prose, or shell commands.\n\n"
)
REASONING_PREFIX = "Reason through this problem, then output only FINAL=<option letter>.\n\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_verified(url: str, target: Path, expected: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256_file(target) == expected:
        return
    partial = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    actual = sha256_file(partial)
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise ValueError(f"download hash mismatch: expected {expected}, got {actual}")
    partial.replace(target)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_url(source: dict[str, Any]) -> str:
    return (
        f"https://huggingface.co/datasets/{source['repo_id']}/resolve/"
        f"{source['revision']}/{source['filename']}"
    )


def load_release(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {str(row["task_id"]): row for row in rows}


def code_prompt(
    row: dict[str, Any],
    releases: dict[str, dict[str, dict[str, Any]]],
    parquet_docs: dict[str, list[dict[str, Any]]],
) -> str:
    dataset = row["source"]["dataset"]
    if dataset == "evalplus/humanevalplus":
        source_name = "humanevalplus"
    elif dataset == "evalplus/mbppplus":
        source_name = "mbppplus"
    else:
        raise ValueError(f"unexpected code dataset {dataset}")
    task_id = str(row["source"]["evalplus_task_id"])
    row_index = int(row["source"]["row_index"])
    try:
        parquet_doc = parquet_docs[source_name][row_index]
    except IndexError as exc:
        raise ValueError(f"invalid {source_name} row index {row_index}") from exc
    parquet_task_id = (
        str(parquet_doc["task_id"])
        if source_name == "humanevalplus"
        else f"Mbpp/{int(parquet_doc['task_id'])}"
    )
    if parquet_task_id != task_id:
        raise ValueError(f"{source_name} identity mismatch at row {row_index}")
    try:
        official = releases[source_name][task_id]
    except KeyError as exc:
        raise ValueError(f"upstream release is missing {task_id}") from exc
    return CODE_PREFIX + str(official["prompt"])


def reasoning_prompt(row: dict[str, Any], docs: list[dict[str, Any]]) -> str:
    row_index = int(row["source"]["row_index"])
    if row_index < 0 or row_index >= len(docs):
        raise ValueError(f"invalid MMLU-Pro row index {row_index}")
    doc = docs[row_index]
    if int(doc["question_id"]) != int(row["source"]["question_id"]):
        raise ValueError(f"MMLU-Pro identity mismatch at row {row_index}")
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rendered = "\n".join(f"{labels[index]}. {value}" for index, value in enumerate(doc["options"]))
    return REASONING_PREFIX + f"{doc['question']}\n\n{rendered}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.jsonl")
    parser.add_argument("--sources", type=Path, default=ROOT / "sources.json")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "model-evaluation-core-300-v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    if len(manifest) != 300:
        raise ValueError(f"expected 300 manifest rows, got {len(manifest)}")
    registry = json.loads(args.sources.read_text(encoding="utf-8"))
    sources = registry["sources"]

    parquet_paths: dict[str, Path] = {}
    releases: dict[str, dict[str, dict[str, Any]]] = {}
    for name, source in sources.items():
        raw_path = args.cache_dir / f"{name}-{source['revision']}.parquet"
        fetch_verified(source_url(source), raw_path, source["raw_sha256"])
        parquet_paths[name] = raw_path
        release = source.get("evalplus_release")
        if release:
            release_path = args.cache_dir / release["filename"]
            fetch_verified(release["url"], release_path, release["sha256"])
            releases[name] = load_release(release_path)

    parquet_docs = {name: pq.read_table(path).to_pylist() for name, path in parquet_paths.items()}
    for name, docs in parquet_docs.items():
        if len(docs) != int(sources[name]["rows"]):
            raise ValueError(f"{name}: expected {sources[name]['rows']} rows, got {len(docs)}")
    for name, release_rows in releases.items():
        if len(release_rows) != int(sources[name]["rows"]):
            raise ValueError(f"{name} release: expected {sources[name]['rows']} rows, got {len(release_rows)}")
    mmlu_docs = parquet_docs["mmlu_pro"]
    rebuilt: list[dict[str, Any]] = []
    for row in manifest:
        if row["family"] == "public_code":
            prompt = code_prompt(row, releases, parquet_docs)
        elif row["family"] == "public_reasoning":
            prompt = reasoning_prompt(row, mmlu_docs)
        else:
            raise ValueError(f"unexpected family {row['family']}")
        actual = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if actual != row["prompt_sha256"]:
            raise ValueError(
                f"prompt hash mismatch for {row['task_id']}: expected {row['prompt_sha256']}, got {actual}"
            )
        rebuilt.append({
            "index": row["index"],
            "task_id": row["task_id"],
            "prompt": prompt,
            "prompt_sha256": actual,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rebuilt:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"rebuilt and verified {len(rebuilt)} prompts: {args.output}")


if __name__ == "__main__":
    main()
