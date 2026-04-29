#!/usr/bin/env python3
"""
Multi-Dataset Downloader for LMCS LLM
======================================
Downloads conversational datasets from HuggingFace datasets-server.
Supports multiple datasets configured in a JSON array.

Usage:
    python download_datasets.py
    python download_datasets.py --config datasets_config.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError


DEFAULT_DATASETS = [
    {
        "name": "brazilian-customer-service",
        "dataset": "RichardSakaguchiMS/brazilian-customer-service-conversations",
        "config": "default",
        "split": "train",
        "format": "conversational",
        "role_mapping": {
            "customer": "user",
            "agent": "assistant"
        }
    },
    {
        "name": "multi-woz",
        "dataset": "multi_woz",
        "config": "v2.2",
        "split": "train",
        "format": "conversational",
        "role_mapping": {
            "user": "user",
            "system": "assistant"
        }
    }
]


def fetch_page(base_url: str, offset: int, length: int, retries: int = 3) -> Optional[Dict[str, Any]]:
    """Fetch a single page from HuggingFace datasets-server."""
    req = Request(base_url + f"&offset={offset}&length={length}")
    req.add_header("User-Agent", "LMCS-LLM/1.0")

    for attempt in range(retries):
        try:
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429:
                wait = 2 ** attempt
                print(f"   Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"   HTTP error {e.code}: {e.reason}")
                return None
        except Exception as e:
            print(f"   Error: {e}. Retry {attempt + 1}/{retries}")
            time.sleep(1)
    return None


def download_dataset(ds_config: Dict[str, Any], output_dir: Path) -> Path:
    """Download a single dataset and save as JSONL."""
    name = ds_config["name"]
    dataset = ds_config["dataset"]
    config = ds_config.get("config", "default")
    split = ds_config.get("split", "train")
    role_mapping = ds_config.get("role_mapping", {})

    print(f"\n📦 Downloading dataset: {name}")
    print(f"   Source: {dataset}")

    base_url = (
        f"https://datasets-server.huggingface.co/rows?"
        f"dataset={dataset.replace('/', '%2F')}&"
        f"config={config}&split={split}"
    )

    output_file = output_dir / f"{name}.jsonl"
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    length = 100
    page = 0

    while True:
        print(f"   Page {page + 1} (offset={offset})...", end=" ")
        data = fetch_page(base_url, offset, length)

        if data is None:
            print("failed")
            break

        rows = data.get("rows", [])
        if not rows:
            print("no more data")
            break

        normalized = normalize_rows(rows, role_mapping)
        all_rows.extend(normalized)
        print(f"got {len(normalized)} rows (total: {len(all_rows)})")

        if len(rows) < length:
            print("   Last page reached")
            break

        offset += length
        page += 1
        time.sleep(0.5)

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"   Saved to {output_file} ({len(all_rows)} conversations)")
    return output_file


def normalize_rows(rows: List[Dict[str, Any]], role_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """Normalize raw HuggingFace rows into a unified conversational format."""
    normalized = []
    for row in rows:
        row_data = row.get("row", {})
        messages = extract_messages(row_data, role_mapping)
        if messages:
            normalized.append({
                "messages": messages,
                "metadata": {
                    k: v for k, v in row_data.items()
                    if k != "messages" and not isinstance(v, (list, dict))
                }
            })
    return normalized


def extract_messages(row_data: Dict[str, Any], role_mapping: Dict[str, str]) -> List[Dict[str, str]]:
    """Extract messages from various dataset formats."""
    messages = []

    # Format 1: messages array with role/content
    if "messages" in row_data and isinstance(row_data["messages"], list):
        for msg in row_data["messages"]:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    mapped_role = role_mapping.get(role, role)
                    messages.append({"role": mapped_role, "content": content})

    # Format 2: turns array
    elif "turns" in row_data and isinstance(row_data["turns"], list):
        for turn in row_data["turns"]:
            if isinstance(turn, dict):
                speaker = turn.get("speaker", turn.get("role", ""))
                text = turn.get("text", turn.get("utterance", turn.get("content", "")))
                if speaker and text:
                    mapped_role = role_mapping.get(speaker, speaker)
                    messages.append({"role": mapped_role, "content": text})

    # Format 3: instruction/response pairs
    elif "instruction" in row_data and "response" in row_data:
        messages.append({"role": "user", "content": str(row_data["instruction"])})
        messages.append({"role": "assistant", "content": str(row_data["response"])})

    # Format 4: input/output pairs
    elif "input" in row_data and "output" in row_data:
        messages.append({"role": "user", "content": str(row_data["input"])})
        messages.append({"role": "assistant", "content": str(row_data["output"])})

    # Format 5: question/answer pairs
    elif "question" in row_data and "answer" in row_data:
        messages.append({"role": "user", "content": str(row_data["question"])})
        messages.append({"role": "assistant", "content": str(row_data["answer"])})

    return messages


def save_config(datasets: List[Dict[str, Any]], path: Path) -> None:
    """Save dataset configuration to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(datasets, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Dataset config saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Download conversational datasets")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to dataset config JSON")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                        help="Output directory for raw datasets")
    parser.add_argument("--save-config", action="store_true",
                        help="Save default config and exit")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create config
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            datasets = json.load(f)
        print(f"Loaded config from {args.config}")
    else:
        datasets = DEFAULT_DATASETS
        config_path = output_dir / "datasets_config.json"
        save_config(datasets, config_path)
        if args.save_config:
            return

    print(f"\n{'='*60}")
    print("LMCS LLM - Multi-Dataset Downloader")
    print(f"{'='*60}")
    print(f"Datasets to download: {len(datasets)}")

    downloaded = []
    for ds in datasets:
        path = download_dataset(ds, output_dir)
        downloaded.append({"name": ds["name"], "path": str(path), "config": ds})

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(downloaded, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Download complete! {len(downloaded)} datasets saved.")
    print(f"Manifest: {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
