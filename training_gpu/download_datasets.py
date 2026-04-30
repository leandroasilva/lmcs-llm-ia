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
        "name": "instruct-aira-pt",
        "dataset": "nicholasKluge/instruct-aira-dataset-v2",
        "config": "default",
        "split": "portuguese",
        "format": "conversations_list",
        "role_mapping": {}
    },
    {
        "name": "told-br",
        "dataset": "mteb/told-br",
        "config": "default",
        "split": "train",
        "format": "text",
        "role_mapping": {}
    },
    {
        "name": "brazilian-toxic-tweets",
        "dataset": "mteb/BrazilianToxicTweetsClassification",
        "config": "default",
        "split": "train",
        "format": "text",
        "role_mapping": {}
    },
    {
        "name": "nemotron-personas-brazil",
        "dataset": "nvidia/Nemotron-Personas-Brazil",
        "config": "default",
        "split": "train",
        "format": "persona",
        "max_rows": 10000,
        "role_mapping": {}
    },
    {
        "name": "news-brazillian-clean",
        "dataset": "chenuneris/news-brazillian-clean",
        "config": "default",
        "split": "train",
        "format": "text",
        "max_rows": 50000,
        "role_mapping": {}
    },
    {
        "name": "musts-brazilian-portuguese",
        "dataset": "musts/brazilian_portuguese",
        "config": "default",
        "split": "train",
        "format": "sentence_pair",
        "role_mapping": {}
    },
    {
        "name": "brazilian-news-article-summarization-dpo",
        "dataset": "maikerdr/brazilian-news-article-summarization-DPO",
        "config": "default",
        "split": "train",
        "format": "article_dpo",
        "role_mapping": {}
    }
]



def fetch_page(base_url: str, offset: int, length: int, retries: int = 5, hf_token: str = None) -> Optional[Dict[str, Any]]:
    """Fetch a single page from HuggingFace datasets-server."""
    req = Request(base_url + f"&offset={offset}&length={length}")
    req.add_header("User-Agent", "LMCS-LLM/1.0")
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")

    for attempt in range(retries):
        try:
            with urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429:
                wait = min(2 ** (attempt + 2), 60)  # 4s, 8s, 16s up to 60s
                print(f"   Rate limited. Waiting {wait}s... (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
            elif e.code == 401:
                print(f"   HTTP 401: Unauthorized. Set HF_TOKEN environment variable.")
                return None
            else:
                print(f"   HTTP error {e.code}: {e.reason}")
                return None
        except Exception as e:
            print(f"   Error: {e}. Retry {attempt + 1}/{retries}")
            time.sleep(3)
    return None


def download_dataset(ds_config: Dict[str, Any], output_dir: Path, hf_token: str = None) -> Path:
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
    max_rows = ds_config.get("max_rows", 0)
    offset = 0
    length = 100
    page = 0

    while True:
        if max_rows > 0 and len(all_rows) >= max_rows:
            print(f"   Max rows limit reached ({max_rows})")
            all_rows = all_rows[:max_rows]
            break

        print(f"   Page {page + 1} (offset={offset})...", end=" ")
        data = fetch_page(base_url, offset, length, hf_token=hf_token)

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

    # Format 6: free text (single text field)
    elif "text" in row_data and isinstance(row_data["text"], str) and len(row_data["text"]) > 10:
        text = row_data["text"].strip()
        if text:
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": text})

    # Format 7: sentence pair (sentence1/sentence2 or sentence_1/sentence_2)
    elif ("sentence1" in row_data and "sentence2" in row_data) or \
         ("sentence_1" in row_data and "sentence_2" in row_data):
        s1 = str(row_data.get("sentence1", row_data.get("sentence_1", ""))).strip()
        s2 = str(row_data.get("sentence2", row_data.get("sentence_2", ""))).strip()
        if s1 and s2:
            messages.append({"role": "user", "content": s1})
            messages.append({"role": "assistant", "content": s2})

    # Format 8: conversations list (list of {role, content} dicts)
    elif "conversations" in row_data and isinstance(row_data["conversations"], list):
        for msg in row_data["conversations"]:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    mapped_role = role_mapping.get(role, role)
                    messages.append({"role": mapped_role, "content": content})

    # Format 9: persona fields (persona, cultural_background, etc.)
    elif "persona" in row_data and isinstance(row_data["persona"], str):
        parts = []
        for key in ["persona", "cultural_background", "skills_and_expertise",
                    "hobbies_and_interests", "career_goals_and_ambitions"]:
            val = row_data.get(key, "")
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        combined = "\n\n".join(parts)
        if len(combined) > 50:
            messages.append({"role": "user", "content": f"Descreva a pessoa: {row_data.get('occupation', 'pessoa')} de {row_data.get('state', 'Brasil')}"})
            messages.append({"role": "assistant", "content": combined})

    # Format 10: article with DPO triplets
    elif "article" in row_data and isinstance(row_data["article"], str) and len(row_data["article"]) > 50:
        article = row_data["article"].strip()
        triplets = row_data.get("triplets", [])
        if isinstance(triplets, list) and triplets:
            # Use first triplet's instruction + chosen as a conversation
            first = triplets[0] if isinstance(triplets[0], dict) else {}
            instruction = first.get("instruction", "Faça um resumo desta notícia.")
            chosen = first.get("chosen", "")
            if chosen:
                messages.append({"role": "user", "content": instruction})
                messages.append({"role": "assistant", "content": chosen})
            else:
                # Fallback: use article as text
                messages.append({"role": "user", "content": "Faça um resumo desta notícia."})
                messages.append({"role": "assistant", "content": article[:2000]})
        else:
            messages.append({"role": "user", "content": "Faça um resumo desta notícia."})
            messages.append({"role": "assistant", "content": article[:2000]})

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

    # HF token from environment
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print(f"HF_TOKEN found, using authenticated requests")
    else:
        print(f"No HF_TOKEN set. Some datasets may require authentication.")

    print(f"\n{'='*60}")
    print("LMCS LLM - Multi-Dataset Downloader")
    print(f"{'='*60}")
    print(f"Datasets to download: {len(datasets)}")

    downloaded = []
    for i, ds in enumerate(datasets):
        if i > 0:
            print(f"\n⏳ Waiting 5s before next dataset (rate limit protection)...")
            time.sleep(5)
        path = download_dataset(ds, output_dir, hf_token=hf_token)
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
