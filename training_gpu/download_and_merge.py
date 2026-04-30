#!/usr/bin/env python3
"""
Download and merge all datasets using HuggingFace datasets library.
Avoids API rate limiting by downloading parquet files directly.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

def extract_conversations(row, ds_name):
    """Extract text from various dataset formats."""
    texts = []
    
    # Format: messages array
    if "messages" in row and isinstance(row["messages"], list):
        for msg in row["messages"]:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    texts.append(f"{role}: {content}")
    
    # Format: conversations array
    elif "conversations" in row and isinstance(row["conversations"], list):
        for msg in row["conversations"]:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    texts.append(f"{role}: {content}")
    
    # Format: text field
    elif "text" in row and isinstance(row["text"], str) and len(row["text"]) > 10:
        t = row["text"].strip()
        if t:
            texts.append(f"text: {t}")
    
    # Format: sentence_1 / sentence_2
    elif "sentence_1" in row and "sentence_2" in row:
        s1 = str(row["sentence_1"]).strip()
        s2 = str(row["sentence_2"]).strip()
        if s1 and s2:
            texts.append(f"user: {s1}")
            texts.append(f"assistant: {s2}")
    
    # Format: persona fields
    elif "persona" in row and isinstance(row.get("persona"), str):
        parts = []
        for key in ["persona", "cultural_background", "skills_and_expertise",
                    "hobbies_and_interests", "career_goals_and_ambitions"]:
            val = row.get(key, "")
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        combined = "\n\n".join(parts)
        if len(combined) > 50:
            texts.append(f"user: Descreva a pessoa: {row.get('occupation', 'pessoa')} de {row.get('state', 'Brasil')}")
            texts.append(f"assistant: {combined}")
    
    # Format: article with DPO triplets
    elif "article" in row and isinstance(row["article"], str) and len(row["article"]) > 50:
        article = row["article"].strip()
        triplets = row.get("triplets", [])
        if isinstance(triplets, list) and triplets:
            first = triplets[0] if isinstance(triplets[0], dict) else {}
            instruction = first.get("instruction", "Faça um resumo desta notícia.")
            chosen = first.get("chosen", "")
            if chosen:
                texts.append(f"user: {instruction}")
                texts.append(f"assistant: {chosen}")
            else:
                texts.append(f"user: Faça um resumo desta notícia.")
                texts.append(f"assistant: {article[:2000]}")
        else:
            texts.append(f"user: Faça um resumo desta notícia.")
            texts.append(f"assistant: {article[:2000]}")
    
    # Format: turns array
    elif "turns" in row and isinstance(row["turns"], list):
        for turn in row["turns"]:
            if isinstance(turn, dict):
                speaker = turn.get("speaker", turn.get("role", ""))
                text = turn.get("text", turn.get("utterance", turn.get("content", "")))
                if speaker and text:
                    texts.append(f"{speaker}: {text}")
    
    # Format: instruction/response
    elif "instruction" in row and "response" in row:
        texts.append(f"user: {row['instruction']}")
        texts.append(f"assistant: {row['response']}")
    
    # Format: input/output
    elif "input" in row and "output" in row:
        texts.append(f"user: {row['input']}")
        texts.append(f"assistant: {row['output']}")
    
    # Format: question/answer
    elif "question" in row and "answer" in row:
        texts.append(f"user: {row['question']}")
        texts.append(f"assistant: {row['answer']}")
    
    return "\n".join(texts) if texts else ""


def download_and_save(name, dataset_id, config, split, max_rows=0):
    """Download a dataset and save as JSONL."""
    print(f"\n📦 Downloading: {name} ({dataset_id})")
    try:
        ds = load_dataset(dataset_id, config, split=split, trust_remote_code=True)
        
        if max_rows > 0 and len(ds) > max_rows:
            ds = ds.select(range(max_rows))
        
        output_file = output_dir / f"{name}.jsonl"
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for row in ds:
                text = extract_conversations(row, name)
                if text and len(text) > 20:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    count += 1
        
        print(f"   Saved {count} examples to {output_file}")
        return count
    except Exception as e:
        print(f"   Error: {e}")
        return 0


def merge_all():
    """Merge all JSONL files into train.txt and val.txt."""
    all_texts = []
    
    for jsonl_file in sorted(output_dir.glob("*.jsonl")):
        if jsonl_file.stat().st_size == 0:
            continue
        print(f"Merging: {jsonl_file.name}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "text" in data and data["text"]:
                        all_texts.append(data["text"])
                    elif "messages" in data:
                        msgs = data["messages"]
                        text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs if m.get("content"))
                        if text:
                            all_texts.append(text)
                except json.JSONDecodeError:
                    continue
    
    # Split 95/5
    split_idx = int(len(all_texts) * 0.95)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    # Write train.txt
    with open("data/train.txt", "w", encoding="utf-8") as f:
        for text in train_texts:
            f.write(text + "\n<|endoftext|>\n")
    
    # Write val.txt
    with open("data/val.txt", "w", encoding="utf-8") as f:
        for text in val_texts:
            f.write(text + "\n<|endoftext|>\n")
    
    total_chars = sum(len(t) for t in all_texts)
    print(f"\n{'='*60}")
    print(f"Merged: {len(all_texts)} total examples")
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
    print(f"Total chars: {total_chars:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    datasets_to_download = [
        ("brazilian-customer-service", "RichardSakaguchiMS/brazilian-customer-service-conversations", "default", "train"),
        ("instruct-aira-pt", "nicholasKluge/instruct-aira-dataset-v2", "default", "portuguese"),
        ("told-br", "mteb/told-br", "default", "train"),
        ("brazilian-toxic-tweets", "mteb/BrazilianToxicTweetsClassification", "default", "train"),
        ("nemotron-personas-brazil", "nvidia/Nemotron-Personas-Brazil", "default", "train", 10000),
        ("news-brazillian-clean", "chenuneris/news-brazillian-clean", "default", "train", 50000),
        ("musts-brazilian-portuguese", "musts/brazilian_portuguese", "default", "train"),
        ("brazilian-news-article-summarization-dpo", "maikerdr/brazilian-news-article-summarization-DPO", "default", "train"),
    ]
    
    total_count = 0
    for item in datasets_to_download:
        name, ds_id, config, split = item[0], item[1], item[2], item[3]
        max_rows = item[4] if len(item) > 4 else 0
        count = download_and_save(name, ds_id, config, split, max_rows)
        total_count += count
    
    print(f"\n{'='*60}")
    print(f"Total examples downloaded: {total_count}")
    print(f"{'='*60}")
    
    merge_all()
