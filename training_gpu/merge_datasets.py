#!/usr/bin/env python3
"""
Dataset Merger for LMCS LLM
===========================
Normalizes and merges multiple conversational datasets into a unified
training corpus compatible with the Go Transformer model.

Usage:
    python merge_datasets.py --input-dir data/raw --output data/merged_train.txt
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List


def clean_text(text: str) -> str:
    """Clean and normalize text for training."""
    text = text.strip()
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove very short or empty messages
    if len(text) < 2:
        return ""
    return text


def format_conversation(messages: List[Dict[str, str]]) -> str:
    """Format a conversation into training text."""
    lines = []
    for msg in messages:
        role = msg.get("role", "")
        content = clean_text(msg.get("content", ""))
        if not content:
            continue

        if role in ("user", "customer", "human"):
            lines.append(f"Usuário: {content}")
        elif role in ("assistant", "agent", "ai", "system"):
            lines.append(f"Assistente: {content}")
        else:
            lines.append(f"{role.capitalize()}: {content}")

    if not lines:
        return ""

    return "\n".join(lines) + "\n"


def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file."""
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conversations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return conversations


def merge_datasets(input_dir: Path, output_path: Path, shuffle: bool = True,
                   min_turns: int = 2, max_turns: int = 20) -> Dict:
    """Merge all datasets in input_dir into a single training file."""
    all_conversations: List[str] = []
    stats = {"datasets": {}, "total_conversations": 0, "total_turns": 0}

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in {input_dir}")
        return stats

    for file_path in jsonl_files:
        name = file_path.stem
        print(f"Processing {name}...")

        conversations = load_jsonl(file_path)
        dataset_blocks = []
        dataset_turns = 0

        for conv in conversations:
            messages = conv.get("messages", [])
            if not messages:
                continue

            # Filter by turn count
            turn_count = len(messages)
            if turn_count < min_turns or turn_count > max_turns:
                continue

            block = format_conversation(messages)
            if block:
                dataset_blocks.append(block)
                dataset_turns += turn_count

        all_conversations.extend(dataset_blocks)
        stats["datasets"][name] = {
            "conversations": len(dataset_blocks),
            "turns": dataset_turns
        }
        stats["total_conversations"] += len(dataset_blocks)
        stats["total_turns"] += dataset_turns
        print(f"   {len(dataset_blocks)} conversations, {dataset_turns} turns")

    if shuffle:
        random.shuffle(all_conversations)

    # Write merged output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for block in all_conversations:
            f.write(block)
            f.write("\n")  # Separator between conversations

    # Write statistics
    stats_path = output_path.parent / "merge_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Merged dataset saved to: {output_path}")
    print(f"Total conversations: {stats['total_conversations']}")
    print(f"Total turns: {stats['total_turns']}")
    print(f"Output size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*50}")

    return stats


def split_train_val(input_path: Path, train_path: Path, val_path: Path,
                   val_ratio: float = 0.05) -> None:
    """Split merged dataset into train and validation sets."""
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by double newline (conversation separator)
    conversations = [c.strip() for c in content.split("\n\n") if c.strip()]
    random.shuffle(conversations)

    split_idx = int(len(conversations) * (1 - val_ratio))
    train_convs = conversations[:split_idx]
    val_convs = conversations[split_idx:]

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(train_convs))

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(val_convs))

    print(f"Train set: {len(train_convs)} conversations -> {train_path}")
    print(f"Val set: {len(val_convs)} conversations -> {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge conversational datasets")
    parser.add_argument("--input-dir", type=str, default="data/raw",
                        help="Directory containing .jsonl datasets")
    parser.add_argument("--output", type=str, default="data/merged_train.txt",
                        help="Output merged training file")
    parser.add_argument("--train-output", type=str, default="data/train.txt",
                        help="Train split output")
    parser.add_argument("--val-output", type=str, default="data/val.txt",
                        help="Validation split output")
    parser.add_argument("--split", action="store_true",
                        help="Also create train/val split")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle conversations")
    parser.add_argument("--min-turns", type=int, default=2,
                        help="Minimum turns per conversation")
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Maximum turns per conversation")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        print("Run download_datasets.py first.")
        sys.exit(1)

    merge_datasets(
        input_dir=input_dir,
        output_path=output_path,
        shuffle=args.shuffle,
        min_turns=args.min_turns,
        max_turns=args.max_turns
    )

    if args.split:
        split_train_val(
            output_path,
            Path(args.train_output),
            Path(args.val_output)
        )


if __name__ == "__main__":
    main()
