#!/usr/bin/env python3
"""
Extract vocabulary from training dataset to enable text generation testing
"""

import json
import sys
from pathlib import Path

def extract_vocabulary(dataset_path, vocab_size=8000):
    """Extract word-level vocabulary from training text"""
    
    print(f"Reading dataset: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset size: {len(text)} characters")
    
    # Tokenize by whitespace (same as training)
    words = text.lower().split()
    
    # Clean punctuation
    cleaned_words = []
    for word in words:
        word = word.strip(".,!?;:\"'()[]{}")
        if len(word) > 0:
            cleaned_words.append(word)
    
    print(f"Total words: {len(cleaned_words)}")
    
    # Count frequency
    word_freq = {}
    for word in cleaned_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    print(f"Unique words: {len(word_freq)}")
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Limit vocabulary
    if len(sorted_words) > vocab_size - 4:  # Reserve 0-3 for special tokens
        sorted_words = sorted_words[:vocab_size - 4]
    
    # Create mappings
    word_to_id = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
    }
    id_to_word = {
        0: "<PAD>",
        1: "<UNK>",
        2: "<BOS>",
        3: "<EOS>",
    }
    
    for i, (word, freq) in enumerate(sorted_words):
        token_id = i + 4
        word_to_id[word] = token_id
        id_to_word[token_id] = word
    
    print(f"Vocabulary size: {len(word_to_id)}")
    
    # Save to JSON
    output_path = "vocab_trained.json"
    vocab_data = {
        "vocab_size": len(word_to_id),
        "word_to_id": word_to_id,
        "id_to_word": {str(k): v for k, v in id_to_word.items()},
        "top_words": [word for word, _ in sorted_words[:100]],
    }
    
    print(f"Saving vocabulary to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Vocabulary extracted successfully!")
    print(f"\nTop 20 most frequent words:")
    for i, (word, freq) in enumerate(sorted_words[:20]):
        print(f"  {i+1}. {word} (freq: {freq})")
    
    return output_path

if __name__ == "__main__":
    # Find training dataset
    dataset_path = "dataset/data/train.txt"
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    if not Path(dataset_path).exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Usage: python extract_vocabulary.py [dataset_path]")
        sys.exit(1)
    
    extract_vocabulary(dataset_path)
