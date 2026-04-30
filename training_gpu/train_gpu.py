#!/usr/bin/env python3
"""
GPU Training for LMCS LLM - PyTorch Version
=============================================
Always uses GPU (Apple Metal MPS or NVIDIA CUDA).
Falls back to CPU only if no GPU is available.

Usage:
    python train_gpu.py --epochs 300 --batch-size 16
    python train_gpu.py --epochs 100 --batch-size 32
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Device selection — always prefer GPU
# ============================================================

def get_device() -> torch.device:
    """Select the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using Apple Metal GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU found, falling back to CPU")
    return device


# ============================================================
# Configuration
# ============================================================

class TransformerConfig:
    """Configuration for Transformer model."""

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 256,
        ff_hidden: int = 1024,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.ff_hidden = ff_hidden
        self.dropout = dropout

    def to_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TransformerConfig':
        # Filter out unknown keys
        known = {k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames}
        return cls(**known)


# ============================================================
# Model
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class CausalSelfAttention(nn.Module):
    """Efficient causal self-attention with manual QKV projection."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)

        # Pre-register causal mask buffer
        self.register_buffer("causal_mask", None)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)  # [B, S, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, S, D]

        # Reshape to heads
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, S, HD]
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]

        # Apply causal mask
        if self.causal_mask is None or self.causal_mask.size(0) < S:
            self.causal_mask = self._create_causal_mask(S, x.device)
        attn_weights.masked_fill_(self.causal_mask[:S, :S], float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum
        attn_out = torch.matmul(attn_weights, v)  # [B, H, S, HD]

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block with causal self-attention."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.ff_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(x)
        x = self.ln1(x + attn_out)

        # Feed-forward with residual
        x = self.ln2(x + self.ff(x))
        return x


class TransformerLM(nn.Module):
    """Transformer Language Model for text generation."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.final_ln = nn.LayerNorm(config.d_model, eps=1e-6)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights: output projection = token embedding transpose
        self.output_proj.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits [batch, seq_len, vocab_size]."""
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_ln(x)
        logits = self.output_proj(x)
        return logits


def create_transformer_model(config: TransformerConfig) -> TransformerLM:
    """Create a Transformer language model (PyTorch)."""
    return TransformerLM(config)


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):
    """Dataset for training on text sequences."""

    def __init__(self, text: str, word_to_id: Dict[str, int], max_seq_len: int = 256, stride: int = 128):
        self.word_to_id = word_to_id
        self.max_seq_len = max_seq_len
        self.unk_id = word_to_id.get("<UNK>", 1)
        self.pad_id = word_to_id.get("<PAD>", 0)

        # Tokenize text
        words = text.lower().split()
        self.tokens = []
        for word in words:
            word = word.strip(".,!?;:\"'()[]{}")
            if word:
                self.tokens.append(word_to_id.get(word, self.unk_id))

        # Create sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - max_seq_len, stride):
            self.sequences.append(self.tokens[i:i + max_seq_len + 1])

        print(f"Created {len(self.sequences)} sequences from {len(self.tokens)} tokens")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        # Pad if necessary
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = F.pad(input_ids, (0, pad_len), value=self.pad_id)
            target_ids = F.pad(target_ids, (0, pad_len), value=self.pad_id)
        return input_ids, target_ids


# ============================================================
# Vocabulary
# ============================================================

def extract_vocabulary(text: str, vocab_size: int = 8000) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Extract word-level vocabulary from text."""
    words = text.lower().split()
    cleaned = []
    for word in words:
        word = word.strip(".,!?;:\"'()[]{}")
        if len(word) > 0:
            cleaned.append(word)

    # Count frequency
    word_freq = {}
    for word in cleaned:
        word_freq[word] = word_freq.get(word, 0) + 1

    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_words) > vocab_size - 4:
        sorted_words = sorted_words[:vocab_size - 4]

    word_to_id = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    id_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}

    for i, (word, _) in enumerate(sorted_words):
        token_id = i + 4
        word_to_id[word] = token_id
        id_to_word[token_id] = word

    return word_to_id, id_to_word


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, dataloader, optimizer, scheduler, device, pad_id=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)  # [batch, seq, vocab]

        # Reshape for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)

        # Masked loss — ignore PAD tokens
        mask = (targets_flat != pad_id)
        loss = F.cross_entropy(logits_flat[mask], targets_flat[mask])

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        # scheduler is epoch-level, no step() here

        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()

        # Accuracy
        preds = logits_flat.argmax(dim=-1)
        correct = (preds[mask] == targets_flat[mask]).sum().item()
        total_correct += correct

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device, pad_id=0):
    """Evaluate on validation data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)

        mask = (targets_flat != pad_id)
        if mask.sum() == 0:
            continue
        loss = F.cross_entropy(logits_flat[mask], targets_flat[mask])
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description='GPU Training for LMCS LLM (PyTorch)')

    # Training
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max-seq-len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ff-hidden', type=int, default=1024, help='FFN hidden dimension')

    # Data
    parser.add_argument('--data-path', type=str, default='data/train.txt', help='Path to training data')
    parser.add_argument('--val-path', type=str, default=None, help='Path to validation data')
    parser.add_argument('--vocab-size', type=int, default=8000, help='Vocabulary size')

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--warmup-steps', type=int, default=200, help='Number of warmup steps')
    parser.add_argument('--save-path', type=str, default='checkpoints/model.pt', help='Path to save checkpoints')

    # Device (always GPU if available)
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, mps, cuda, cpu')

    args = parser.parse_args()

    # ---- Device ----
    if args.device == "auto":
        device = get_device()
    elif args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: Using CPU — training will be slow!")

    # ---- Load data ----
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        sys.exit(1)

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded {len(text)} characters from {data_path}")

    # ---- Vocabulary ----
    print("Extracting vocabulary...")
    word_to_id, id_to_word = extract_vocabulary(text, args.vocab_size)
    vocab_size = len(word_to_id)
    print(f"Vocabulary size: {vocab_size}")

    # Save vocabulary
    vocab_data = {
        "vocab_size": vocab_size,
        "word_to_id": word_to_id,
        "id_to_word": {str(k): v for k, v in id_to_word.items()},
    }
    with open("vocab_trained.json", 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)

    # ---- Config ----
    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        ff_hidden=args.ff_hidden,
    )

    # Save config
    with open("config.train.json", 'w') as f:
        json.dump({"model": config.to_dict()}, f, indent=2)

    # ---- Dataset ----
    train_dataset = TextDataset(text, word_to_id, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

    val_loader = None
    if args.val_path and Path(args.val_path).exists():
        with open(args.val_path, 'r', encoding='utf-8') as f:
            val_text = f.read()
        val_dataset = TextDataset(val_text, word_to_id, max_seq_len=args.max_seq_len)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- Model ----
    model = TransformerLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Device: {device}")

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}, LR: {args.learning_rate}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        scheduler.step()
        elapsed = time.time() - t0

        # Validation
        val_str = ""
        if val_loader:
            val_loss = evaluate(model, val_loader, device)
            val_str = f" | val_loss: {val_loss:.4f}"

        # Logging
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} | loss: {train_loss:.4f} | acc: {train_acc:.4f}{val_str} | lr: {lr:.2e} | {elapsed:.1f}s")

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'config': config.to_dict(),
            }, args.save_path.replace('.pt', '_best.pt'))
            print(f"  New best loss: {train_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'config': config.to_dict(),
            }, args.save_path)

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'config': config.to_dict(),
    }, args.save_path)

    # Completion marker
    completion_marker = Path(args.save_path).parent / "training_complete.txt"
    with open(completion_marker, 'w') as f:
        f.write(f"Exported to ../config.trained.json at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
