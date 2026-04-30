#!/usr/bin/env python3
"""
Export PyTorch Model to Go-Compatible JSON
==========================================
Converts a trained PyTorch model checkpoint into a JSON format
that the Go server can load and serve.

Usage:
    python export_to_go.py --checkpoint checkpoints/model_best.pt \
                           --vocab vocab_trained.json \
                           --config config.train.json \
                           --output ../config.trained.json
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Import training module for model creation
sys.path.insert(0, str(Path(__file__).parent))
from train_gpu import TransformerConfig, TransformerLM, create_transformer_model


def positional_encoding(max_seq_len: int, d_model: int) -> np.ndarray:
    """Create sinusoidal positional encodings matching the training script."""
    positions = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((max_seq_len, d_model))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    return pe


def map_pytorch_weights_to_go(model: TransformerLM, config: TransformerConfig) -> dict:
    """Extract and rename PyTorch weights to match Go loader expectations."""
    weights = {}
    state = model.state_dict()

    # Token embedding — PyTorch stores as [vocab_size, d_model]
    weights["token_embedding.weight"] = state["token_embedding.weight"].numpy().tolist()

    # Output projection is tied with token_embedding, so Go clones it.
    # But we also export BOut as zeros.
    weights["output_projection.bias"] = np.zeros(config.vocab_size, dtype=np.float32).tolist()

    # Position embedding — computed, not learned; generate for Go
    pos_enc = positional_encoding(config.max_seq_len, config.d_model)
    weights["position_embedding.weight"] = pos_enc.tolist()

    # Transformer layers
    for i in range(config.n_layers):
        prefix = f"layers.{i}"
        pfx_state = f"layers.{i}"

        # ---- Self-attention ----
        # CausalSelfAttention stores qkv_proj and out_proj
        qkv_w = state[f"{pfx_state}.attn.qkv_proj.weight"].numpy()  # [3*d_model, d_model]
        qkv_b = state[f"{pfx_state}.attn.qkv_proj.bias"].numpy()    # [3*d_model]

        weights[f"{prefix}.self_attn.in_proj_weight"] = qkv_w.tolist()
        weights[f"{prefix}.self_attn.in_proj_bias"] = qkv_b.tolist()

        # Output projection
        out_proj_w = state[f"{pfx_state}.attn.out_proj.weight"].numpy()  # [d_model, d_model]
        out_proj_b = state[f"{pfx_state}.attn.out_proj.bias"].numpy()    # [d_model]
        weights[f"{prefix}.self_attn.out_proj.weight"] = out_proj_w.tolist()
        weights[f"{prefix}.self_attn.out_proj.bias"] = out_proj_b.tolist()

        # ---- LayerNorm 1 (after attention) ----
        ln1_w = state[f"{pfx_state}.ln1.weight"].numpy()
        ln1_b = state[f"{pfx_state}.ln1.bias"].numpy()
        weights[f"{prefix}.norm1.weight"] = ln1_w.tolist()
        weights[f"{prefix}.norm1.bias"] = ln1_b.tolist()

        # ---- LayerNorm 2 (after FFN) ----
        ln2_w = state[f"{pfx_state}.ln2.weight"].numpy()
        ln2_b = state[f"{pfx_state}.ln2.bias"].numpy()
        weights[f"{prefix}.norm2.weight"] = ln2_w.tolist()
        weights[f"{prefix}.norm2.bias"] = ln2_b.tolist()

        # ---- Feed-forward network ----
        # FF1 (linear1): [ff_hidden, d_model]
        ff1_w = state[f"{pfx_state}.ff.0.weight"].numpy()
        ff1_b = state[f"{pfx_state}.ff.0.bias"].numpy()
        weights[f"{prefix}.linear1.weight"] = ff1_w.tolist()
        weights[f"{prefix}.linear1.bias"] = ff1_b.tolist()

        # FF2 (linear2): [d_model, ff_hidden]
        ff2_w = state[f"{pfx_state}.ff.3.weight"].numpy()
        ff2_b = state[f"{pfx_state}.ff.3.bias"].numpy()
        weights[f"{prefix}.linear2.weight"] = ff2_w.tolist()
        weights[f"{prefix}.linear2.bias"] = ff2_b.tolist()

    # Final layer norm
    fln_w = state["final_ln.weight"].numpy()
    fln_b = state["final_ln.bias"].numpy()
    weights["final_layer_norm.weight"] = fln_w.tolist()
    weights["final_layer_norm.bias"] = fln_b.tolist()

    return weights


def export(checkpoint_path: str, vocab_path: str, config_path: str,
           output_path: str, model_config: dict = None):
    """Export a trained model to Go-compatible JSON."""
    print(f"Loading checkpoint: {checkpoint_path}")

    if model_config is None:
        model_config = {
            "vocab_size": 8000,
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 6,
            "max_seq_len": 256,
            "ff_hidden": 1024,
        }

    config = TransformerConfig(**model_config)

    # Build model and load weights
    model = TransformerLM(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', '?')
        loss = ckpt.get('loss', '?')
        print(f"Loaded checkpoint: epoch={epoch}, loss={loss}")
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print("Model loaded successfully")

    # Load vocabulary
    vocab_data = {}
    if Path(vocab_path).exists():
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        print(f"Vocabulary loaded: {vocab_data.get('vocab_size', 'unknown')} tokens")

    # Map weights
    print("Mapping weights to Go format...")
    weights = map_pytorch_weights_to_go(model, config)

    # Calculate total parameters
    total_params = sum(np.prod(np.array(w).shape) for w in weights.values())

    export_data = {
        "metadata": {
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "framework": "pytorch",
            "pytorch_version": torch.__version__,
            "training_completed": True,
            "total_parameters": int(total_params),
        },
        "config": config.to_dict(),
        "vocab": vocab_data,
        "weights": weights
    }

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)

    file_size = os.path.getsize(output_path) / 1e6
    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"Output: {output_path}")
    print(f"Size: {file_size:.2f} MB")
    print(f"Parameters: {total_params:,}")
    print(f"{'='*60}")

    # Completion marker
    completion_marker = Path(output_path).parent / "training_output"
    completion_marker.mkdir(parents=True, exist_ok=True)
    with open(completion_marker / "export_complete.txt", 'w') as f:
        f.write(f"Exported at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        f.write(f"Output: {output_path}\n")
        f.write(f"Size: {file_size:.2f} MB\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to Go JSON")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to PyTorch checkpoint (.pt file)")
    parser.add_argument("--vocab", type=str, default="vocab_trained.json",
                       help="Path to vocabulary JSON")
    parser.add_argument("--config", type=str, default="config.train.json",
                       help="Path to training config JSON")
    parser.add_argument("--output", type=str, default="../config.trained.json",
                       help="Output JSON path")
    args = parser.parse_args()

    # Load config if exists
    model_config = None
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            cfg = json.load(f)
            model_config = cfg.get("model", cfg)

    export(args.checkpoint, args.vocab, args.config, args.output, model_config)


if __name__ == "__main__":
    main()
