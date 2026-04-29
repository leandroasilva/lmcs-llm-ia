#!/usr/bin/env python3
"""
Export TensorFlow/Keras Model to Go-Compatible JSON
===================================================
Converts a trained Keras model checkpoint into a JSON format
that the Go server can load and serve.

Usage:
    python export_to_go.py --checkpoint checkpoints/model_best.weights.h5 \
                           --vocab vocab_trained.json \
                           --config config.train.json \
                           --output ../config.trained.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import training module for model creation
sys.path.insert(0, str(Path(__file__).parent))
from train_gpu import TransformerConfig, create_transformer_model


def positional_encoding(max_seq_len: int, d_model: int) -> np.ndarray:
    """Create sinusoidal positional encodings matching the training script."""
    positions = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((max_seq_len, d_model))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    return pe


def map_keras_weights_to_go(model: keras.Model, config: TransformerConfig) -> dict:
    """Extract and rename Keras weights to match Go loader expectations."""
    weights = {}

    for layer in model.layers:
        layer_weights = layer.get_weights()
        if not layer_weights:
            continue

        name = layer.name
        w_list = layer_weights

        if name == "token_embedding":
            # [vocab_size, d_model]
            weights["token_embedding.weight"] = w_list[0].tolist()

        elif name == "output_projection":
            # [d_model, vocab_size] - tied with token embedding transpose
            # Go model clones token_embedding to WOut, so we don't need to export this
            pass

        elif "mha" in name:
            # Keras MultiHeadAttention stores: query, key, value, output weights/biases
            # Extract layer index from name like "transformer_layer_0_mha"
            layer_idx = name.replace("transformer_layer_", "").replace("_mha", "")
            prefix = f"transformer.layers.{layer_idx}"
            # Keras MHA weights order: Wq, bq, Wk, bk, Wv, bv, Wo, bo
            if len(w_list) >= 6:
                # Concatenate Q, K, V weights for in_proj
                wq, wk, wv = w_list[0], w_list[2], w_list[4]
                bq, bk, bv = w_list[1], w_list[3], w_list[5]

                # Keras stores as [d_model, d_model] each; Go expects [3*d_model, d_model]
                in_proj_w = np.concatenate([wq, wk, wv], axis=0)
                in_proj_b = np.concatenate([bq, bk, bv], axis=0)

                weights[f"{prefix}.self_attn.in_proj_weight"] = in_proj_w.tolist()
                weights[f"{prefix}.self_attn.in_proj_bias"] = in_proj_b.tolist()

            if len(w_list) >= 8:
                wo, bo = w_list[6], w_list[7]
                weights[f"{prefix}.self_attn.out_proj.weight"] = wo.tolist()
                weights[f"{prefix}.self_attn.out_proj.bias"] = bo.tolist()

        elif "ln1" in name:
            # Extract layer index from name like "transformer_layer_0_ln1"
            layer_idx = name.replace("transformer_layer_", "").replace("_ln1", "")
            prefix = f"transformer.layers.{layer_idx}"
            # LayerNorm: gamma (weight), beta (bias)
            if len(w_list) >= 2:
                weights[f"{prefix}.norm1.weight"] = w_list[0].tolist()
                weights[f"{prefix}.norm1.bias"] = w_list[1].tolist()

        elif "ln2" in name:
            layer_idx = name.replace("transformer_layer_", "").replace("_ln2", "")
            prefix = f"transformer.layers.{layer_idx}"
            if len(w_list) >= 2:
                weights[f"{prefix}.norm2.weight"] = w_list[0].tolist()
                weights[f"{prefix}.norm2.bias"] = w_list[1].tolist()

        elif "ff1" in name:
            layer_idx = name.replace("transformer_layer_", "").replace("_ff1", "")
            prefix = f"transformer.layers.{layer_idx}"
            # Dense layer: Keras stores [input_dim, output_dim] = [d_model, ff_hidden]
            # Go expects [ff_hidden, d_model]
            weights[f"{prefix}.linear1.weight"] = w_list[0].T.tolist()
            if len(w_list) > 1:
                weights[f"{prefix}.linear1.bias"] = w_list[1].tolist()

        elif "ff2" in name:
            layer_idx = name.replace("transformer_layer_", "").replace("_ff2", "")
            prefix = f"transformer.layers.{layer_idx}"
            weights[f"{prefix}.linear2.weight"] = w_list[0].T.tolist()
            if len(w_list) > 1:
                weights[f"{prefix}.linear2.bias"] = w_list[1].tolist()

    # Position embedding is not a trainable layer in Keras (computed as constant)
    # Generate it explicitly for Go compatibility
    pos_enc = positional_encoding(config.max_seq_len, config.d_model)
    weights["position_embedding.weight"] = pos_enc.tolist()

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
    model = create_transformer_model(config)
    model.build(input_shape=(None, config.max_seq_len))
    model.load_weights(checkpoint_path)

    print("Model loaded successfully")

    # Load vocabulary
    vocab_data = {}
    if Path(vocab_path).exists():
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        print(f"Vocabulary loaded: {vocab_data.get('vocab_size', 'unknown')} tokens")

    # Map weights
    print("Mapping weights to Go format...")
    weights = map_keras_weights_to_go(model, config)

    # Calculate total parameters
    total_params = sum(np.prod(np.array(w).shape) for w in weights.values())

    export_data = {
        "metadata": {
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "framework": "tensorflow",
            "tensorflow_version": tf.__version__,
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

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export TF model to Go JSON")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to Keras weights file")
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
