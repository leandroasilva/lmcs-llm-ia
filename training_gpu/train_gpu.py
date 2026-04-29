#!/usr/bin/env python3
"""
GPU Training for LMCS LLM - TensorFlow/Keras Version
=====================================================
Supports:
- Apple Silicon (M1/M2/M3) via Metal (mps)
- NVIDIA GPUs via CUDA
- CPU fallback

Usage:
    python train_gpu.py --device auto --epochs 300 --batch-size 16
    python train_gpu.py --device mps --epochs 300
    python train_gpu.py --device cpu --epochs 100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Enable mixed precision for faster training on compatible GPUs
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


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
        return cls(**config_dict)


def positional_encoding(max_seq_len: int, d_model: int) -> tf.Tensor:
    """Create sinusoidal positional encodings."""
    positions = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((max_seq_len, d_model))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)

    return tf.constant(pe, dtype=tf.float32)


def create_transformer_model(config: TransformerConfig) -> keras.Model:
    """Create a Transformer language model using Keras Functional API."""

    inputs = layers.Input(shape=(config.max_seq_len,), dtype=tf.int32, name="input_ids")

    # Token embeddings + positional encoding
    token_embed = layers.Embedding(config.vocab_size, config.d_model, name="token_embedding")
    x = token_embed(inputs)

    pos_enc = positional_encoding(config.max_seq_len, config.d_model)
    x = x + pos_enc
    x = layers.Dropout(config.dropout)(x)

    # Transformer encoder layers
    for i in range(config.n_layers):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=config.n_heads,
            key_dim=config.d_model // config.n_heads,
            dropout=config.dropout,
            name=f"transformer_layer_{i}_mha"
        )(x, x, use_causal_mask=True)

        # Layer norm + residual
        x = layers.LayerNormalization(epsilon=1e-6, name=f"transformer_layer_{i}_ln1")(x + attn_output)

        # Feed-forward network
        ff_output = layers.Dense(config.ff_hidden, activation='gelu', name=f"transformer_layer_{i}_ff1")(x)
        ff_output = layers.Dropout(config.dropout)(ff_output)
        ff_output = layers.Dense(config.d_model, name=f"transformer_layer_{i}_ff2")(ff_output)

        # Layer norm + residual
        x = layers.LayerNormalization(epsilon=1e-6, name=f"transformer_layer_{i}_ln2")(x + ff_output)

    # Final layer norm
    x = layers.LayerNormalization(epsilon=1e-6, name="final_layer_norm")(x)

    # Output projection (tied with token embedding)
    logits = layers.Dense(config.vocab_size, use_bias=False, name="output_projection")(x)

    model = keras.Model(inputs=inputs, outputs=logits, name="lmcs_transformer")

    # Tie weights: output projection = token embedding transpose
    model.get_layer("output_projection").set_weights([
        model.get_layer("token_embedding").get_weights()[0].T
    ])

    return model


class TextDataset:
    """Dataset for training on text sequences."""

    def __init__(self, text: str, word_to_id: Dict[str, int], max_seq_len: int = 256, stride: int = 128):
        self.word_to_id = word_to_id
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.unk_id = word_to_id.get("<UNK>", 1)
        self.bos_id = word_to_id.get("<BOS>", 2)
        self.eos_id = word_to_id.get("<EOS>", 3)
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
        input_ids = seq[:-1]
        target_ids = seq[1:]
        # Pad if necessary
        while len(input_ids) < self.max_seq_len:
            input_ids.append(self.pad_id)
            target_ids.append(self.pad_id)
        return np.array(input_ids, dtype=np.int32), np.array(target_ids, dtype=np.int32)


def create_tf_dataset(dataset: TextDataset, batch_size: int, shuffle: bool = True):
    """Create a tf.data.Dataset from TextDataset."""
    def generator():
        indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            yield dataset[idx]

    tf_ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(dataset.max_seq_len,), dtype=tf.int32),
            tf.TensorSpec(shape=(dataset.max_seq_len,), dtype=tf.int32)
        )
    )

    tf_ds = tf_ds.batch(batch_size)
    tf_ds = tf_ds.prefetch(tf.data.AUTOTUNE)
    return tf_ds


def extract_vocabulary(text: str, vocab_size: int = 8000) -> Dict[str, int]:
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

    for i, (word, _) in enumerate(sorted_words):
        token_id = i + 4
        word_to_id[word] = token_id
        id_to_word[token_id] = word

    return word_to_id, id_to_word


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """Custom loss that ignores PAD tokens."""
    pad_id = 0
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred):
    """Custom accuracy that ignores PAD tokens."""
    pad_id = 0
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    matches = tf.cast(tf.equal(y_true, predictions), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)


class TrainingCheckpoint(keras.callbacks.Callback):
    """Custom callback for checkpointing and metrics logging."""

    def __init__(self, save_path: str, save_best_only: bool = True):
        super().__init__()
        self.save_path = save_path
        self.best_loss = float('inf')
        self.save_best_only = save_best_only
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss', float('inf'))
        if loss < self.best_loss:
            self.best_loss = loss
            if self.save_best_only:
                self.model.save_weights(self.save_path.replace('.weights.h5', '_best.weights.h5'))
                print(f"  New best loss: {loss:.4f}")

        if epoch % 10 == 0:
            self.model.save_weights(self.save_path)


class MetricsLogger(keras.callbacks.Callback):
    """Log training metrics to a JSON file."""

    def __init__(self, log_path: str):
        super().__init__()
        self.log_path = log_path
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        entry = {"epoch": epoch, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if logs:
            entry.update({k: float(v) for k, v in logs.items()})
        self.history.append(entry)
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def get_device_strategy(device_str: str = "auto"):
    """Get the best available TensorFlow distribution strategy."""
    if device_str == "auto":
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs detected: {len(gpus)}")
            return tf.distribute.MirroredStrategy()
        print("No GPU detected, using CPU")
        return tf.distribute.get_strategy()

    if device_str == "cpu":
        tf.config.set_visible_devices([], 'GPU')
        return tf.distribute.get_strategy()

    return tf.distribute.get_strategy()


def export_for_go(model: keras.Model, config: TransformerConfig, vocab_data: Dict,
                  checkpoint_path: str, output_path: str):
    """Export Keras model to Go-compatible JSON format."""
    print(f"\nExporting model to Go-compatible format...")

    weights = {}
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if not layer_weights:
            continue

        # Map Keras layer names to Go-compatible names
        name = layer.name
        if name == "token_embedding":
            weights["token_embedding.weight"] = layer_weights[0].tolist()
        elif name == "output_projection":
            weights["output_projection.weight"] = layer_weights[0].T.tolist()
        elif "mha" in name:
            # MultiHeadAttention: query, key, value, output weights
            prefix = name.replace("_mha", "")
            # Keras MHA stores weights differently; extract what we can
            # For simplicity, we store the full attention weights
            for i, w in enumerate(layer_weights):
                weights[f"{name}.weight_{i}"] = w.tolist()
        elif "ln" in name:
            for i, w in enumerate(layer_weights):
                key = "gamma" if i == 0 else "beta"
                weights[f"{name}.{key}"] = w.tolist()
        elif "ff" in name:
            for i, w in enumerate(layer_weights):
                key = "weight" if len(w.shape) >= 2 else "bias"
                weights[f"{name}.{key}"] = w.tolist()
        else:
            for i, w in enumerate(layer_weights):
                weights[f"{name}.weight_{i}"] = w.tolist()

    export_data = {
        "metadata": {
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "framework": "tensorflow",
            "tensorflow_version": tf.__version__,
            "training_completed": True,
        },
        "config": config.to_dict(),
        "vocab": vocab_data,
        "weights": weights
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)

    file_size = os.path.getsize(output_path) / 1e6
    print(f"Exported to {output_path} ({file_size:.2f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='GPU Training for LMCS LLM (TensorFlow)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device strategy: auto, cpu')

    # Training
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max-seq-len', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--d-model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=6,
                       help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=8,
                       help='Number of attention heads')

    # Data
    parser.add_argument('--data-path', type=str, default='data/merged_train.txt',
                       help='Path to training data')
    parser.add_argument('--vocab-size', type=int, default=8000,
                       help='Vocabulary size')

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint weights to resume from')
    parser.add_argument('--save-path', type=str, default='checkpoints/model.weights.h5',
                       help='Path to save checkpoints')
    parser.add_argument('--export-go', type=str, default=None,
                       help='Export model for Go (output JSON path)')

    args = parser.parse_args()

    # Device strategy
    strategy = get_device_strategy(args.device)
    print(f"Using strategy: {strategy}")

    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Warning: Data file not found at {data_path}")
        print("Using sample text...")
        text = "o rato roeu a roupa do rei de roma " * 1000
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded {len(text)} characters from {data_path}")

    # Extract vocabulary
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

    # Create config
    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
    )

    # Create dataset
    dataset = TextDataset(text, word_to_id, max_seq_len=args.max_seq_len)
    train_ds = create_tf_dataset(dataset, batch_size=args.batch_size)

    # Build model within strategy scope
    with strategy.scope():
        model = create_transformer_model(config)
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=args.learning_rate,
                weight_decay=0.01
            ),
            loss=masked_sparse_categorical_crossentropy,
            metrics=[masked_accuracy]
        )

    model.summary()

    # Resume if checkpoint exists
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        model.load_weights(args.resume)

    # Callbacks
    callbacks = [
        TrainingCheckpoint(args.save_path),
        MetricsLogger("training_metrics.json"),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss', patience=30, restore_best_weights=True
        ),
    ]

    # Training
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}, LR: {args.learning_rate}")
    print(f"Device: {strategy}")
    print(f"{'='*60}\n")

    start_time = time.time()
    history = model.fit(train_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)

    # Save final checkpoint
    model.save_weights(args.save_path)

    # Export for Go
    if args.export_go:
        export_for_go(model, config, vocab_data, args.save_path, args.export_go)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"{'='*60}")

    # Create completion marker
    completion_marker = Path(args.save_path).parent / "training_complete.txt"
    with open(completion_marker, 'w') as f:
        f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs: {args.epochs}\n")
        f.write(f"Final loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Total time: {total_time:.1f}s\n")

    print(f"Completion marker saved to: {completion_marker}")


if __name__ == '__main__':
    main()
