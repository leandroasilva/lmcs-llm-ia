"""
GPU Training for LMCS LLM - AMD GPU Support via PyTorch
========================================================

This module provides GPU-accelerated training for the LMCS Transformer model.
It supports:
- Apple Silicon (M1/M2/M3) via MPS (Metal Performance Shaders)
- AMD GPUs on Linux via ROCm
- CPU fallback for any platform

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
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class TransformerConfig:
    """Configuration for Transformer model"""
    
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


class TransformerModel(nn.Module):
    """Transformer Language Model compatible with Go implementation"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_hidden,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Token + positional embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = self.dropout(token_embeds + position_embeds)
        
        # Create causal mask (autoregressive)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
            diagonal=1
        )
        
        # Transformer encoder
        if attention_mask is not None:
            # Convert attention mask from 0/1 to 0/-inf
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            combined_mask = attention_mask + causal_mask
        else:
            combined_mask = causal_mask
        
        # Forward through transformer
        hidden = self.transformer(x, mask=combined_mask)
        
        # Final layer norm
        hidden = self.layer_norm(hidden)
        
        # Output projection
        logits = self.output_projection(hidden)
        
        return logits


class TextDataset(Dataset):
    """Dataset for training on text sequences"""
    
    def __init__(self, text: str, max_seq_len: int = 256, stride: int = 128):
        self.tokens = text.split()
        self.max_seq_len = max_seq_len
        self.stride = stride
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - max_seq_len, stride):
            self.sequences.append(self.tokens[i:i + max_seq_len])
        
        print(f"Created {len(self.sequences)} sequences from {len(self.tokens)} tokens")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Return input and target (shifted by 1)
        return torch.tensor([hash(token) % 8000 for token in sequence[:-1]], dtype=torch.long), \
               torch.tensor([hash(token) % 8000 for token in sequence[1:]], dtype=torch.long)


class Trainer:
    """Training loop with GPU support"""
    
    def __init__(
        self,
        model: TransformerModel,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
    ):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        
        # Move model to device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training stats
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_epoch(self, dataloader: DataLoader, log_interval: int = 10) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        for batch_idx, (input_ids, targets) in enumerate(pbar):
            # Move to device
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Compute loss
            # Reshape for CrossEntropyLoss: (batch*seq, vocab)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Stats
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            if batch_idx % log_interval == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{self.learning_rate:.6f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for input_ids, targets in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            logits = self.model(input_ids)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'config': self.model.config.to_dict(),
            'global_step': self.global_step,
        }
        
        # Save current checkpoint
        torch.save(checkpoint, path)
        
        # Save best checkpoint
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        
        print(f"✓ Loaded checkpoint from epoch {self.epoch}")


def get_device(device_str: str = "auto") -> torch.device:
    """Get the best available device"""
    if device_str == "auto":
        # Try MPS (Apple Silicon) first
        if torch.backends.mps.is_available():
            print("✓ Using Apple Silicon GPU (MPS)")
            return torch.device("mps")
        
        # Try CUDA (NVIDIA)
        if torch.cuda.is_available():
            print(f"✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        
        # Fallback to CPU
        print("✓ Using CPU")
        return torch.device("cpu")
    
    return torch.device(device_str)


def export_for_go(checkpoint_path: str, output_path: str):
    """Export PyTorch model to Go-compatible format"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    
    # Convert to numpy
    export_data = {
        'config': config,
        'weights': {
            k: v.cpu().numpy().tolist()
            for k, v in state_dict.items()
        }
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✓ Exported model to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='GPU Training for LMCS LLM')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, mps, cuda, cpu')
    
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
    parser.add_argument('--data-path', type=str, default='../dataset/data/train_enriched.txt',
                       help='Path to training data')
    parser.add_argument('--vocab-size', type=int, default=8000,
                       help='Vocabulary size')
    
    # Checkpoint
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save-path', type=str, default='./checkpoints/model.pt',
                       help='Path to save checkpoints')
    parser.add_argument('--export-go', type=str, default=None,
                       help='Export model for Go (output JSON path)')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else device}")
    
    # Create config
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
    )
    
    # Create model
    model = TransformerModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {num_params / 1e6:.2f}M parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
    )
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Load data
    if os.path.exists(args.data_path):
        with open(args.data_path, 'r') as f:
            text = f.read()
    else:
        print(f"Warning: Data file not found at {args.data_path}, using sample text")
        text = "o rato roeu a roupa do rei de roma " * 1000
    
    dataset = TextDataset(text, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to >0 for faster data loading
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    print(f"Device: {device}\n")
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        trainer.epoch = epoch
        
        # Train
        train_loss = trainer.train_epoch(dataloader)
        
        # Log
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Loss: {train_loss:.4f} - "
              f"Time: {elapsed:.1f}s")
        
        # Save checkpoint
        is_best = train_loss < trainer.best_loss
        if is_best:
            trainer.best_loss = train_loss
        
        if epoch % 10 == 0 or is_best:
            trainer.save_checkpoint(args.save_path, is_best=is_best)
    
    # Export for Go if requested
    if args.export_go:
        export_for_go(args.save_path, args.export_go)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Training completed!")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Best loss: {trainer.best_loss:.4f}")
    print(f"Final epoch: {args.epochs}")
    print(f"{'='*60}")
    
    # Create completion marker file
    completion_marker = Path(args.save_path).parent / "training_complete.txt"
    with open(completion_marker, 'w') as f:
        f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs: {args.epochs}\n")
        f.write(f"Best loss: {trainer.best_loss:.4f}\n")
        f.write(f"Total time: {total_time:.1f}s\n")
    
    print(f"Completion marker saved to: {completion_marker}")


if __name__ == '__main__':
    main()
