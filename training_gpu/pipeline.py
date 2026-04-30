#!/usr/bin/env python3
"""
LMCS LLM Training Pipeline
==========================
Orchestrates the full training workflow:
1. Download datasets
2. Merge and normalize
3. Train model with PyTorch (GPU)
4. Export to Go-compatible JSON

Usage:
    python pipeline.py
    python pipeline.py --skip-download --skip-merge
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list, cwd: Path = None):
    """Run a pipeline step with error handling."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd or Path(__file__).parent)
    if result.returncode != 0:
        print(f"ERROR: Step '{name}' failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"Step '{name}' completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="LMCS LLM Training Pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-merge", action="store_true", help="Skip dataset merge")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-export", action="store_true", help="Skip export to Go")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, mps, cuda, cpu")
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # 1. Download datasets
    if not args.skip_download:
        run_step(
            "Download Datasets",
            [sys.executable, "download_datasets.py"],
            cwd=base_dir,
        )

    # 2. Merge datasets
    if not args.skip_merge:
        run_step(
            "Merge Datasets",
            [sys.executable, "merge_datasets.py", "--split"],
            cwd=base_dir,
        )

    # 3. Train model
    if not args.skip_train:
        run_step(
            "Train Model",
            [
                sys.executable,
                "train_gpu.py",
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--device", args.device,
                "--data-path", "data/train.txt",
                "--save-path", "checkpoints/model.pt",
            ],
            cwd=base_dir,
        )

    # 4. Export to Go
    if not args.skip_export:
        # Find best checkpoint
        checkpoint = base_dir / "checkpoints" / "model_best.pt"
        if not checkpoint.exists():
            checkpoint = base_dir / "checkpoints" / "model.pt"

        if checkpoint.exists():
            run_step(
                "Export to Go",
                [
                    sys.executable,
                    "export_to_go.py",
                    "--checkpoint", str(checkpoint),
                    "--vocab", "vocab_trained.json",
                    "--config", "config.train.json",
                    "--output", "../config.trained.json",
                ],
                cwd=base_dir,
            )
        else:
            print("WARNING: No checkpoint found to export.")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print("Next steps:")
    print("  1. Copy config.trained.json to project root")
    print("  2. Run: ./lmcs-llm serve --config config.trained.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
