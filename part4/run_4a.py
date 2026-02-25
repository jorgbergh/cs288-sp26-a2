#!/usr/bin/env python3
"""
Task 4A: Pre-train + Fine-tune pipeline.

Runs:
  1. Train BPE tokenizer on TinyStories (part1)
  2. Pre-train Transformer LM on TinyStories (part2 + part3)
  3. Fine-tune on multiple-choice QA with pooled head (part4)
  4. Run inference on dev set and write finetuned_predictions.json

IMPORTANT (submission order):
  - Prediction labels must be in the SAME order as the input dev file
    (e.g. squad_dev.json). This script uses shuffle=False at inference
    so output order matches the dev JSON row order.
  - Do NOT run setup_datasets.py before generating submission predictions:
    it overwrites squad_dev.json with a shuffled version. Use the
    existing fixtures/squad_dev.json so order matches the grader.

Usage:
  python part4/run_4a.py --quick          # Small data, fast run
  python part4/run_4a.py --full            # Full TinyStories + SQuAD (default)
  python part4/run_4a.py --output path.json
  python part4/run_4a.py --dev-file part4/fixtures/squad_dev.json  # Explicit dev for submission
"""

import sys
import json
import argparse
import torch
from pathlib import Path

# Repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from part4.evaluate_models import (
    get_config,
    train_tokenizer,
    pretrain_model,
    finetune_qa_model,
)
from part4.qa_model import evaluate_qa_model
from part4.datasets import create_qa_dataloader

# Default output path (same dir as script)
FIXTURES_DIR = Path(__file__).parent / "fixtures"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
DEFAULT_DEV_FILE = FIXTURES_DIR / "squad_dev.json"
DEFAULT_OUTPUT = Path(__file__).parent / "finetuned_predictions.json"


def run_inference_and_save(
    qa_model,
    tokenizer,
    config: dict,
    dev_path: Path,
    output_path: Path,
    device: str = "cpu",
) -> dict:
    """
    Run inference on dev set (no shuffling) and save predictions in dev order.
    """
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    dev_dataloader = create_qa_dataloader(
        data=dev_data,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        num_choices=4,
        shuffle=False,  # Critical: preserve order for submission
    )

    results = evaluate_qa_model(qa_model, dev_dataloader, device)
    predictions = results["predictions"]
    accuracy = results["accuracy"]

    out = {
        "predictions": predictions,
        "accuracy": accuracy,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return results


def save_4a_checkpoint(qa_model, tokenizer, config: dict, checkpoint_dir: Path) -> None:
    """
    Save tokenizer and finetuned transformer so 4B can load them.
    Writes: 4a_tokenizer.json, 4a_config.json, 4a_transformer.pt
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer: vocab id -> bytes, merges list of (bytes, bytes), special_tokens
    vocab = tokenizer.vocab
    vocab_serializable = {str(k): list(v) for k, v in vocab.items()}
    merges_serializable = [[list(p[0]), list(p[1])] for p in tokenizer.merges]
    tokenizer_payload = {
        "vocab": vocab_serializable,
        "merges": merges_serializable,
        "special_tokens": tokenizer.special_tokens,
    }
    with open(checkpoint_dir / "4a_tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_payload, f, indent=0)

    # Model config (needed to rebuild TransformerLM)
    model_config = {
        "vocab_size": len(vocab),
        "context_length": config["context_length"],
        "d_model": config["d_model"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "d_ff": config["d_ff"],
    }
    with open(checkpoint_dir / "4a_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    # Finetuned transformer (backbone only; 4B uses it for prompting)
    transformer = qa_model.transformer
    torch.save(transformer.state_dict(), checkpoint_dir / "4a_transformer.pt")
    print(f"Checkpoints saved to {checkpoint_dir} (for run_4b.py)")


def main():
    parser = argparse.ArgumentParser(description="CS288 Part 4A: Pre-train + Fine-tune")
    parser.add_argument("--quick", action="store_true", help="Use smaller datasets for quick testing")
    parser.add_argument("--full", action="store_true", help="Use full TinyStories + SQuAD (default)")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path for finetuned_predictions.json (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=None,
        help="Dev JSON for inference order (default: from config, e.g. squad_dev.json for submission)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Directory to save tokenizer + transformer for 4B (default: part4/checkpoints)",
    )
    parser.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="Do not save checkpoint (skip if you do not plan to run 4B)",
    )
    args = parser.parse_args()

    mode = "quick" if args.quick else "full"
    config = get_config(mode)

    # Dev file for inference: explicit override, or from config (must exist)
    dev_path = args.dev_file
    if dev_path is None:
        dev_path = config["qa_dev"]
    dev_path = Path(dev_path)
    if not dev_path.exists():
        print(f"Dev file not found: {dev_path}")
        print("Use --dev-file or ensure squad_dev.json/qa_dev.json exists in part4/fixtures.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("CS288 Part 4A: Pre-train + Fine-tune")
    print("=" * 60)
    print(f"Mode: {mode.upper()}")
    print(f"Device: {device}")
    print(f"Dev set (order preserved): {dev_path}")
    print(f"Output: {args.output}")
    print()

    # Data checks
    if not config["pretrain_data"].exists():
        print(f"Pretrain data not found: {config['pretrain_data']}")
        print("Use --quick for bundled small data, or run part4/setup_datasets.py for full.")
        sys.exit(1)
    if not config["qa_train"].exists():
        print(f"QA train data not found: {config['qa_train']}")
        print("Use --quick for bundled qa_train.json, or run part4/setup_datasets.py.")
        sys.exit(1)

    # 1. Train tokenizer
    tokenizer, vocab, merges = train_tokenizer(config)

    # 2. Pre-train LM
    pretrained_model = pretrain_model(tokenizer, config, device)

    # 3. Fine-tune QA
    qa_model = finetune_qa_model(pretrained_model, tokenizer, config, device)

    # 4. Inference on dev (no shuffle) and save
    print("=" * 60)
    print("Step 5: Inference on Dev (order preserved) → finetuned_predictions.json")
    print("=" * 60)
    results = run_inference_and_save(
        qa_model, tokenizer, config, dev_path, args.output, device
    )
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Saved: {args.output}")

    if not args.no_save_checkpoint:
        save_4a_checkpoint(qa_model, tokenizer, config, args.checkpoint_dir)

    print("Done.")


if __name__ == "__main__":
    main()
