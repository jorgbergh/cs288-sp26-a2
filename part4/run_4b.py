#!/usr/bin/env python3
"""
Task 4B: Prompting with the 4A model (few-shot, high-quality prompts).

Prerequisites:
  - You must have run run_4a.py --full (or --quick) so that part4/checkpoints/
    contains: 4a_tokenizer.json, 4a_config.json, 4a_transformer.pt

This script:
  1. Loads the tokenizer and finetuned transformer from 4A checkpoints
  2. Runs few-shot prompting on the dev set (same order as 4A)
  3. Writes prompting_predictions.json

Usage:
  python part4/run_4b.py
  python part4/run_4b.py --checkpoint-dir part4/checkpoints --dev-file part4/fixtures/squad_dev.json
  python part4/run_4b.py --num-few-shot 3 --output part4/prompting_predictions.json
"""

import sys
import json
import argparse
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from part2.model import TransformerLM
from part3.nn_utils import softmax

from part4.prompt_templates_4b import (
    template_instruction,
    build_few_shot_prompt,
)

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
DEFAULT_DEV_FILE = FIXTURES_DIR / "squad_dev.json"
DEFAULT_TRAIN_FILE = FIXTURES_DIR / "squad_train.json"
DEFAULT_OUTPUT = Path(__file__).parent / "prompting_predictions.json"


def load_tokenizer_from_checkpoint(checkpoint_dir: Path):
    """Load tokenizer from 4a_tokenizer.json (saved by run_4a.py)."""
    from part1.tokenizer import get_tokenizer

    path = Path(checkpoint_dir) / "4a_tokenizer.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # vocab: str(id) -> list of ints (bytes)  ->  int -> bytes
    vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
    merges = [(bytes(p[0]), bytes(p[1])) for p in data["merges"]]
    special_tokens = data.get("special_tokens", ["<|endoftext|>", "<|pad|>"])

    return get_tokenizer(vocab, merges, special_tokens)


def load_transformer_from_checkpoint(checkpoint_dir: Path, device: str):
    """Load TransformerLM from 4a_config.json and 4a_transformer.pt."""
    config_path = Path(checkpoint_dir) / "4a_config.json"
    state_path = Path(checkpoint_dir) / "4a_transformer.pt"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def get_choice_token_ids(tokenizer, labels=("A", "B", "C", "D")):
    """Map each choice label to the token id used for next-token prediction."""
    choice_ids = {}
    for label in labels:
        for prefix in ["", " "]:
            token_ids = tokenizer.encode(prefix + label)
            if token_ids:
                choice_ids[label] = token_ids[-1]
                break
        if label not in choice_ids:
            choice_ids[label] = None
    return choice_ids


def predict_from_prompt(model, tokenizer, prompt: str, num_choices: int, device: str):
    """
    Run model on prompt and predict choice from next-token logits (A/B/C/D).
    Returns predicted index in 0..num_choices-1.
    """
    model.eval()
    labels = ["A", "B", "C", "D", "E", "F", "G", "H"][:num_choices]
    choice_ids = get_choice_token_ids(tokenizer, tuple(labels))

    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids)[:, -1, :]

    choice_logits = []
    for label in labels:
        tid = choice_ids.get(label)
        if tid is not None:
            choice_logits.append(logits[0, tid].item())
        else:
            choice_logits.append(float("-inf"))

    choice_logits_t = torch.tensor(choice_logits, device=device)
    pred_idx = softmax(choice_logits_t, dim=-1).argmax().item()
    return pred_idx


def run_prompting(
    model,
    tokenizer,
    dev_examples: list,
    train_examples: list,
    num_few_shot: int,
    device: str,
    template_fn=template_instruction,
):
    """
    Run few-shot prompting on dev_examples. Preserves dev order.
    Uses train_examples (or dev if no train) for few-shot examples.
    """
    # Pool of examples for few-shot (prefer train so we don't leak dev)
    few_shot_pool = train_examples if train_examples else dev_examples
    predictions = []

    for i, ex in enumerate(dev_examples):
        prompt = build_few_shot_prompt(
            test_context=ex["context"],
            test_question=ex["question"],
            test_choices=ex["choices"],
            example_examples=few_shot_pool,
            template_fn=template_fn,
            num_few_shot=num_few_shot,
        )
        pred = predict_from_prompt(
            model, tokenizer, prompt, num_choices=len(ex["choices"]), device=device
        )
        predictions.append(pred)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="CS288 Part 4B: Prompting (few-shot)")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Directory with 4a_tokenizer.json, 4a_config.json, 4a_transformer.pt",
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=DEFAULT_DEV_FILE,
        help="Dev JSON (same order as 4A; predictions match this order)",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=None,
        help="Train JSON for few-shot examples (default: squad_train.json if exists)",
    )
    parser.add_argument(
        "--num-few-shot",
        type=int,
        default=2,
        help="Number of few-shot examples in each prompt (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for prompting_predictions.json",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(args.checkpoint_dir)
    dev_path = Path(args.dev_file)

    if not (checkpoint_dir / "4a_tokenizer.json").exists():
        print(f"Checkpoint not found: {checkpoint_dir}/4a_tokenizer.json")
        print("Run run_4a.py first (without --no-save-checkpoint) to create checkpoints.")
        sys.exit(1)
    if not dev_path.exists():
        print(f"Dev file not found: {dev_path}")
        sys.exit(1)

    print("=" * 60)
    print("CS288 Part 4B: Prompting (few-shot)")
    print("=" * 60)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Dev file:      {dev_path}")
    print(f"Few-shot:      {args.num_few_shot} examples")
    print(f"Device:        {device}")
    print()

    # Load 4A tokenizer and transformer
    print("Loading tokenizer and model from 4A checkpoints...")
    tokenizer = load_tokenizer_from_checkpoint(checkpoint_dir)
    model = load_transformer_from_checkpoint(checkpoint_dir, device)
    print("Done.")
    print()

    # Load dev (and optionally train for few-shot)
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_examples = json.load(f)

    train_examples = []
    train_path = args.train_file or (FIXTURES_DIR / "squad_train.json")
    if Path(train_path).exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_examples = json.load(f)
        print(f"Using {len(train_examples)} train examples for few-shot pool.")
    else:
        print("No train file found; using dev pool for few-shot (avoid for submission).")

    # Run prompting in dev order (no shuffle)
    print(f"Running few-shot prompting on {len(dev_examples)} dev examples...")
    predictions = run_prompting(
        model,
        tokenizer,
        dev_examples,
        train_examples,
        num_few_shot=args.num_few_shot,
        device=device,
        template_fn=template_instruction,
    )

    # Accuracy (if dev has "answer" field)
    labels = [ex.get("answer", -1) for ex in dev_examples]
    valid = [(p, l) for p, l in zip(predictions, labels) if l >= 0]
    accuracy = sum(1 for p, l in valid if p == l) / len(valid) if valid else 0.0

    out = {"predictions": predictions, "accuracy": accuracy}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Saved:    {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
