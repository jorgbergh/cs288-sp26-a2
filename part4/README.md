# Part 4: Pre-train + Fine-tune + Prompting (Bonus)

## Task 4A: Pre-train + Fine-tune

**Run the 4A pipeline (tokenizer → pretrain → finetune → predictions):**

```bash
# From repo root
python part4/run_4a.py --quick    # Small data, fast (uses part1/fixtures + part4/fixtures)
python part4/run_4a.py --full    # Full TinyStories + SQuAD (needs setup_datasets for pretrain/QA train)
```

Output: `part4/finetuned_predictions.json` with `predictions` (list of answer indices) and `accuracy`.

### Submission order (important)

- Prediction labels **must** be in the **same order** as the input dev file (e.g. `squad_dev.json`). Do **not** shuffle at inference time.
- `run_4a.py` uses `shuffle=False` on the dev dataloader so the written predictions match the dev JSON row order.
- **Do not run `setup_datasets.py`** before generating submission predictions: it overwrites `squad_dev.json` with a shuffled version. Use the existing `fixtures/squad_dev.json` so order matches the grader. (SQuAD download in setup_datasets is commented out; data is already in fixtures.)

### Options

- `--output path.json` — where to write `finetuned_predictions.json`
- `--dev-file path.json` — dev set used for inference (default: from config; use `fixtures/squad_dev.json` for submission)
- `--checkpoint-dir dir` — where to save tokenizer + transformer for 4B (default: `part4/checkpoints`)
- `--no-save-checkpoint` — skip saving checkpoints (use if you do not plan to run 4B)

## Task 4B: Prompting

**Prerequisite:** Run 4A first so that `part4/checkpoints/` contains `4a_tokenizer.json`, `4a_config.json`, and `4a_transformer.pt`.

**Run 4B (few-shot prompting, same dev order as 4A):**

```bash
python part4/run_4b.py
python part4/run_4b.py --num-few-shot 3 --output part4/prompting_predictions.json
```

- Loads the 4A tokenizer and finetuned transformer from `part4/checkpoints/`.
- Builds prompts with **few-shot examples** (default 2) from the train set and an **instruction-style** template.
- Predicts by next-token logits for A/B/C/D; writes `prompting_predictions.json` in the **same order** as the dev file.

### Options

- `--checkpoint-dir dir` — directory with 4A checkpoints (default: `part4/checkpoints`)
- `--dev-file path.json` — dev JSON (same file as 4A for submission order)
- `--train-file path.json` — JSON for few-shot example pool (default: `part4/fixtures/squad_train.json` if exists)
- `--num-few-shot N` — number of examples per prompt (default: 2)
- `--output path.json` — output path for `prompting_predictions.json`

### Customizing prompts

Edit `part4/prompt_templates_4b.py` to change the template (e.g. `template_instruction`, `template_simple`, `template_qa`) or use a custom template in `run_4b.py` (e.g. pass `template_fn=template_simple` to `run_prompting`).

## Grading

- **4A**: Score scales from 30% accuracy (0 pts) to 50%+ (full pts).
- **4B**: Must outperform 4A by at least 2% (PDF says 2%; grade_submissions.py uses 4% for full score — confirm with course materials).
- Backbone must be your part1–3 implementation; prompting must be based on the 4A (pretrained/finetuned) model.

## Data

- **Quick**: Uses `part1/fixtures/tinystories_sample_5M.txt`, `part4/fixtures/qa_train.json`, `part4/fixtures/qa_dev.json` (small, in repo).
- **Full**: Uses `part4/fixtures/tinystories_100k.txt` or `tinystories_full.txt`, `squad_train.json`, `squad_dev.json` (run `python part4/setup_datasets.py` to download TinyStories and generate SQuAD splits; do **not** use the rewritten squad_dev for submission — use the original fixtures copy).
