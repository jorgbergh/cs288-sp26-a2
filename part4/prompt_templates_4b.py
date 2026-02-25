"""
Task 4B: High-quality prompt templates and few-shot formatting.

Use these to build prompts so the model sees a clear task and (optionally)
a few examples before the question to answer. Predictions are made by
comparing next-token logits for choice labels (A, B, C, D).
"""

from typing import List, Dict, Any


# -----------------------------------------------------------------------------
# Prompt templates (single question, no few-shot)
# -----------------------------------------------------------------------------

def format_choices(choices: List[str], labels: List[str] = None) -> str:
    """Format choices as 'A. first choice' etc."""
    if labels is None:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"][: len(choices)]
    return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))


def template_instruction(context: str, question: str, choices: List[str]) -> str:
    """Clear instruction-style prompt. Good for zero-shot."""
    choices_text = format_choices(choices)
    return (
        f"Read the passage and choose the best answer.\n\n"
        f"Passage:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choices_text}\n\n"
        f"The answer is"
    )


def template_simple(context: str, question: str, choices: List[str]) -> str:
    """Minimal prompt. Often works well with few-shot."""
    choices_text = format_choices(choices)
    return f"Passage:\n{context}\n\nQuestion: {question}\n\n{choices_text}\n\nAnswer:"


def template_qa(context: str, question: str, choices: List[str]) -> str:
    """Q&A style with explicit 'Answer:' at the end (model completes with A/B/C/D)."""
    choices_text = format_choices(choices)
    return (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"{choices_text}\n\n"
        f"Answer:"
    )


# -----------------------------------------------------------------------------
# Few-shot: format one example (context, question, choices, correct index)
# -----------------------------------------------------------------------------

def format_one_example(
    context: str,
    question: str,
    choices: List[str],
    answer_index: int,
    template_fn,
    labels: List[str] = None,
) -> str:
    """Format a single complete example ending with the correct answer letter."""
    prompt_part = template_fn(context, question, choices).strip()
    if labels is None:
        labels = ["A", "B", "C", "D"]
    answer_letter = labels[answer_index] if answer_index >= 0 else "?"
    return f"{prompt_part} {answer_letter}"


# -----------------------------------------------------------------------------
# Build full prompt: optional few-shot examples + test question (no answer)
# -----------------------------------------------------------------------------

def build_few_shot_prompt(
    test_context: str,
    test_question: str,
    test_choices: List[str],
    example_examples: List[Dict[str, Any]],
    template_fn=template_instruction,
    num_few_shot: int = 2,
) -> str:
    """
    Build a prompt with up to num_few_shot examples, then the test question.

    Each item in example_examples must have: context, question, choices, answer (index).
    The test question is formatted with the same template but without the answer,
    so the model predicts the next token (A/B/C/D).
    """
    parts = []

    # Few-shot examples (complete with answer)
    for ex in example_examples[:num_few_shot]:
        ctx = ex["context"]
        q = ex["question"]
        ch = ex["choices"]
        ans = ex.get("answer", 0)
        one = format_one_example(ctx, q, ch, ans, template_fn)
        parts.append(one)

    # Test question (no answer)
    test_part = template_fn(test_context, test_question, test_choices)
    parts.append(test_part)

    return "\n\n".join(parts)


def get_default_template():
    """Default template for 4B (instruction-style, works well with few-shot)."""
    return template_instruction
