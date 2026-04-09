"""
infer.py — Run inference on the base or fine-tuned model.

Usage:
    # Test the BASE model (before fine-tuning):
    python src/infer.py

    # Test the FINE-TUNED model (after training):
    python src/infer.py --adapter

    # Custom prompt:
    python src/infer.py --adapter --prompt "My order is late, what do I do?"
"""

import argparse
import sys
import os

# Allow imports from src/ when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from utils import build_model, free_model, generate_answer

# ── Demo prompts that exercise different customer-support intents ───────────────
DEMO_PROMPTS = [
    "I want to reset my password. What should I do?",
    "My refund hasn't arrived yet. Can you help?",
    "How do I update my billing address?",
    "I never received my order. Who do I contact?",
    "Can I cancel my subscription right now?",
]


def run_demo(use_adapter: bool, custom_prompt: str | None = None) -> None:
    model_label = "FINE-TUNED MODEL" if use_adapter else "BASE MODEL (before fine-tuning)"
    print(f"\n{'='*65}")
    print(f"  {model_label}")
    print(f"{'='*65}\n")

    model, tokenizer = build_model(use_adapter=use_adapter)

    prompts = [custom_prompt] if custom_prompt else DEMO_PROMPTS

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}] USER:  {prompt}")
        answer = generate_answer(model, tokenizer, prompt)
        print(f"    ASST:  {answer}")
        print()

    free_model(model)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on base or fine-tuned Qwen2.5 model."
    )
    parser.add_argument(
        "--adapter", action="store_true",
        help="Use the fine-tuned LoRA adapter (default: base model only)"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Custom prompt to test (optional)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run BOTH base and fine-tuned and compare side-by-side"
    )
    args = parser.parse_args()

    if args.compare:
        # Run base first, free memory, then run fine-tuned
        run_demo(use_adapter=False, custom_prompt=args.prompt)
        run_demo(use_adapter=True,  custom_prompt=args.prompt)
    else:
        run_demo(use_adapter=args.adapter, custom_prompt=args.prompt)


if __name__ == "__main__":
    main()
