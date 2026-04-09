"""
evaluate.py — Industry-standard evaluation of base vs fine-tuned model.

Metrics produced (same used at Google, Meta, HuggingFace for LLM evals):
  - ROUGE-1, ROUGE-2, ROUGE-L  (n-gram overlap — standard for generation)
  - BLEU                        (precision-based n-gram; translation/generation)
  - BERTScore                   (embedding similarity — better than ROUGE for semantics)
  - Exact Match                 (strict string equality after normalisation)
  - Response Length Stats        (verbosity analysis)
  - Improvement Delta            (fine-tuned minus baseline for each metric)

MEMORY DESIGN: Models are loaded ONE AT A TIME.
  Step 1 → load base model → generate all predictions → free GPU memory
  Step 2 → load fine-tuned  → generate all predictions → free GPU memory
  Step 3 → compute all metrics on CPU (no GPU needed)
  Step 4 → save results to JSON + CSV → call report.py to generate charts

Usage:
    python src/evaluate.py
    python src/evaluate.py --samples 100   # evaluate on 100 test examples
"""

import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_from_disk

# Lazy import — only needed after predictions are generated
import evaluate as hf_evaluate

from config import (
    ADAPTER_DIR,
    DATA_RAW,
    EVAL_DIR,
    EVAL_SAMPLE_SIZE,
    MAX_NEW_TOKENS,
    SYSTEM_PROMPT,
)
from utils import build_model, free_model, generate_answer


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace — for Exact Match."""
    return re.sub(r"\s+", " ", text.lower().strip())


def compute_perplexity_approx(predictions: list[str], references: list[str]) -> float:
    """
    Approximate perplexity from ROUGE-1 recall.
    True token-level perplexity requires a second forward pass through the model.
    This gives a useful proxy without loading the model again.
    """
    # Lower ROUGE recall = more "surprised" model = higher approximate perplexity
    avg_rouge1_recall = sum(
        len(set(p.split()) & set(r.split())) / max(len(r.split()), 1)
        for p, r in zip(predictions, references)
    ) / len(predictions)
    # Map recall [0,1] → perplexity-like score [1, ∞]
    return 1.0 / max(avg_rouge1_recall, 1e-6)


# ── Prediction generation ──────────────────────────────────────────────────────

def generate_predictions(
    dataset,
    use_adapter: bool,
    max_new_tokens: int,
    label: str,
) -> tuple[list[str], float]:
    """
    Load a model, generate predictions for all examples, free the model.

    Returns:
        (list of predictions, average generation time in seconds)
    """
    print(f"\n[{label}] Loading model (use_adapter={use_adapter})...")
    model, tokenizer = build_model(use_adapter=use_adapter)

    predictions = []
    total_time  = 0.0

    for i, ex in enumerate(dataset):
        t0  = time.perf_counter()
        out = generate_answer(model, tokenizer, ex["instruction"],
                              max_new_tokens=max_new_tokens,
                              system_prompt=SYSTEM_PROMPT)
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        predictions.append(out)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{label}] {i+1}/{len(dataset)}  ({elapsed:.1f}s/example)")

    avg_time = total_time / len(dataset)
    print(f"  [{label}] Done. Avg generation: {avg_time:.2f}s/example")

    free_model(model)
    return predictions, avg_time


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_metrics(
    predictions: list[str],
    references:  list[str],
) -> dict:
    """
    Compute all evaluation metrics for one set of predictions.

    Returns a dict with ROUGE, BLEU, BERTScore, Exact Match, and length stats.
    """
    rouge   = hf_evaluate.load("rouge")
    bleu    = hf_evaluate.load("bleu")

    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_scores  = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )

    # BERTScore — uses DeBERTa embeddings; best semantic similarity metric available
    try:
        bertscore = hf_evaluate.load("bertscore")
        bs = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
        )
        bert_f1 = sum(bs["f1"]) / len(bs["f1"])
    except Exception:
        # BERTScore can fail if model is not downloaded; fall back gracefully
        bert_f1 = None

    exact_match = sum(
        normalize_text(p) == normalize_text(r)
        for p, r in zip(predictions, references)
    ) / len(predictions)

    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths  = [len(r.split()) for r in references]

    return {
        "rouge1":           round(rouge_scores["rouge1"],  4),
        "rouge2":           round(rouge_scores["rouge2"],  4),
        "rougeL":           round(rouge_scores["rougeL"],  4),
        "rougeLsum":        round(rouge_scores["rougeLsum"], 4),
        "bleu":             round(bleu_scores["bleu"],      4),
        "bertscore_f1":     round(bert_f1, 4) if bert_f1 is not None else "N/A",
        "exact_match":      round(exact_match, 4),
        "approx_perplexity":round(compute_perplexity_approx(predictions, references), 2),
        "avg_pred_length_words": round(sum(pred_lengths) / len(pred_lengths), 1),
        "avg_ref_length_words":  round(sum(ref_lengths)  / len(ref_lengths),  1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned model.")
    parser.add_argument("--samples", type=int, default=EVAL_SAMPLE_SIZE,
                        help=f"Number of test examples to evaluate (default: {EVAL_SAMPLE_SIZE})")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    if not os.path.isdir(ADAPTER_DIR) or not any(True for _ in os.scandir(ADAPTER_DIR)):
        print(f"\nERROR: No adapter found at '{ADAPTER_DIR}'.")
        print("Run 'python src/train_qlora.py' first, then evaluate.\n")
        sys.exit(1)

    os.makedirs(EVAL_DIR, exist_ok=True)

    # ── Load test data ─────────────────────────────────────────────────────────
    print(f"\nLoading {args.samples} test examples...")
    ds_raw = load_from_disk(DATA_RAW)["test"]
    n = min(args.samples, len(ds_raw))
    ds_test = ds_raw.select(range(n))
    references = [ex["response"] for ex in ds_test]
    print(f"  Loaded {n} examples from test split.")

    # ── Phase 1: Base model predictions ───────────────────────────────────────
    base_preds, base_time = generate_predictions(
        ds_test, use_adapter=False,
        max_new_tokens=args.max_new_tokens,
        label="BASE",
    )

    # ── Phase 2: Fine-tuned model predictions ─────────────────────────────────
    ft_preds, ft_time = generate_predictions(
        ds_test, use_adapter=True,
        max_new_tokens=args.max_new_tokens,
        label="FINE-TUNED",
    )

    # ── Phase 3: Compute metrics ───────────────────────────────────────────────
    print("\nComputing metrics (CPU — this is fast)...")
    base_metrics = compute_metrics(base_preds, references)
    ft_metrics   = compute_metrics(ft_preds,   references)

    # Improvement deltas (fine-tuned minus base)
    delta = {}
    for k in base_metrics:
        try:
            delta[k] = round(ft_metrics[k] - base_metrics[k], 4)
        except (TypeError, ValueError):
            delta[k] = "N/A"

    # Generation latency
    base_metrics["avg_generation_time_sec"] = round(base_time, 3)
    ft_metrics["avg_generation_time_sec"]   = round(ft_time,   3)

    # ── Phase 4: Save outputs ──────────────────────────────────────────────────

    # 1) Full predictions CSV (prompt / reference / base_answer / ft_answer)
    df = pd.DataFrame({
        "prompt":    [ex["instruction"] for ex in ds_test],
        "reference": references,
        "baseline":  base_preds,
        "fine_tuned":ft_preds,
        "intent":    [ex.get("intent", "") for ex in ds_test],
        "category":  [ex.get("category", "") for ex in ds_test],
    })
    csv_path = os.path.join(EVAL_DIR, "predictions.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nPredictions saved to: {csv_path}")

    # 2) Metrics JSON
    results = {
        "eval_samples":   n,
        "base_model":     base_metrics,
        "fine_tuned":     ft_metrics,
        "improvement":    delta,
    }
    json_path = os.path.join(EVAL_DIR, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {json_path}")

    # ── Phase 5: Print summary to console ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    header = f"{'Metric':<28} {'Baseline':>10} {'Fine-Tuned':>12} {'Delta':>9}"
    print(header)
    print("-" * 65)

    metric_labels = {
        "rouge1":           "ROUGE-1",
        "rouge2":           "ROUGE-2",
        "rougeL":           "ROUGE-L",
        "bleu":             "BLEU",
        "bertscore_f1":     "BERTScore F1",
        "exact_match":      "Exact Match",
        "approx_perplexity":"Approx Perplexity (low=better)",
        "avg_pred_length_words": "Avg Response Length (words)",
        "avg_generation_time_sec": "Avg Generation Time (sec)",
    }

    for key, label in metric_labels.items():
        b = base_metrics.get(key, "N/A")
        f = ft_metrics.get(key,   "N/A")
        d = delta.get(key, "N/A")
        try:
            sign = "+" if float(d) >= 0 else ""
            print(f"  {label:<26} {b:>10} {f:>12} {sign}{d:>8}")
        except (TypeError, ValueError):
            print(f"  {label:<26} {b:>10} {f:>12} {'N/A':>9}")

    print("=" * 65)
    print("\nNext step: run 'python src/report.py' to generate charts and the")
    print("full HTML/Markdown evaluation report.")


if __name__ == "__main__":
    main()
