"""
report.py — Generate evaluation report with charts, tables, and analysis.

Reads:  outputs/eval/metrics.json
        outputs/eval/predictions.csv
        outputs/eval/training_log.json  (optional, from training)

Writes: outputs/eval/report.md           (full Markdown report)
        outputs/eval/plots/              (all PNG charts)
          - training_loss_curve.png
          - metrics_comparison_bar.png
          - rouge_breakdown.png
          - response_length_dist.png
          - per_intent_rouge.png
          - improvement_heatmap.png

Usage:
    python src/report.py
"""

import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from config import EVAL_DIR, MODEL_ID, ADAPTER_DIR

# ── Check for matplotlib early ───────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (no GUI needed)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("[WARN] matplotlib or numpy not installed. Charts will be skipped.")
    print("       Run: pip install matplotlib numpy")


PLOTS_DIR = os.path.join(EVAL_DIR, "plots")


# ── Colour palette ────────────────────────────────────────────────────────────
BLUE    = "#2563EB"
GREEN   = "#16A34A"
ORANGE  = "#EA580C"
PURPLE  = "#7C3AED"
RED     = "#DC2626"
GREY    = "#6B7280"
BG      = "#F8FAFC"

PLT_STYLE = {
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    "#CBD5E1",
    "axes.grid":         True,
    "grid.color":        "#E2E8F0",
    "grid.linewidth":    0.8,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


def load_data() -> tuple[dict, pd.DataFrame, list]:
    metrics_path = os.path.join(EVAL_DIR, "metrics.json")
    preds_path   = os.path.join(EVAL_DIR, "predictions.csv")
    log_path     = os.path.join(EVAL_DIR, "training_log.json")

    if not os.path.exists(metrics_path):
        print(f"\nERROR: {metrics_path} not found.")
        print("Run 'python src/run_eval.py' first.\n")
        sys.exit(1)

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    df = pd.read_csv(preds_path) if os.path.exists(preds_path) else pd.DataFrame()

    training_log = []
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            training_log = json.load(f)

    return metrics, df, training_log


# ── Plot generators ───────────────────────────────────────────────────────────

def plot_training_loss(training_log: list, out_dir: str) -> str | None:
    """Training loss curve — shows model learning progress."""
    train_steps = [e["step"] for e in training_log if "loss" in e]
    train_loss  = [e["loss"] for e in training_log if "loss" in e]
    eval_steps  = [e["step"] for e in training_log if "eval_loss" in e]
    eval_loss   = [e["eval_loss"] for e in training_log if "eval_loss" in e]

    if not train_steps:
        return None

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_steps, train_loss,  color=BLUE,   linewidth=2,   label="Train Loss",      marker="o", markersize=3)
        if eval_steps:
            ax.plot(eval_steps, eval_loss, color=ORANGE, linewidth=2.5, label="Validation Loss", marker="s", markersize=5)
        ax.set_xlabel("Training Step",  fontsize=12)
        ax.set_ylabel("Loss",           fontsize=12)
        ax.set_title("Training & Validation Loss Curve\n(lower = model is learning better)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        path = os.path.join(out_dir, "training_loss_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


def plot_metrics_comparison(metrics: dict, out_dir: str) -> str:
    """Bar chart comparing all key metrics between base and fine-tuned."""
    metric_keys   = ["rouge1", "rouge2", "rougeL", "bleu", "exact_match"]
    metric_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "Exact Match"]

    base_vals = [metrics["base_model"].get(k, 0) or 0 for k in metric_keys]
    ft_vals   = [metrics["fine_tuned"].get(k, 0)  or 0 for k in metric_keys]

    x = np.arange(len(metric_labels))
    width = 0.35

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(11, 6))
        bars1 = ax.bar(x - width/2, base_vals, width, label="Base Model",      color=GREY,  alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width/2, ft_vals,   width, label="Fine-Tuned Model", color=BLUE,  alpha=0.95, edgecolor="white")

        # Value labels on bars
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=8.5, color=GREY)
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=8.5, color=BLUE, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylabel("Score (higher = better)", fontsize=12)
        ax.set_title("Base Model vs Fine-Tuned Model\nEvaluation Metrics Comparison", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_ylim(0, min(1.1, max(max(base_vals), max(ft_vals)) * 1.25))
        plt.tight_layout()
        path = os.path.join(out_dir, "metrics_comparison_bar.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


def plot_rouge_breakdown(metrics: dict, out_dir: str) -> str:
    """Radar/spider chart showing all ROUGE sub-scores."""
    labels   = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"]
    keys     = ["rouge1",  "rouge2",  "rougeL",  "rougeLsum"]
    base_v   = [metrics["base_model"].get(k, 0) or 0 for k in keys]
    ft_v     = [metrics["fine_tuned"].get(k, 0)  or 0 for k in keys]

    angles  = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    base_v  += base_v[:1];  base_v_arr = base_v
    ft_v    += ft_v[:1];    ft_v_arr   = ft_v
    angles  += angles[:1]

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.plot(angles, base_v_arr, "o-",  color=GREY,  linewidth=2,   label="Base Model")
        ax.fill(angles, base_v_arr,        color=GREY,  alpha=0.15)
        ax.plot(angles, ft_v_arr,   "o-",  color=BLUE,  linewidth=2.5, label="Fine-Tuned")
        ax.fill(angles, ft_v_arr,          color=BLUE,  alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
        ax.set_ylim(0, max(max(base_v_arr), max(ft_v_arr)) * 1.2)
        ax.set_title("ROUGE Score Breakdown\n(Radar Chart)", fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
        plt.tight_layout()
        path = os.path.join(out_dir, "rouge_breakdown.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


def plot_response_length(df: pd.DataFrame, out_dir: str) -> str:
    """Histogram comparing response word lengths."""
    base_lens = df["baseline"].dropna().apply(lambda x: len(str(x).split()))
    ft_lens   = df["fine_tuned"].dropna().apply(lambda x: len(str(x).split()))
    ref_lens  = df["reference"].dropna().apply(lambda x: len(str(x).split()))

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.linspace(0, max(base_lens.max(), ft_lens.max(), ref_lens.max()) + 5, 35)
        ax.hist(ref_lens,  bins=bins, alpha=0.55, color=GREEN,  label="Reference (ground truth)", edgecolor="white")
        ax.hist(base_lens, bins=bins, alpha=0.55, color=GREY,   label="Base Model",               edgecolor="white")
        ax.hist(ft_lens,   bins=bins, alpha=0.70, color=BLUE,   label="Fine-Tuned Model",         edgecolor="white")
        ax.axvline(ref_lens.mean(),  color=GREEN,  linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvline(base_lens.mean(), color=GREY,   linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvline(ft_lens.mean(),   color=BLUE,   linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Response Length (words)", fontsize=12)
        ax.set_ylabel("Number of Examples",      fontsize=12)
        ax.set_title("Response Length Distribution\n(dashed lines = mean)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        plt.tight_layout()
        path = os.path.join(out_dir, "response_length_dist.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


def plot_per_intent_rouge(df: pd.DataFrame, out_dir: str) -> str | None:
    """ROUGE-1 per intent category — shows where model improved most."""
    if "intent" not in df.columns or df["intent"].isna().all():
        return None

    from evaluate import load as hf_load
    rouge = hf_load("rouge")

    records = []
    for intent in df["intent"].dropna().unique():
        sub = df[df["intent"] == intent]
        if len(sub) < 2:
            continue
        b = rouge.compute(predictions=sub["baseline"].tolist(), references=sub["reference"].tolist())["rouge1"]
        f = rouge.compute(predictions=sub["fine_tuned"].tolist(), references=sub["reference"].tolist())["rouge1"]
        records.append({"intent": intent, "base_rouge1": b, "ft_rouge1": f, "delta": f - b})

    if not records:
        return None

    rec_df = pd.DataFrame(records).sort_values("delta", ascending=True)

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(11, max(6, len(rec_df) * 0.4 + 2)))
        y = np.arange(len(rec_df))
        ax.barh(y - 0.2, rec_df["base_rouge1"], height=0.35, color=GREY,  alpha=0.8, label="Base Model")
        ax.barh(y + 0.2, rec_df["ft_rouge1"],   height=0.35, color=BLUE,  alpha=0.9, label="Fine-Tuned")
        ax.set_yticks(y)
        ax.set_yticklabels(rec_df["intent"], fontsize=9)
        ax.set_xlabel("ROUGE-1",  fontsize=12)
        ax.set_title("ROUGE-1 per Customer Intent\n(base vs fine-tuned)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        plt.tight_layout()
        path = os.path.join(out_dir, "per_intent_rouge.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


def plot_improvement_heatmap(metrics: dict, out_dir: str) -> str:
    """Delta heatmap — positive = green (improved), negative = red (regressed)."""
    keys    = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1", "exact_match"]
    labels  = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore F1", "Exact Match"]
    deltas  = []
    for k in keys:
        try:
            deltas.append(float(metrics["improvement"].get(k, 0) or 0))
        except (TypeError, ValueError):
            deltas.append(0.0)

    data = np.array(deltas).reshape(1, -1)
    vmax = max(abs(min(deltas)), abs(max(deltas)), 0.01)

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 2.5))
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticks([])
        for j, val in enumerate(deltas):
            sign = "+" if val >= 0 else ""
            color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(j, 0, f"{sign}{val:.4f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
        plt.colorbar(im, ax=ax, label="Delta (fine-tuned − base)", shrink=0.9)
        ax.set_title("Metric Improvement Heatmap\n(green = improved, red = regressed)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, "improvement_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


# ── Markdown report generator ─────────────────────────────────────────────────

def make_markdown_report(
    metrics: dict,
    df: pd.DataFrame,
    training_log: list,
    plot_paths: dict,
    out_path: str,
) -> None:
    """Write the full evaluation report as a Markdown file."""
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
    base = metrics["base_model"]
    ft   = metrics["fine_tuned"]
    imp  = metrics["improvement"]
    n    = metrics["eval_samples"]

    def fmt(v, digits=4):
        try:
            return f"{float(v):.{digits}f}"
        except (TypeError, ValueError):
            return str(v)

    def delta_emoji(v):
        try:
            f = float(v)
            if f > 0.001:  return f"🟢 +{f:.4f}"
            if f < -0.001: return f"🔴 {f:.4f}"
            return f"⚪ {f:.4f}"
        except:
            return str(v)

    lines = [
        f"# LLM Fine-Tuning Evaluation Report",
        f"",
        f"> **Generated:** {ts}  ",
        f"> **Model:** `{MODEL_ID}`  ",
        f"> **Adapter:** `{ADAPTER_DIR}`  ",
        f"> **Eval samples:** {n}  ",
        f"",
        "---",
        "",
        "## Executive Summary",
        "",
    ]

    # determine overall verdict
    rouge1_delta = float(imp.get("rouge1", 0) or 0)
    bleu_delta   = float(imp.get("bleu", 0)   or 0)
    if rouge1_delta > 0.02 and bleu_delta > 0:
        verdict = "✅ **Fine-tuning was SUCCESSFUL.** The model clearly improved on domain-specific customer support responses."
    elif rouge1_delta > 0:
        verdict = "🟡 **Fine-tuning showed marginal improvement.** Consider training for more epochs or on a larger subset."
    else:
        verdict = "🔴 **Fine-tuning did not improve the model.** Check training loss curve — the model may have overfit or underfit."

    lines += [
        verdict,
        "",
        f"| | Base Model | Fine-Tuned | Delta |",
        f"|---|---|---|---|",
        f"| **ROUGE-1** | {fmt(base['rouge1'])} | {fmt(ft['rouge1'])} | {delta_emoji(imp['rouge1'])} |",
        f"| **ROUGE-2** | {fmt(base['rouge2'])} | {fmt(ft['rouge2'])} | {delta_emoji(imp['rouge2'])} |",
        f"| **ROUGE-L** | {fmt(base['rougeL'])} | {fmt(ft['rougeL'])} | {delta_emoji(imp['rougeL'])} |",
        f"| **BLEU**    | {fmt(base['bleu'])}   | {fmt(ft['bleu'])}   | {delta_emoji(imp['bleu'])}   |",
        f"| **BERTScore F1** | {fmt(base.get('bertscore_f1','N/A'))} | {fmt(ft.get('bertscore_f1','N/A'))} | {delta_emoji(imp.get('bertscore_f1','N/A'))} |",
        f"| **Exact Match** | {fmt(base['exact_match'])} | {fmt(ft['exact_match'])} | {delta_emoji(imp['exact_match'])} |",
        f"| **Approx Perplexity** ↓ | {fmt(base['approx_perplexity'],2)} | {fmt(ft['approx_perplexity'],2)} | {delta_emoji(imp['approx_perplexity'])} |",
        f"| **Avg Response Length (words)** | {base['avg_pred_length_words']} | {ft['avg_pred_length_words']} | — |",
        f"| **Avg Generation Time (sec)** | {base.get('avg_generation_time_sec','—')} | {ft.get('avg_generation_time_sec','—')} | — |",
        "",
        "---",
        "",
    ]

    # Charts section
    if HAS_PLOTTING:
        lines += ["## Charts & Visualisations", ""]
        chart_info = [
            ("metrics_comparison_bar", "Metric Comparison Bar Chart",
             "Side-by-side comparison of all metrics between base and fine-tuned model."),
            ("training_loss_curve",    "Training Loss Curve",
             "How the model's loss decreased over training steps."),
            ("rouge_breakdown",        "ROUGE Score Radar Chart",
             "All four ROUGE sub-scores visualised as a spider/radar chart."),
            ("response_length_dist",   "Response Length Distribution",
             "How verbose each model is compared to the ground-truth references."),
            ("per_intent_rouge",       "Per-Intent ROUGE-1",
             "Where the model improved most (by customer service intent)."),
            ("improvement_heatmap",    "Improvement Heatmap",
             "Green = improved, Red = regressed, compared to base."),
        ]
        for key, title, desc in chart_info:
            path = plot_paths.get(key)
            if path and os.path.exists(path):
                rel = os.path.relpath(path, os.path.dirname(out_path)).replace("\\", "/")
                lines += [f"### {title}", f"*{desc}*", f"", f"![{title}]({rel})", ""]

        lines += ["---", ""]

    # Training curve stats (if available)
    if training_log:
        train_losses = [e["loss"] for e in training_log if "loss" in e]
        eval_losses  = [e["eval_loss"] for e in training_log if "eval_loss" in e]
        lines += [
            "## Training Statistics",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Initial train loss  | {train_losses[0]:.4f} |" if train_losses else "",
            f"| Final train loss    | {train_losses[-1]:.4f} |" if train_losses else "",
            f"| Best eval loss      | {min(eval_losses):.4f} |" if eval_losses else "",
            f"| Training steps      | {training_log[-1].get('step', '—')} |",
            "",
            "---",
            "",
        ]

    # Side-by-side examples
    lines += ["## Side-by-Side Response Examples", ""]
    sample = df.head(8) if not df.empty else pd.DataFrame()
    for i, row in sample.iterrows():
        lines += [
            f"### Example {i+1}",
            f"**Intent:** `{row.get('intent','—')}` | **Category:** `{row.get('category','—')}`",
            f"",
            f"**Prompt:** {row['prompt']}",
            f"",
            f"**Reference (ground truth):**",
            f"> {str(row['reference'])[:300]}{'...' if len(str(row['reference'])) > 300 else ''}",
            f"",
            f"**Base Model:**",
            f"> {str(row['baseline'])[:300]}{'...' if len(str(row['baseline'])) > 300 else ''}",
            f"",
            f"**Fine-Tuned Model:**",
            f"> {str(row['fine_tuned'])[:300]}{'...' if len(str(row['fine_tuned'])) > 300 else ''}",
            f"",
            "---",
            "",
        ]

    # Metric explanations
    lines += [
        "## Metric Reference Guide",
        "",
        "| Metric | What It Measures | Range | Good Range for Support Chat |",
        "|---|---|---|---|",
        "| **ROUGE-1** | Unigram (word) overlap | 0–1 | > 0.30 |",
        "| **ROUGE-2** | Bigram overlap | 0–1 | > 0.10 |",
        "| **ROUGE-L** | Longest common subsequence | 0–1 | > 0.25 |",
        "| **BLEU** | Precision n-gram match | 0–1 | > 0.05 |",
        "| **BERTScore F1** | Semantic similarity (embedding-level) | 0–1 | > 0.85 |",
        "| **Exact Match** | Perfect string equality | 0–1 | > 0.01 |",
        "| **Perplexity** | How 'surprised' the model is (lower = better) | 1–∞ | < 50 |",
        "",
        "> **Note:** ROUGE and BLEU measure surface-level text overlap — they are standard but imperfect.",
        "> BERTScore is more meaningful for open-ended generation. For true production validation,",
        "> add human evaluation or LLM-as-judge scoring on top of these automatic metrics.",
        "",
        "---",
        "",
        "*Report generated by `src/report.py` — LLM Fine-Tuning Pipeline*",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Markdown report saved to: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading evaluation data...")
    metrics, df, training_log = load_data()
    print(f"  Metrics loaded. Eval samples: {metrics['eval_samples']}")
    print(f"  Predictions loaded: {len(df)} rows")
    print(f"  Training log: {len(training_log)} entries")

    plot_paths = {}
    if HAS_PLOTTING:
        print("\nGenerating charts...")

        path = plot_training_loss(training_log, PLOTS_DIR)
        if path:
            print(f"  training_loss_curve.png")
            plot_paths["training_loss_curve"] = path

        path = plot_metrics_comparison(metrics, PLOTS_DIR)
        print(f"  metrics_comparison_bar.png")
        plot_paths["metrics_comparison_bar"] = path

        path = plot_rouge_breakdown(metrics, PLOTS_DIR)
        print(f"  rouge_breakdown.png")
        plot_paths["rouge_breakdown"] = path

        if not df.empty:
            path = plot_response_length(df, PLOTS_DIR)
            print(f"  response_length_dist.png")
            plot_paths["response_length_dist"] = path

            path = plot_per_intent_rouge(df, PLOTS_DIR)
            if path:
                print(f"  per_intent_rouge.png")
                plot_paths["per_intent_rouge"] = path

        path = plot_improvement_heatmap(metrics, PLOTS_DIR)
        print(f"  improvement_heatmap.png")
        plot_paths["improvement_heatmap"] = path

    print("\nGenerating Markdown report...")
    report_path = os.path.join(EVAL_DIR, "report.md")
    make_markdown_report(metrics, df, training_log, plot_paths, report_path)

    print(f"\n{'='*55}")
    print("REPORT GENERATION COMPLETE")
    print(f"{'='*55}")
    print(f"  Report:  {report_path}")
    print(f"  Charts:  {PLOTS_DIR}/")
    print(f"  Plots:   {len(plot_paths)} generated")
    print()


if __name__ == "__main__":
    main()
