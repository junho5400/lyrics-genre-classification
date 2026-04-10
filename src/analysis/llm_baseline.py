#!/usr/bin/env python
"""Classify lyrics using Claude API to establish a proprietary-model baseline."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

GENRE_NAMES = ["Rap/Hip-Hop", "Pop", "Rock", "Country", "R&B"]
LABEL_MAP = {"rap": 0, "pop": 1, "rock": 2, "country": 3, "r&b": 4, "rb": 4}

SYSTEM_PROMPT = """You are a music genre classifier. Given song lyrics, classify the genre as exactly one of: rap, pop, rock, country, r&b.

Respond with ONLY the genre label (one word, lowercase). Do not explain your reasoning."""

USER_TEMPLATE = """Classify the genre of the following song lyrics:

{lyrics}

Genre:"""


def classify_with_claude(lyrics: str, client, model: str) -> str:
    """Send lyrics to Claude and return the predicted genre string."""
    # Truncate very long lyrics to stay within token limits
    if len(lyrics) > 3000:
        lyrics = lyrics[:3000]

    message = client.messages.create(
        model=model,
        max_tokens=10,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(lyrics=lyrics)}],
    )
    return message.content[0].text.strip().lower()


def parse_prediction(pred_text: str) -> int:
    """Map Claude's response to an integer label."""
    cleaned = pred_text.strip().lower().rstrip(".")
    if cleaned in LABEL_MAP:
        return LABEL_MAP[cleaned]
    # Handle common variations
    for name, label in LABEL_MAP.items():
        if name in cleaned:
            return label
    if "hip" in cleaned or "hip-hop" in cleaned:
        return 0
    return -1


def main():
    parser = argparse.ArgumentParser(description="Claude API genre classification baseline")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="results/figures")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Claude model to use")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of test samples per genre to classify")
    parser.add_argument("--results_file", default=None,
                        help="Path to existing results JSON to resume from (skip API calls)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "llm_baseline_results.json"

    if args.results_file and Path(args.results_file).exists():
        print(f"Loading existing results from {args.results_file}")
        with open(args.results_file) as f:
            saved = json.load(f)
        all_true = saved["true_labels"]
        all_pred = saved["pred_labels"]
        raw_preds = saved.get("raw_predictions", [])
    else:
        import anthropic
        client = anthropic.Anthropic()

        print(f"Loading test set from {args.data_dir} ...")
        try:
            ds = load_from_disk(args.data_dir)
            test_ds = ds["test"]
        except FileNotFoundError:
            # Only test split available
            from datasets import DatasetDict, Dataset
            test_ds = Dataset.load_from_disk(str(Path(args.data_dir) / "test"))
            ds = DatasetDict({"test": test_ds})

        # Sample n_samples per genre from test set
        rng = np.random.default_rng(42)
        genre_rows = {i: [] for i in range(5)}
        for row in test_ds:
            genre_rows[row["label"]].append(row)

        samples = []
        for label_id in range(5):
            rows = genre_rows[label_id]
            n = min(args.n_samples, len(rows))
            indices = rng.choice(len(rows), size=n, replace=False)
            for idx in indices:
                samples.append(rows[int(idx)])

        rng.shuffle(samples)

        print(f"Classifying {len(samples)} samples with {args.model} ...")
        all_true = []
        all_pred = []
        raw_preds = []

        for i, sample in enumerate(samples):
            try:
                pred_text = classify_with_claude(sample["lyrics"], client, args.model)
                pred_label = parse_prediction(pred_text)
                all_true.append(sample["label"])
                all_pred.append(pred_label)
                raw_preds.append(pred_text)

                if (i + 1) % 10 == 0:
                    valid = [p for p in all_pred if p >= 0]
                    print(f"  [{i+1}/{len(samples)}] Latest: '{pred_text}' -> {pred_label}, "
                          f"Running acc: {sum(t == p for t, p in zip(all_true, all_pred) if p >= 0) / max(len(valid), 1):.3f}")

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                all_true.append(sample["label"])
                all_pred.append(-1)
                raw_preds.append(f"ERROR: {e}")
                time.sleep(1)

        # Save raw results
        saved = {
            "model": args.model,
            "n_samples_per_genre": args.n_samples,
            "true_labels": all_true,
            "pred_labels": all_pred,
            "raw_predictions": raw_preds,
        }
        with open(results_path, "w") as f:
            json.dump(saved, f, indent=2)
        print(f"\nRaw results saved to {results_path}")

    # Compute metrics (filter out invalid predictions)
    true_arr = np.array(all_true)
    pred_arr = np.array(all_pred)
    valid = pred_arr >= 0
    n_invalid = (~valid).sum()

    if n_invalid > 0:
        print(f"\nWarning: {n_invalid} invalid predictions filtered out")

    true_valid = true_arr[valid]
    pred_valid = pred_arr[valid]

    acc = accuracy_score(true_valid, pred_valid)
    f1_macro = f1_score(true_valid, pred_valid, average="macro", zero_division=0)
    f1_weighted = f1_score(true_valid, pred_valid, average="weighted", zero_division=0)

    print(f"\n{'='*50}")
    print(f"Claude Baseline Results ({args.model})")
    print(f"{'='*50}")
    print(f"  Samples: {len(true_valid)} (valid) / {len(true_arr)} (total)")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weight): {f1_weighted:.4f}")
    print(f"\n{classification_report(true_valid, pred_valid, target_names=GENRE_NAMES, digits=4, zero_division=0)}")

    # Confusion matrix
    cm = confusion_matrix(true_valid, pred_valid, labels=list(range(5)))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=GENRE_NAMES, yticklabels=GENRE_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Claude ({args.model})")
    plt.tight_layout()
    cm_path = out_dir / "cm_claude_baseline.png"
    plt.savefig(str(cm_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # Save final metrics
    metrics = {
        "model": args.model,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "n_valid": int(valid.sum()),
        "n_invalid": int(n_invalid),
    }
    metrics_path = out_dir / "llm_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
