#!/usr/bin/env python
"""Contrastive Integrated Gradients + misclassification analysis.

For each frequently-confused genre pair (e.g. Pop vs Rock), attributes the
*difference* in logits (logit_A - logit_B) to input tokens.  This reveals which
words push the model toward one genre over another — far more informative than
single-class IG when genres are highly similar.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from peft import PeftModel
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import LayerIntegratedGradients

GENRE_NAMES = {0: "Rap/Hip-Hop", 1: "Pop", 2: "Rock", 3: "Country", 4: "R&B"}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class ContrastiveWrapper(nn.Module):
    """Returns logit_A - logit_B for contrastive attribution."""
    def __init__(self, model, class_a, class_b):
        super().__init__()
        self.model = model
        self.class_a = class_a
        self.class_b = class_b

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return (logits[:, self.class_a] - logits[:, self.class_b]).unsqueeze(-1)


def get_emb_layer(model):
    base = getattr(model, "base_model", model)
    return base.roberta.embeddings.word_embeddings


def get_predictions(model, tokenizer, test_ds, device, batch_size=32):
    """Run model on test set and return predictions + labels."""
    from src.data.dataset import get_dataset
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def find_top_confused_pairs(y_true, y_pred, top_k=3):
    """Find the most confused genre pairs from the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    # Zero out diagonal
    np.fill_diagonal(cm, 0)
    # Find top-k off-diagonal entries
    pairs = []
    flat = cm.flatten()
    for _ in range(top_k):
        idx = flat.argmax()
        i, j = divmod(idx, 5)
        pairs.append((i, j, int(flat[idx])))
        flat[idx] = 0
    return pairs


def contrastive_attribution(model, emb_layer, tokenizer, text, class_a, class_b,
                            device, max_length=512, n_steps=30):
    """Compute contrastive IG: attribute (logit_a - logit_b) to input tokens."""
    wrapper = ContrastiveWrapper(model, class_a, class_b)
    lig = LayerIntegratedGradients(wrapper, emb_layer)

    enc = tokenizer(text, truncation=True, max_length=max_length,
                    return_tensors="pt", padding=False)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    attr = lig.attribute(ids, additional_forward_args=(mask,),
                         n_steps=n_steps, return_convergence_delta=False)
    scores = attr.sum(dim=-1).squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids[0].cpu())
    sl = mask.sum().item()
    return tokens[:sl], scores[:sl]


def plot_contrastive_bar(tokens, scores, genre_a, genre_b, title, out_path, top_k=20):
    """Plot tokens colored by which genre they push toward."""
    abs_scores = np.abs(scores)
    indices = np.argsort(abs_scores)[::-1][:top_k]
    top_tokens = [tokens[i].replace("\u0120", "").replace("\u2581", " ").strip() for i in indices]
    top_scores = scores[indices]

    colors = [sns.color_palette("Set2")[0] if s > 0 else sns.color_palette("Set2")[1] for s in top_scores]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(top_tokens))
    ax.barh(y_pos, top_scores, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_tokens, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"← {genre_b}    Attribution Score    {genre_a} →")
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_contrastive_vocabulary(model, emb_layer, tokenizer, ds_rows,
                                 class_a, class_b, device, max_seq_len,
                                 n_samples=20):
    """Aggregate contrastive attribution across samples to find discriminative tokens."""
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(ds_rows), size=min(n_samples, len(ds_rows)), replace=False)

    vocab_a = Counter()  # tokens pushing toward class_a
    vocab_b = Counter()  # tokens pushing toward class_b

    for idx in tqdm(sample_idx, desc=f"  Contrastive vocab {GENRE_NAMES[class_a]} vs {GENRE_NAMES[class_b]}", leave=False):
        row = ds_rows[int(idx)]
        tokens, scores = contrastive_attribution(
            model, emb_layer, tokenizer, row["lyrics"],
            class_a, class_b, device, max_seq_len, n_steps=20)
        for tok, s in zip(tokens, scores):
            clean = tok.replace("\u0120", "").replace("\u2581", "").lower().strip()
            if len(clean) > 2 and clean.isalpha():
                if s > 0:
                    vocab_a[clean] += float(s)
                else:
                    vocab_b[clean] += abs(float(s))
    return vocab_a.most_common(20), vocab_b.most_common(20)


def main():
    parser = argparse.ArgumentParser(description="Contrastive attribution + misclassification analysis")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="results/checkpoints")
    parser.add_argument("--output_dir", default="results/figures")
    parser.add_argument("--model", default="roberta")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    out_dir = Path(args.output_dir) / "contrastive"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_seq_len = cfg["training"]["max_seq_len"]

    # Load model
    ckpt = str(Path(args.checkpoint_dir) / args.model / "best")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["models"][args.model]["name"], num_labels=5)
    model = PeftModel.from_pretrained(base, ckpt).to(device).eval()
    emb_layer = get_emb_layer(model)

    # Load test data and get predictions
    from src.data.dataset import get_dataset
    test_ds = get_dataset("test", tokenizer, "encoder", args.data_dir, max_seq_len)
    preds, labels = get_predictions(model, tokenizer, test_ds, device)

    ds_raw = load_from_disk(args.data_dir)["test"]

    # --- 1. Find top confused pairs ---
    confused_pairs = find_top_confused_pairs(labels, preds, top_k=3)
    print("\nTop confused genre pairs (true -> predicted, count):")
    for true_g, pred_g, count in confused_pairs:
        print(f"  {GENRE_NAMES[true_g]} -> {GENRE_NAMES[pred_g]}: {count} errors")

    # --- 2. Contrastive attribution for each confused pair ---
    for true_g, pred_g, count in confused_pairs:
        pair_name = f"{GENRE_NAMES[true_g]}_vs_{GENRE_NAMES[pred_g]}"
        print(f"\n{'='*50}")
        print(f"Contrastive analysis: {GENRE_NAMES[true_g]} vs {GENRE_NAMES[pred_g]} ({count} errors)")

        # Find misclassified examples for this pair
        misclassified_idx = np.where((labels == true_g) & (preds == pred_g))[0]

        # Show 2 example misclassifications with contrastive attribution
        for ex_i, idx in enumerate(misclassified_idx[:2]):
            row = ds_raw[int(idx)]
            tokens, scores = contrastive_attribution(
                model, emb_layer, tokenizer, row["lyrics"],
                true_g, pred_g, device, max_seq_len)
            plot_contrastive_bar(
                tokens, scores, GENRE_NAMES[true_g], GENRE_NAMES[pred_g],
                f"Misclassified: True={GENRE_NAMES[true_g]}, Pred={GENRE_NAMES[pred_g]}",
                str(out_dir / f"misclass_{pair_name}_{ex_i}.png"))

        # Build contrastive vocabulary from ALL samples of both genres
        combined_rows = [r for r in ds_raw if r["label"] in (true_g, pred_g)]
        vocab_a, vocab_b = build_contrastive_vocabulary(
            model, emb_layer, tokenizer, combined_rows,
            true_g, pred_g, device, max_seq_len, n_samples=30)

        # Plot contrastive vocabulary
        if vocab_a and vocab_b:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

            if vocab_a:
                words_a, weights_a = zip(*vocab_a)
                ax1.barh(range(len(words_a)), weights_a, color=sns.color_palette("Set2")[0], alpha=0.85)
                ax1.set_yticks(range(len(words_a)))
                ax1.set_yticklabels(words_a, fontsize=9)
                ax1.invert_yaxis()
                ax1.set_xlabel("Cumulative Attribution")
                ax1.set_title(f"Tokens pushing toward {GENRE_NAMES[true_g]}")

            if vocab_b:
                words_b, weights_b = zip(*vocab_b)
                ax2.barh(range(len(words_b)), weights_b, color=sns.color_palette("Set2")[1], alpha=0.85)
                ax2.set_yticks(range(len(words_b)))
                ax2.set_yticklabels(words_b, fontsize=9)
                ax2.invert_yaxis()
                ax2.set_xlabel("Cumulative Attribution")
                ax2.set_title(f"Tokens pushing toward {GENRE_NAMES[pred_g]}")

            fig.suptitle(f"Contrastive Vocabulary: {GENRE_NAMES[true_g]} vs {GENRE_NAMES[pred_g]}", fontsize=13)
            plt.tight_layout()
            plt.savefig(str(out_dir / f"vocab_{pair_name}.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # --- 3. Misclassification summary ---
    cm = confusion_matrix(labels, preds, labels=list(range(5)))
    # Normalized confusion matrix (row-normalized)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    genre_labels = [GENRE_NAMES[i] for i in range(5)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=genre_labels, yticklabels=genre_labels, ax=ax1)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.set_title("Confusion Matrix (counts)")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=genre_labels, yticklabels=genre_labels, ax=ax2)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
    ax2.set_title("Confusion Matrix (row-normalized)")
    plt.tight_layout()
    plt.savefig(str(out_dir / "confusion_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save summary
    summary = {
        "confused_pairs": [
            {"true": GENRE_NAMES[t], "predicted": GENRE_NAMES[p], "count": c}
            for t, p, c in confused_pairs
        ],
        "accuracy": float((preds == labels).mean()),
        "per_class_accuracy": {
            GENRE_NAMES[i]: float((preds[labels == i] == i).mean()) for i in range(5)
        },
    }
    with open(out_dir / "contrastive_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nContrastive analysis saved to {out_dir}/")


if __name__ == "__main__":
    main()
