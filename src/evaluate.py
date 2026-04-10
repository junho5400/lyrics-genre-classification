#!/usr/bin/env python
"""Evaluate trained models on the held-out test set."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    GPT2ForSequenceClassification,
    T5ForConditionalGeneration,
    AutoTokenizer,
)

from src.data.dataset import get_dataset
from src.models.t5_classifier import decode_prediction, prediction_to_label

GENRE_NAMES = ["Rap/Hip-Hop", "Pop", "Rock", "Country", "R&B"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_trained_model(model_key: str, cfg: dict, checkpoint_dir: str, device):
    """Load a PEFT-wrapped model from checkpoint."""
    model_cfg = cfg["models"][model_key]
    model_name = model_cfg["name"]
    ckpt = str(Path(checkpoint_dir) / model_key / "best")

    if model_key == "roberta":
        base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
        model = PeftModel.from_pretrained(base, ckpt)
    elif model_key == "gpt2":
        base = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=5)
        base.config.pad_token_id = base.config.eos_token_id
        model = PeftModel.from_pretrained(base, ckpt)
    elif model_key == "t5":
        base = T5ForConditionalGeneration.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base, ckpt)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_classification(model, loader, device):
    """Evaluate an encoder-only or decoder-only classifier."""
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def evaluate_seq2seq(model, tokenizer, loader, device):
    """Evaluate T5 by generating and mapping outputs."""
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing T5", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            genre_labels = batch["genre_label"]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
            )
            for gen_ids, true_label in zip(generated, genre_labels):
                pred_text = decode_prediction(tokenizer, gen_ids)
                pred_label = prediction_to_label(pred_text)
                all_preds.append(pred_label)
                all_labels.append(true_label.item())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, model_name: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=GENRE_NAMES,
        yticklabels=GENRE_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained genre classifiers")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="results/checkpoints")
    parser.add_argument("--output_dir", default="results/figures")
    parser.add_argument("--models", nargs="+", default=["roberta", "gpt2", "t5"])
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_key in args.models:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_key}")
        print(f"{'='*50}")

        model_type = cfg["models"][model_key]["type"]
        model, tokenizer = load_trained_model(model_key, cfg, args.checkpoint_dir, device)

        test_ds = get_dataset("test", tokenizer, model_type, args.data_dir, cfg["training"]["max_seq_len"])
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        if model_type == "encoder-decoder":
            preds, labels = evaluate_seq2seq(model, tokenizer, test_loader, device)
            valid_mask = preds >= 0
            if (~valid_mask).sum() > 0:
                print(f"  Warning: {(~valid_mask).sum()} unrecognised T5 predictions")
            preds_valid = preds[valid_mask]
            labels_valid = labels[valid_mask]
        else:
            preds_valid, labels_valid = evaluate_classification(model, test_loader, device)

        acc = accuracy_score(labels_valid, preds_valid)
        f1_macro = f1_score(labels_valid, preds_valid, average="macro", zero_division=0)
        f1_weighted = f1_score(labels_valid, preds_valid, average="weighted", zero_division=0)

        print(f"\n  Accuracy:    {acc:.4f}")
        print(f"  F1 (macro):  {f1_macro:.4f}")
        print(f"  F1 (weight): {f1_weighted:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(labels_valid, preds_valid, target_names=GENRE_NAMES, digits=4, zero_division=0))

        plot_confusion_matrix(
            labels_valid, preds_valid, model_key.upper(), str(out_dir / f"cm_{model_key}.png")
        )

        all_results[model_key] = {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
        }

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Comparison bar chart
    if len(all_results) > 1:
        model_names = list(all_results.keys())
        metrics = ["accuracy", "f1_macro", "f1_weighted"]
        x = np.arange(len(model_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, metric in enumerate(metrics):
            vals = [all_results[m][metric] for m in model_names]
            bars = ax.bar(x + i * width, vals, width, label=metric.replace("_", " ").title())
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("Score")
        ax.set_title("Model Comparison on Test Set")
        ax.set_xticks(x + width)
        ax.set_xticklabels([n.upper() for n in model_names])
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(str(out_dir / "model_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved comparison chart to {out_dir / 'model_comparison.png'}")

    with open(out_dir / "test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
