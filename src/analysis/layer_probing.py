#!/usr/bin/env python
"""Layer-wise probing: train a linear classifier on each RoBERTa layer's CLS
representation to reveal where genre information emerges in the network.

Early layers capture surface-level features (word identity, syntax).
Later layers capture semantics.  If genre accuracy rises sharply at a specific
layer, that's where the model learns to distinguish genres.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.dataset import get_dataset

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


def extract_all_layers(model, tokenizer, data_dir, device, max_seq_len,
                       split="test", batch_size=32):
    """Extract CLS embeddings from every layer of the model.

    Returns: dict mapping layer_idx -> (embeddings array, labels array)
    Layer 0 = token embeddings, layer 1..12 = transformer layers.
    """
    ds = get_dataset(split, tokenizer, "encoder", data_dir, max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    layer_embs = {}  # layer_idx -> list of batch arrays
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting layers ({split})"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=ids, attention_mask=mask,
                           output_hidden_states=True)

            # hidden_states: tuple of (batch, seq, hidden) for each layer
            # Layer 0 = embeddings, 1..12 = transformer outputs
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                cls_emb = hidden[:, 0, :].cpu().numpy()  # CLS token
                if layer_idx not in layer_embs:
                    layer_embs[layer_idx] = []
                layer_embs[layer_idx].append(cls_emb)

            all_labels.extend(batch["labels"].numpy())

    labels = np.array(all_labels)
    result = {}
    for layer_idx in layer_embs:
        result[layer_idx] = (np.concatenate(layer_embs[layer_idx]), labels)
    return result


def probe_layers(train_layers, test_layers):
    """Train a logistic regression probe on each layer and return metrics."""
    results = {}
    n_layers = len(train_layers)

    for layer_idx in tqdm(range(n_layers), desc="Probing layers"):
        X_train, y_train = train_layers[layer_idx]
        X_test, y_test = test_layers[layer_idx]

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 multi_class="multinomial", random_state=42)
        clf.fit(X_train, y_train)

        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)

        results[layer_idx] = {
            "train_acc": float(accuracy_score(y_train, train_pred)),
            "test_acc": float(accuracy_score(y_test, test_pred)),
            "train_f1": float(f1_score(y_train, train_pred, average="macro")),
            "test_f1": float(f1_score(y_test, test_pred, average="macro")),
            "per_class_f1": {
                GENRE_NAMES[i]: float(f1_score(y_test == i, test_pred == i))
                for i in range(5)
            },
        }

    return results


def plot_probing_results(results, out_path):
    """Plot accuracy and F1 vs layer number."""
    layers = sorted(results.keys())
    train_acc = [results[l]["train_acc"] for l in layers]
    test_acc = [results[l]["test_acc"] for l in layers]
    train_f1 = [results[l]["train_f1"] for l in layers]
    test_f1 = [results[l]["test_f1"] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(layers, train_acc, "o-", label="Train", color="steelblue", alpha=0.7)
    ax1.plot(layers, test_acc, "s-", label="Test", color="coral", alpha=0.9, linewidth=2)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Probing Accuracy by Layer")
    ax1.legend()
    ax1.set_xticks(layers)
    ax1.grid(True, alpha=0.3)

    ax2.plot(layers, train_f1, "o-", label="Train", color="steelblue", alpha=0.7)
    ax2.plot(layers, test_f1, "s-", label="Test", color="coral", alpha=0.9, linewidth=2)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Probing F1 by Layer")
    ax2.legend()
    ax2.set_xticks(layers)
    ax2.grid(True, alpha=0.3)

    # Annotate best layer
    best_layer = layers[np.argmax(test_f1)]
    best_f1 = max(test_f1)
    ax2.annotate(f"Best: Layer {best_layer}\nF1={best_f1:.3f}",
                 xy=(best_layer, best_f1), xytext=(best_layer - 2, best_f1 - 0.05),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=10, color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_by_layer(results, out_path):
    """Plot per-genre F1 across layers to see which genres emerge when."""
    layers = sorted(results.keys())
    palette = sns.color_palette("Set2", n_colors=5)

    fig, ax = plt.subplots(figsize=(10, 6))
    for genre_id in range(5):
        f1s = [results[l]["per_class_f1"][GENRE_NAMES[genre_id]] for l in layers]
        ax.plot(layers, f1s, "o-", label=GENRE_NAMES[genre_id],
                color=palette[genre_id], linewidth=2, alpha=0.85)

    ax.set_xlabel("Layer")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Genre Probing F1 by Layer")
    ax.legend(title="Genre", loc="lower right")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Layer-wise probing analysis")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="results/checkpoints")
    parser.add_argument("--output_dir", default="results/figures")
    parser.add_argument("--model", default="roberta")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_seq_len = cfg["training"]["max_seq_len"]

    # Load model
    ckpt = str(Path(args.checkpoint_dir) / args.model / "best")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["models"][args.model]["name"], num_labels=5, output_hidden_states=True)
    model = PeftModel.from_pretrained(base, ckpt).to(device).eval()

    print("Extracting embeddings from all layers ...")
    train_layers = extract_all_layers(model, tokenizer, args.data_dir, device,
                                       max_seq_len, split="train", batch_size=args.batch_size)
    test_layers = extract_all_layers(model, tokenizer, args.data_dir, device,
                                      max_seq_len, split="test", batch_size=args.batch_size)

    print(f"Found {len(train_layers)} layers (0=embeddings, 1-12=transformer)")

    print("Training linear probes ...")
    results = probe_layers(train_layers, test_layers)

    # Print summary
    print("\n=== Layer Probing Results ===")
    print(f"{'Layer':>5}  {'Train Acc':>10}  {'Test Acc':>10}  {'Train F1':>10}  {'Test F1':>10}")
    for layer in sorted(results.keys()):
        r = results[layer]
        print(f"{layer:>5}  {r['train_acc']:>10.4f}  {r['test_acc']:>10.4f}  "
              f"{r['train_f1']:>10.4f}  {r['test_f1']:>10.4f}")

    best_layer = max(results, key=lambda l: results[l]["test_f1"])
    print(f"\nBest layer: {best_layer} (test F1 = {results[best_layer]['test_f1']:.4f})")

    # Plot
    plot_probing_results(results, str(out_dir / "layer_probing.png"))
    plot_per_class_by_layer(results, str(out_dir / "layer_probing_per_genre.png"))

    # Save results
    with open(out_dir / "layer_probing_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print(f"\nProbing figures saved to {out_dir}/")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
