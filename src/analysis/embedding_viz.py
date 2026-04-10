#!/usr/bin/env python
"""Extract final-layer embeddings and visualise with UMAP / t-SNE."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    GPT2Model,
    T5EncoderModel,
    AutoModelForSequenceClassification,
    GPT2ForSequenceClassification,
    T5ForConditionalGeneration,
)

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from src.data.dataset import get_dataset

GENRE_NAMES = {0: "Rap/Hip-Hop", 1: "Pop", 2: "Rock", 3: "Country", 4: "R&B"}
PALETTE = sns.color_palette("Set2", n_colors=5)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_roberta_embeddings(checkpoint_dir, cfg, data_dir, device, max_seq_len, batch_size):
    """Extract [CLS] token embeddings from fine-tuned RoBERTa."""
    ckpt = str(Path(checkpoint_dir) / "roberta" / "best")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["models"]["roberta"]["name"], num_labels=5, output_hidden_states=True
    )
    model = PeftModel.from_pretrained(base, ckpt)
    model = model.to(device).eval()

    ds = get_dataset("test", tokenizer, "encoder", data_dir, max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    embeddings, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="RoBERTa embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Last hidden state -> CLS token
            hidden = outputs.hidden_states[-1][:, 0, :]
            embeddings.append(hidden.cpu().numpy())
            labels.extend(batch["labels"].numpy())

    return np.concatenate(embeddings), np.array(labels)


def extract_gpt2_embeddings(checkpoint_dir, cfg, data_dir, device, max_seq_len, batch_size):
    """Extract last-token embeddings from fine-tuned GPT-2."""
    ckpt = str(Path(checkpoint_dir) / "gpt2" / "best")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    base = GPT2ForSequenceClassification.from_pretrained(
        cfg["models"]["gpt2"]["name"], num_labels=5, output_hidden_states=True
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, ckpt)
    model = model.to(device).eval()

    ds = get_dataset("test", tokenizer, "decoder", data_dir, max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    embeddings, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="GPT-2 embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.hidden_states[-1]
            # Use last non-padding token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1
            last_hidden = hidden[torch.arange(hidden.size(0)), seq_lengths]
            embeddings.append(last_hidden.cpu().numpy())
            labels.extend(batch["labels"].numpy())

    return np.concatenate(embeddings), np.array(labels)


def extract_t5_embeddings(checkpoint_dir, cfg, data_dir, device, max_seq_len, batch_size):
    """Extract mean-pooled encoder embeddings from fine-tuned T5."""
    ckpt = str(Path(checkpoint_dir) / "t5" / "best")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    base = T5ForConditionalGeneration.from_pretrained(cfg["models"]["t5"]["name"])
    model = PeftModel.from_pretrained(base, ckpt)
    model = model.to(device).eval()

    ds = get_dataset("test", tokenizer, "encoder-decoder", data_dir, max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    embeddings, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="T5 embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
            hidden = encoder_outputs.last_hidden_state
            # Mean pool over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            embeddings.append(pooled.cpu().numpy())
            labels.extend(batch["genre_label"].numpy())

    return np.concatenate(embeddings), np.array(labels)


EXTRACTORS = {
    "roberta": extract_roberta_embeddings,
    "gpt2": extract_gpt2_embeddings,
    "t5": extract_t5_embeddings,
}


def plot_projections(embeddings_2d, labels, model_name, method, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    for label_id in sorted(set(labels)):
        mask = labels == label_id
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=GENRE_NAMES[label_id],
            color=PALETTE[label_id],
            alpha=0.7,
            s=30,
        )
    ax.set_title(f"{model_name.upper()} — {method} of Final-Layer Embeddings")
    ax.legend(title="Genre", loc="best")
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Embedding visualization")
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
    max_seq_len = cfg["training"]["max_seq_len"]

    for model_key in args.models:
        print(f"\nExtracting embeddings: {model_key}")
        embs, labels = EXTRACTORS[model_key](
            args.checkpoint_dir, cfg, args.data_dir, device, max_seq_len, args.batch_size
        )
        print(f"  Shape: {embs.shape}")

        # t-SNE
        print("  Computing t-SNE ...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        embs_tsne = tsne.fit_transform(embs)
        plot_projections(embs_tsne, labels, model_key, "t-SNE", str(out_dir / f"tsne_{model_key}.png"))

        # UMAP
        if HAS_UMAP:
            print("  Computing UMAP ...")
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            embs_umap = reducer.fit_transform(embs)
            plot_projections(embs_umap, labels, model_key, "UMAP", str(out_dir / f"umap_{model_key}.png"))
        else:
            print("  umap-learn not installed, skipping UMAP")

        np.savez(str(out_dir / f"embeddings_{model_key}.npz"), embeddings=embs, labels=labels)


if __name__ == "__main__":
    main()
