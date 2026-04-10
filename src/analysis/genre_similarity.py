#!/usr/bin/env python
"""Compute inter-genre similarity using sentence embeddings to show task difficulty."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

GENRE_NAMES = {0: "Rap/Hip-Hop", 1: "Pop", 2: "Rock", 3: "Country", 4: "R&B"}


def main():
    parser = argparse.ArgumentParser(description="Genre similarity analysis")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="results/figures")
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model to use")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max samples per genre for embedding (for speed)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.data_dir} ...")
    ds = load_from_disk(args.data_dir)
    # Use all splits combined for a fuller picture
    from datasets import concatenate_datasets
    all_data = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])

    print(f"Loading sentence-transformers model: {args.model_name} ...")
    model = SentenceTransformer(args.model_name)

    # Collect lyrics per genre
    genre_lyrics = {i: [] for i in range(5)}
    for row in all_data:
        genre_lyrics[row["label"]].append(row["lyrics"])

    rng = np.random.default_rng(42)
    genre_embeddings = {}

    for label_id in range(5):
        lyrics = genre_lyrics[label_id]
        if len(lyrics) > args.max_samples:
            indices = rng.choice(len(lyrics), size=args.max_samples, replace=False)
            lyrics = [lyrics[i] for i in indices]
        print(f"  Encoding {GENRE_NAMES[label_id]}: {len(lyrics)} samples ...")
        embs = model.encode(lyrics, show_progress_bar=True, batch_size=64)
        genre_embeddings[label_id] = embs

    # --- 1. Inter-genre centroid similarity ---
    centroids = {k: v.mean(axis=0) for k, v in genre_embeddings.items()}
    centroid_matrix = np.stack([centroids[i] for i in range(5)])
    sim_matrix = cosine_similarity(centroid_matrix)

    labels = [GENRE_NAMES[i] for i in range(5)]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels, vmin=0.8, vmax=1.0, ax=ax)
    ax.set_title("Inter-Genre Cosine Similarity (Sentence Embedding Centroids)")
    plt.tight_layout()
    plt.savefig(str(out_dir / "genre_centroid_similarity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved centroid similarity heatmap")

    # --- 2. Pairwise sample-level similarity distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Within-genre vs between-genre similarity distributions
    within_sims = []
    between_sims = []

    for i in range(5):
        embs_i = genre_embeddings[i]
        # Within-genre: sample pairs within the same genre
        n = min(200, len(embs_i))
        idx = rng.choice(len(embs_i), size=n, replace=False)
        subset = embs_i[idx]
        sim = cosine_similarity(subset)
        # Upper triangle only (exclude diagonal)
        triu_idx = np.triu_indices(n, k=1)
        within_sims.extend(sim[triu_idx].tolist())

    for i in range(5):
        for j in range(i + 1, 5):
            n_i = min(100, len(genre_embeddings[i]))
            n_j = min(100, len(genre_embeddings[j]))
            idx_i = rng.choice(len(genre_embeddings[i]), size=n_i, replace=False)
            idx_j = rng.choice(len(genre_embeddings[j]), size=n_j, replace=False)
            sim = cosine_similarity(genre_embeddings[i][idx_i], genre_embeddings[j][idx_j])
            between_sims.extend(sim.flatten().tolist())

    axes[0].hist(within_sims, bins=50, alpha=0.7, label="Within-genre", color="steelblue", density=True)
    axes[0].hist(between_sims, bins=50, alpha=0.7, label="Between-genre", color="coral", density=True)
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Within-Genre vs Between-Genre Similarity")
    axes[0].legend()

    # --- 3. Full pairwise genre similarity matrix (mean of all sample pairs) ---
    full_sim = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if i == j:
                embs = genre_embeddings[i]
                n = min(300, len(embs))
                idx = rng.choice(len(embs), size=n, replace=False)
                sim = cosine_similarity(embs[idx])
                triu = np.triu_indices(n, k=1)
                full_sim[i, j] = sim[triu].mean()
            else:
                n_i = min(200, len(genre_embeddings[i]))
                n_j = min(200, len(genre_embeddings[j]))
                idx_i = rng.choice(len(genre_embeddings[i]), size=n_i, replace=False)
                idx_j = rng.choice(len(genre_embeddings[j]), size=n_j, replace=False)
                sim = cosine_similarity(genre_embeddings[i][idx_i], genre_embeddings[j][idx_j])
                full_sim[i, j] = sim.mean()

    sns.heatmap(full_sim, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Mean Pairwise Cosine Similarity")

    plt.tight_layout()
    plt.savefig(str(out_dir / "genre_similarity_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved similarity analysis figure")

    # --- 4. Print summary statistics ---
    print("\n=== Genre Similarity Summary ===")
    print(f"Mean within-genre similarity:  {np.mean(within_sims):.4f} (std: {np.std(within_sims):.4f})")
    print(f"Mean between-genre similarity: {np.mean(between_sims):.4f} (std: {np.std(between_sims):.4f})")
    print(f"Overlap ratio (between/within): {np.mean(between_sims)/np.mean(within_sims):.4f}")
    print("\nCentroid similarity matrix:")
    for i in range(5):
        row = "  ".join(f"{sim_matrix[i,j]:.3f}" for j in range(5))
        print(f"  {labels[i]:>12s}: {row}")

    # Save numerical results
    results = {
        "within_genre_mean": float(np.mean(within_sims)),
        "within_genre_std": float(np.std(within_sims)),
        "between_genre_mean": float(np.mean(between_sims)),
        "between_genre_std": float(np.std(between_sims)),
        "centroid_similarity": {
            labels[i]: {labels[j]: float(sim_matrix[i, j]) for j in range(5)}
            for i in range(5)
        },
        "pairwise_mean_similarity": {
            labels[i]: {labels[j]: float(full_sim[i, j]) for j in range(5)}
            for i in range(5)
        },
    }
    import json
    with open(out_dir / "genre_similarity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'genre_similarity_results.json'}")


if __name__ == "__main__":
    main()
