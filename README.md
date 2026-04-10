# Lyric-Based Music Genre Classification

An architecture comparison and interpretability study of transformer models for classifying music genre from song lyrics alone. Encoder-only (RoBERTa), decoder-only (GPT-2), and encoder-decoder (T5) are fine-tuned with LoRA under matched adapter budgets, and four independent interpretability analyses locate the performance ceiling in the data rather than the models.

**Author:** Junho Hong &middot; STAT 359 Final Project, Northwestern University, 2026

---

## Summary

Three transformer architectures are fine-tuned with LoRA on a balanced five-genre lyrics dataset of about 10,000 songs. The fine-tuned models plateau between F1 0.46 and 0.61. A zero-shot Claude Opus baseline reaches 0.636. A four-part interpretability analysis examines whether the plateau is model-driven or task-driven and locates the ceiling in the data: Pop, Rock, and R&B are distinguished primarily by musical properties (instrumentation, tempo, production) that are absent from the text. Lyrics-only classification on this taxonomy is bounded at approximately F1 0.65.

## Results

| Model                     | Type                | Trainable | Accuracy | F1 Macro  |
| ------------------------- | ------------------- | --------- | -------- | --------- |
| **RoBERTa-base**          | Fine-tuned (LoRA)   | 1.18M     | 0.613    | **0.611** |
| GPT-2                     | Fine-tuned (LoRA)   | 0.59M     | 0.604    | 0.591     |
| T5-small                  | Fine-tuned (LoRA)   | 0.59M     | 0.454    | 0.463     |
| Claude Haiku              | Zero-shot LLM       | —         | 0.596    | 0.584     |
| **Claude Opus**           | Zero-shot LLM       | —         | 0.649    | **0.636** |
| *Random baseline*         | —                   | —         | *0.200*  | *0.200*   |

The RoBERTa &gt; GPT-2 ranking is confounded by a 150&times; larger trainable classification head (594K vs. 3,800 parameters). T5's larger gap reflects a smaller base model (60M vs. 125M), a text-to-text formulation that adds decoding overhead for a closed-label task, and the absence of a separate trainable head. The cleaner observation is that all fine-tuned approaches, together with two zero-shot LLM baselines, cluster in a narrow band well below perfect accuracy.

## Interpretability Findings

Four independent analyses were run on a RoBERTa variant trained on the larger 5,000-per-genre dataset. Each is designed to probe a different aspect of the problem so that consistency across them is informative.

1. **Genre centroid similarity (Sentence-BERT).** Pop–Rock centroid cosine similarity reaches 0.979; Pop–Country reaches 0.974. At the sample level, within-genre and between-genre pairwise similarity distributions overlap almost entirely (means 0.363 and 0.331). In an embedding space that has never seen the labels, most of the genres are not reliably separable.

2. **Layer probing.** A logistic regression fit on the [CLS] embedding at each of RoBERTa's 13 layers climbs from F1 0.067 at layer 0 (static embeddings) to 0.530 at layer 1, 0.613 at layer 7, and then plateaus. Additional depth adds no linearly accessible genre information.

3. **Representation geometry.** UMAP and t-SNE projections of final-layer [CLS] embeddings show Rap forming a tight, isolated cluster while Pop, Rock, and R&B are heavily interleaved. The cluster structure matches the similarity analysis.

4. **Contrastive Integrated Gradients.** Token attributions computed against the logit difference between confused classes (rather than a single target) isolate the vocabulary that actually distinguishes genre pairs. For Rock vs. Pop, high-attribution tokens toward Rock include intensity terms (*death, blood, pain*); toward Pop, light-emotional terms (*love, night, feel*). Misclassified examples systematically violate these patterns.

Across the 20 directed genre pairs, RoBERTa's per-pair confusion rate and the corresponding centroid similarity correlate at Pearson **r = 0.630**.

## Project Structure

```
.
├── configs/
│   ├── config.yaml                # Architecture comparison hyperparameters
│   └── roberta_optimized.yaml     # Interpretability (Phase 2) config
├── src/
│   ├── data/
│   │   ├── download_and_prepare.py  # Dataset download, filter, split
│   │   └── dataset.py               # PyTorch Dataset classes
│   ├── models/
│   │   ├── roberta_classifier.py    # RoBERTa + LoRA
│   │   ├── gpt2_classifier.py       # GPT-2 + LoRA
│   │   └── t5_classifier.py         # T5 + LoRA (text-to-text)
│   ├── train.py                     # Unified training script
│   ├── evaluate.py                  # Test-set evaluation
│   └── analysis/
│       ├── embedding_viz.py         # UMAP / t-SNE visualization
│       ├── contrastive_analysis.py  # Contrastive Integrated Gradients
│       ├── layer_probing.py         # Linear probing per transformer layer
│       ├── genre_similarity.py      # Sentence-BERT genre similarity
│       └── llm_baseline.py          # Claude API zero-shot baseline
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   └── 02_results.ipynb             # All results and figures
├── paper/
│   ├── main.tex                     # LaTeX paper
│   ├── references.bib               # Bibliography
│   └── figures/                     # Figures for paper
├── scripts/run_all.sh               # One-command reproduction
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+. A GPU is required for model training and interpretability; CPU is sufficient for data preparation and the Sentence-BERT similarity analysis.

## Quick Start

```bash
bash scripts/run_all.sh
```

This runs the full pipeline: data download and preparation, training all three models (Phase 1), evaluation, then the optimized RoBERTa (Phase 2) and all interpretability analyses.

## Step-by-Step Reproduction

### 1. Prepare Data

Downloads the [`sebastiandizon/genius-song-lyrics`](https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics) dataset from HuggingFace, filters to English-only songs across the five genres, samples 2,000 songs per genre (2,500 for Pop to offset higher label noise), and creates stratified 80/10/10 splits.

```bash
python -m src.data.download_and_prepare
```

### 2. Train Models (Architecture Comparison)

```bash
python -m src.train --model roberta
python -m src.train --model gpt2
python -m src.train --model t5
```

### 3. Evaluate

```bash
python -m src.evaluate
```

### 4. Train Interpretability Model (Phase 2)

A larger 5,000-per-genre dataset with expanded LoRA targets, lower learning rate, and data augmentation:

```bash
python -m src.data.download_and_prepare --config configs/roberta_optimized.yaml
python -m src.train --config configs/roberta_optimized.yaml
```

### 5. Interpretability Analyses (GPU)

```bash
# UMAP + t-SNE embedding visualizations
python -m src.analysis.embedding_viz \
  --models roberta \
  --checkpoint_dir results/checkpoints_optimized \
  --output_dir results/figures_optimized

# Contrastive Integrated Gradients for confused pairs
python -m src.analysis.contrastive_analysis \
  --model roberta \
  --checkpoint_dir results/checkpoints_optimized \
  --output_dir results/figures_optimized

# Linear probes per transformer layer
python -m src.analysis.layer_probing \
  --model roberta \
  --checkpoint_dir results/checkpoints_optimized \
  --output_dir results/figures_optimized
```

### 6. Additional Analyses (CPU)

```bash
# Sentence-BERT genre similarity
python -m src.analysis.genre_similarity \
  --data_dir data/processed \
  --output_dir results/figures

# Zero-shot Claude baselines (requires ANTHROPIC_API_KEY)
python -m src.analysis.llm_baseline \
  --data_dir data/processed \
  --output_dir results/figures
```

### 7. View Results

Open `notebooks/02_results.ipynb` for all figures and comparison tables.

## Dataset

- **Source:** [`sebastiandizon/genius-song-lyrics`](https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics) (2.76M songs)
- **Genres:** Rap / Hip-Hop, Pop, Rock, Country, R&B
- **Comparison size:** 2,000 songs per genre (2,500 for Pop), ~8,000 train / 1,000 val / 1,000 test
- **Interpretability size:** 5,000 songs per genre (6,000 for Pop)
- **Preprocessing:** English-only filter by dual CLD3 and fastText agreement, structural annotation stripping ([Verse], [Chorus], etc.), 2,000-character truncation, 20-word minimum length
- **Splits:** Stratified 80/10/10 with validation and test sets balanced to equal counts per class (seed 42)

## Models and Training

All models are fine-tuned with **LoRA** (rank 16, &alpha; = 32, dropout 0.1) via the PEFT library. Adapter parameters are matched at 590K across models so that the comparison isolates architectural differences rather than adapter capacity.

| Model         | Architecture    | Base Params | Trainable                            | LoRA Targets              |
| ------------- | --------------- | ----------- | ------------------------------------ | ------------------------- |
| RoBERTa-base  | Encoder-only    | 125M        | 1.18M (LoRA + 2-layer head)          | query, value              |
| GPT-2         | Decoder-only    | 124M        | 0.59M (LoRA + linear head)           | c_attn                    |
| T5-small      | Encoder-decoder | 60M         | 0.59M (LoRA only, text-to-text mode) | q, v (encoder + decoder)  |

Training uses AdamW with a cosine schedule and 10% linear warmup, maximum sequence length 512, batch size 64, learning rate 2&times;10<sup>-4</sup>, weight decay 0.01, gradient clipping at 1.0, and early stopping with patience 3 on validation macro F1. The interpretability RoBERTa additionally uses expanded LoRA targets (query / key / value), a lower learning rate (1&times;10<sup>-4</sup>), gradient accumulation (effective batch size 64), FP16 mixed precision, and random contiguous word cropping as augmentation.

## Paper

The LaTeX source for the paper is in [`paper/`](paper/). To build:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Limitations

All reported numbers are single-seed point estimates; multi-seed variance is not reported. The architectural comparison is not head-normalized or base-size-normalized. The Genius dataset contains label noise. The 2,000-character truncation disproportionately affects Rap, whose median length is at the cap. The five-genre taxonomy is coarse. LLM baselines are evaluated on a 250-song subset (50 per genre) and should be read as illustrative rather than definitive.

## License

MIT. See [LICENSE](LICENSE).
