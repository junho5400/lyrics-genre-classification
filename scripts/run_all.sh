#!/usr/bin/env bash
# Lyric-Based Genre Classification — Full Pipeline
# Run from the project root: student/final_project/
set -euo pipefail

echo "============================================"
echo "  Lyrics Genre Classification Pipeline"
echo "============================================"

# -----------------------------------------------
# Phase 1: Architecture Comparison (default config)
# -----------------------------------------------

echo ""
echo "[1/10] Preparing dataset (Phase 1) ..."
python -m src.data.download_and_prepare

echo ""
echo "[2/10] Training RoBERTa (baseline) ..."
python -m src.train --model roberta

echo ""
echo "[3/10] Training GPT-2 ..."
python -m src.train --model gpt2

echo ""
echo "[4/10] Training T5 ..."
python -m src.train --model t5

echo ""
echo "[5/10] Evaluating all 3 models ..."
python -m src.evaluate

# -----------------------------------------------
# Phase 2: Optimized RoBERTa + Interpretability
# -----------------------------------------------

echo ""
echo "[6/10] Preparing larger dataset (Phase 2) ..."
python -m src.data.download_and_prepare --config configs/roberta_optimized.yaml

echo ""
echo "[7/10] Training RoBERTa (optimized) ..."
python -m src.train --model roberta \
    --config configs/roberta_optimized.yaml \
    --data_dir data/processed_phase2 \
    --output_dir results/checkpoints_optimized

echo ""
echo "[8/10] Evaluating optimized RoBERTa ..."
python -m src.evaluate \
    --models roberta \
    --data_dir data/processed_phase2 \
    --checkpoint_dir results/checkpoints_optimized \
    --output_dir results/figures_optimized

echo ""
echo "[9/10] Embedding visualisation (optimized RoBERTa) ..."
python -m src.analysis.embedding_viz \
    --models roberta \
    --data_dir data/processed_phase2 \
    --checkpoint_dir results/checkpoints_optimized \
    --output_dir results/figures_optimized

echo ""
echo "[10/12] Contrastive attribution + misclassification analysis ..."
python -m src.analysis.contrastive_analysis \
    --model roberta \
    --data_dir data/processed_phase2 \
    --checkpoint_dir results/checkpoints_optimized \
    --output_dir results/figures_optimized

echo ""
echo "[11/12] Layer-wise probing ..."
python -m src.analysis.layer_probing \
    --model roberta \
    --data_dir data/processed_phase2 \
    --checkpoint_dir results/checkpoints_optimized \
    --output_dir results/figures_optimized

echo ""
echo "============================================"
echo "  GPU Pipeline complete!"
echo "  Phase 1 results: results/figures/"
echo "  Phase 2 results: results/figures_optimized/"
echo ""
echo "  Now run locally (no GPU needed):"
echo "    python -m src.analysis.genre_similarity"
echo "    python -m src.analysis.llm_baseline"
echo "============================================"
