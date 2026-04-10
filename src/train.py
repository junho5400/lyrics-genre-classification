#!/usr/bin/env python
"""Unified training script for all three model architectures.

Supports label smoothing, gradient accumulation, mixed-precision (fp16),
and data augmentation — all controlled via the config YAML or CLI overrides.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.data.dataset import get_dataset
from src.models import roberta_classifier, gpt2_classifier, t5_classifier


MODEL_BUILDERS = {
    "roberta": roberta_classifier.build_model,
    "gpt2": gpt2_classifier.build_model,
    "t5": t5_classifier.build_model,
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Classification (RoBERTa / GPT-2)
# ---------------------------------------------------------------------------

def train_classification_epoch(
    model, loader, optimizer, scheduler, device,
    loss_fn, grad_accum_steps=1, scaler=None,
):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    use_amp = scaler is not None

    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(logits, labels) / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels) / grad_accum_steps
            loss.backward()

        total_loss += loss.item() * grad_accum_steps * input_ids.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = total_loss / len(loader.dataset)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


def eval_classification(model, loader, device, loss_fn, scaler=None):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    use_amp = scaler is not None

    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    loss = loss_fn(logits, labels)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


# ---------------------------------------------------------------------------
# Encoder-decoder (T5)
# ---------------------------------------------------------------------------

def train_encoder_decoder_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(loader.dataset)


def eval_encoder_decoder(model, loader, tokenizer, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            genre_labels = batch["genre_label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * input_ids.size(0)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
            )
            for gen_ids, true_label in zip(generated, genre_labels):
                pred_text = t5_classifier.decode_prediction(tokenizer, gen_ids)
                pred_label = t5_classifier.prediction_to_label(pred_text)
                all_preds.append(pred_label)
                all_labels.append(true_label.item())

    avg_loss = total_loss / len(loader.dataset)
    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)

    valid_mask = all_preds_arr >= 0
    if valid_mask.sum() > 0:
        acc = (all_preds_arr[valid_mask] == all_labels_arr[valid_mask]).mean()
        f1 = f1_score(all_labels_arr[valid_mask], all_preds_arr[valid_mask], average="macro", zero_division=0)
    else:
        acc, f1 = 0.0, 0.0

    invalid_count = (~valid_mask).sum()
    return avg_loss, acc, f1, invalid_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train lyrics genre classifier")
    parser.add_argument("--model", choices=["roberta", "gpt2", "t5"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="results/checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tc = cfg["training"]

    epochs = args.epochs or tc["epochs"]
    batch_size = args.batch_size or tc["batch_size"]
    lr = args.lr or tc["learning_rate"]
    weight_decay = args.weight_decay if args.weight_decay is not None else tc["weight_decay"]
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else tc.get("label_smoothing", 0.0)
    max_seq_len = tc["max_seq_len"]
    patience = tc["early_stopping_patience"]
    grad_accum_steps = tc.get("gradient_accumulation_steps", 1)
    use_fp16 = tc.get("fp16", False)
    augment = tc.get("augment", False)

    device = get_device()
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"  lr={lr}  wd={weight_decay}  ls={label_smoothing}  "
          f"bs={batch_size}  accum={grad_accum_steps}  fp16={use_fp16}  augment={augment}")
    print(f"  LoRA: r={cfg['lora']['rank']}  alpha={cfg['lora']['alpha']}  dropout={cfg['lora']['dropout']}")

    model_type = cfg["models"][args.model]["type"]
    is_seq2seq = model_type == "encoder-decoder"

    print(f"\nBuilding {args.model} model ...")
    model, tokenizer = MODEL_BUILDERS[args.model](cfg)
    model = model.to(device)

    print("Loading datasets ...")
    train_ds = get_dataset("train", tokenizer, model_type, args.data_dir, max_seq_len, augment=augment)
    val_ds = get_dataset("validation", tokenizer, model_type, args.data_dir, max_seq_len, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = int(total_steps * tc["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = None
    if use_fp16 and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        print("  Mixed-precision training enabled (fp16)")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining for up to {epochs} epochs (patience={patience}) ...\n")

    for epoch in range(1, epochs + 1):
        print(f"--- Epoch {epoch}/{epochs} ---")

        if is_seq2seq:
            train_loss = train_encoder_decoder_epoch(model, train_loader, optimizer, scheduler, device)
            val_loss, val_acc, val_f1, n_invalid = eval_encoder_decoder(model, val_loader, tokenizer, device)
            _, train_acc, train_f1, _ = eval_encoder_decoder(model, train_loader, tokenizer, device)
            if n_invalid > 0:
                print(f"  Warning: {n_invalid} unrecognised predictions on val set")
        else:
            train_loss, train_acc, train_f1 = train_classification_epoch(
                model, train_loader, optimizer, scheduler, device,
                loss_fn, grad_accum_steps, scaler,
            )
            val_loss, val_acc, val_f1 = eval_classification(model, val_loader, device, loss_fn, scaler)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train — loss: {train_loss:.4f}  acc: {train_acc:.4f}  f1: {train_f1:.4f}")
        print(f"  Val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}  f1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model.save_pretrained(str(out_dir / "best"))
            tokenizer.save_pretrained(str(out_dir / "best"))
            print(f"  >>> Saved best model (val_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after {epoch} epochs")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    run_cfg = {
        "model": args.model, "config": args.config,
        "lr": lr, "weight_decay": weight_decay, "label_smoothing": label_smoothing,
        "batch_size": batch_size, "grad_accum_steps": grad_accum_steps,
        "fp16": use_fp16, "augment": augment, "best_val_f1": best_val_f1,
        "lora_rank": cfg["lora"]["rank"], "lora_alpha": cfg["lora"]["alpha"],
        "lora_dropout": cfg["lora"]["dropout"],
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    print(f"Checkpoints and history saved to {out_dir}/")


if __name__ == "__main__":
    main()
