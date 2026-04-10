#!/usr/bin/env python
"""RoBERTa-base encoder-only classifier with LoRA."""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType


def build_model(cfg: dict, num_labels: int = 5):
    model_name = cfg["models"]["roberta"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules_encoder"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


def get_tokenizer(cfg: dict):
    return AutoTokenizer.from_pretrained(cfg["models"]["roberta"]["name"])
