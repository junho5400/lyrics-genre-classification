#!/usr/bin/env python
"""GPT-2 decoder-only classifier with LoRA."""

from transformers import AutoTokenizer, GPT2ForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType


def build_model(cfg: dict, num_labels: int = 5):
    model_name = cfg["models"]["gpt2"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules_decoder"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


def get_tokenizer(cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["gpt2"]["name"])
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
