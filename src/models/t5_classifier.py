#!/usr/bin/env python
"""T5-small encoder-decoder classifier with LoRA (text-to-text formulation)."""

from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


LABEL_NAMES = {0: "rap", 1: "pop", 2: "rock", 3: "country", 4: "r&b"}
NAME_TO_LABEL = {v: k for k, v in LABEL_NAMES.items()}


def build_model(cfg: dict, num_labels: int = 5):
    model_name = cfg["models"]["t5"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules_t5"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


def get_tokenizer(cfg: dict):
    return AutoTokenizer.from_pretrained(cfg["models"]["t5"]["name"])


def decode_prediction(tokenizer, generated_ids) -> str:
    """Decode T5 output tokens back to a genre string."""
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()


def prediction_to_label(pred_text: str) -> int:
    """Map decoded text to integer label, defaulting to -1 if unrecognised."""
    # Handle common variations
    cleaned = pred_text.strip().lower()
    if cleaned in NAME_TO_LABEL:
        return NAME_TO_LABEL[cleaned]
    for name, label in NAME_TO_LABEL.items():
        if name in cleaned:
            return label
    return -1
