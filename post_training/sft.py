# train_qwen_lora.py
import os
import math
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
import argparse

@dataclass
class Args:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    train_file: str = field(default="data/train.jsonl")
    valid_file: str = field(default="data/valid.jsonl")
    output_dir: str = field(default="outputs/qwen2.5-lora-planner")
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    num_train_epochs: int = field(default=3)
    learning_rate: float = field(default=2e-4)
    logging_steps: int = field(default=50)
    save_steps: int = field(default=500)
    max_length: int = field(default=512)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)
    fp16: bool = field(default=True)
    seed: int = field(default=42)

def build_prompt_response_pair(example):
    # dataset entries contain "prompt" and "response"
    prompt = example["prompt"].strip()
    response = example["response"].strip()
    # We will concatenate: prompt + " " + response. The model should predict response tokens via labels
    full = prompt + " " + response
    return {"text": full, "prompt": prompt, "response": response}

def tokenize_and_split(examples, tokenizer, max_length):
    # examples["text"] is the concatenation prompt + response
    outputs = tokenizer(examples["text"],
                        truncation=True,
                        max_length=max_length,
                        padding=False)
    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"]

    # We need to create labels such that model is trained to generate the response only.
    # Approach: locate prompt length in tokenized input, set labels for prompt tokens to -100.
    labels = []
    for idx, input_id in enumerate(input_ids):
        # Re-tokenize prompt to know its length
        prompt_tokens = tokenizer(examples["prompt"][idx], truncation=True, max_length=max_length)["input_ids"]
        plen = len(prompt_tokens)
        lbl = [-100] * plen + input_id[plen:]
        # pad/truncate label to same length
        lbl = lbl[:len(input_id)]
        # when label shorter than input, fill remainder with -100
        if len(lbl) < len(input_id):
            lbl += [-100] * (len(input_id) - len(lbl))
        labels.append(lbl)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    args = parser.parse_args()
    cfg = Args()

    torch.manual_seed(cfg.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "left"  # causal models sometimes benefit from left padding
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (jsonl created earlier)
    data_files = {"train": cfg.train_file, "validation": cfg.valid_file}
    raw_datasets = load_dataset("json", data_files=data_files)
    # map to combined text
    raw_datasets = raw_datasets.map(build_prompt_response_pair, remove_columns=raw_datasets["train"].column_names)

    # Tokenize / prepare
    tokenized = raw_datasets.map(
        lambda examples: tokenize_and_split(examples, tokenizer, cfg.max_length),
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    # model load in 4-bit using bitsandbytes
    print("Loading model in 4-bit mode (bitsandbytes)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Prepare for k-bit training then wrap with PEFT LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "gate_proj"],  # common target modules; adjust if model different
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data collator: we will pad to max_length and let labels contain -100 for prompt tokens
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.save_steps,
        save_total_limit=3,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    # Save tokenizer (optional)
    tokenizer.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    main()
