# train_qwen_lora.py
import torch
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

@dataclass
class Args:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    train_file: str = field(default="data/train.jsonl")
    valid_file: str = field(default="data/valid.jsonl")
    output_dir: str = field(default="outputs/qwen2.5-lora-planner")
    per_device_train_batch_size: int = field(default=3)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    num_train_epochs: int = field(default=2)
    learning_rate: float = field(default=2e-4)
    max_length: int = field(default=512)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    fp16: bool = field(default=True)
    seed: int = field(default=42)

cfg = Args()
torch.manual_seed(cfg.seed)

# ---------------------------
# Load tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ---------------------------
# Load dataset
# ---------------------------
data_files = {"train": cfg.train_file, "validation": cfg.valid_file}
raw_datasets = load_dataset("json", data_files=data_files)

# ---------------------------
# Preprocess function
# ---------------------------
def preprocess_function(example):
    # Input is the prompt
    input_text = example["prompt"]
    model_input = tokenizer(
        input_text,
        max_length=cfg.max_length,
        truncation=True,
        padding="max_length",
    )

    # Output is the response
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["response"],
            max_length=cfg.max_length,
            truncation=True,
            padding="max_length",
        )

    model_input["labels"] = labels["input_ids"]
    return model_input

tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=False,
    remove_columns=raw_datasets["train"].column_names
)

# ---------------------------
# Load model in 4-bit with BitsAndBytesConfig
# ---------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading model in 4-bit mode...")
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name_or_path,
    quantization_config=bnb_config,
    device_map={"": 0},  # single GPU
    trust_remote_code=True,
)

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "gate_proj"],
    lora_dropout=cfg.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------
# Training arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir=cfg.output_dir,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    per_device_eval_batch_size=cfg.per_device_eval_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    num_train_epochs=cfg.num_train_epochs,
    learning_rate=cfg.learning_rate,
    fp16=cfg.fp16,
    logging_steps=50,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

# ---------------------------
# Trainer
# ---------------------------

num_train_samples = int(0.25 * len(tokenized_datasets["train"]))
train_subset = tokenized_datasets["train"].select(range(num_train_samples))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)


# ---------------------------
# Train
# ---------------------------
trainer.train()
trainer.save_model(cfg.output_dir)
tokenizer.save_pretrained(cfg.output_dir)
