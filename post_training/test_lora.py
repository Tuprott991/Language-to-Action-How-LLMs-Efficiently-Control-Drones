from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
import torch

# 1️⃣ Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, use_fast=False)

# 2️⃣ Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",    # or "fp4", depending on your setup
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 3️⃣ Prepare model for LoRA (k-bit)
base_model = prepare_model_for_kbit_training(base_model)

# 4️⃣ Load LoRA
lora_path = r"C:\outputs\qwen2.5-lora-planner\checkpoint-552"
model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.float16)

# 5️⃣ Verify LoRA
print("LoRA config:", model.peft_config)
for name, param in model.named_parameters():
    if "lora" in name.lower():
        print(name, param.shape)
        break
