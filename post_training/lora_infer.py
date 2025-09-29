from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Paths
base_model_path = "Qwen/Qwen2.5-7B-Instruct"
lora_path = r"outputs/qwen2.5-lora-planner/checkpoint-552"


# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA
model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.float16)

# Merge LoRA for inference
model = model.merge_and_unload()

# Inference function
def generate(prompt, max_tokens=235):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

while True:
    prompt = input("Enter your prompt: ")
    if prompt.lower() in ["exit", "quit"]:
        break
    result = generate(prompt)
    print(result.strip())