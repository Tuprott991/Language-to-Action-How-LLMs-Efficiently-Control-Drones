from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, use_fast=False)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
model = PeftModel.from_pretrained(base, "outputs/qwen2.5-lora-planner/checkpoint-552")

while True:
    prompt = input("Enter your prompt: ")
    if prompt.lower() in ['exit', 'quit']:
        break
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # generate
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # decode only generated tokens
    generated_ids = out[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(result.strip())
