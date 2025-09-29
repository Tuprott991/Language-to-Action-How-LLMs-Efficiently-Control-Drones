from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, use_fast=False)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto", load_in_4bit=True, trust_remote_code=True,
                                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = PeftModel.from_pretrained(base, "outputs/qwen2.5-lora-planner/checkpoint-552")
prompt = "Hello who are you?"
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
input_ids = inputs.input_ids.to(model.device)
attention_mask = inputs.attention_mask.to(model.device)
out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=200, do_sample=False, temperature=0.0)
result = tokenizer.decode(out[0], skip_special_tokens=True)
print(result[len(prompt):].strip())
#asd