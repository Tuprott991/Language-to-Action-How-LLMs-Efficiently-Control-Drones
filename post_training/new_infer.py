from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", 
    trust_remote_code=True, 
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token

# Load base model in 4-bit (must match training config)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, "outputs_2/qwen2.5-lora-planner")
model.eval()

print("\nModel loaded successfully!")
print("=" * 50)

# Test generation
while True:
    prompt = input("\nEnter your prompt (or 'exit' to quit): ")
    if prompt.lower() in ["exit", "quit"]:
        break
    
    # Format with chat template
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    print("\nGenerating response...")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_ids = out[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print("\nResponse:")
    print(result.strip())
    print("=" * 50)