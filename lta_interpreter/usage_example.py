from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType
from interpreter import DroneInterpreter
import numpy as np

# Create environment with PID control (recommended for high-level control)
env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    initial_xyzs=np.array([[0, 0, 0.1]]),
    physics=Physics.PYB,
    pyb_freq=240,
    ctrl_freq=30,  # Control frequency
    gui=True,  # Set to True to visualize
    record=False,
    act=ActionType.PID  # Use PID control for easier high-level commands
)

# Create interpreter
interpreter = DroneInterpreter(
    env=env,
    grid_shape=(10, 10, 3),
    cell_size=1.0,  # 1 meter per grid cell
    control_freq=30
)

# Reset environment
obs, info = env.reset()

# Example 1: Execute a simple plan
print("Example 1: Simple navigation")
plan = "takeoff(1.5); goto(5,5,1); goto(3,3,1); land();"
interpreter.process_final_response(plan)

# Reset for next example
env.reset()
interpreter.reset()

# Example 2: Complex plan with loops and conditions
print("\nExample 2: Complex plan")
plan = """
takeoff(2.0);
if True {
    goto(4,4,2);
    goto(5,5,2);
}
[2] {
    log('scanning');
    delay(0.5);
}
goto(2,2,1);
land();
"""
interpreter.process_final_response(plan)

# Get execution log
log = interpreter.get_execution_log()
print("\nExecution log:")
for entry in log:
    print(f"  - {entry}")

env.close()

# ============================================
# Example with LLM integration
# ============================================

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

def setup_llm():
    """Load your fine-tuned model."""
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        use_fast=False
    )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(base, "outputs/qwen2.5-lora-planner")
    model.eval()
    
    return model, tokenizer

def llm_plan_and_execute(env, interpreter, task_description):
    """Get plan from LLM and execute it."""
    model, tokenizer = setup_llm()
    
    # Format prompt
    messages = [{"role": "user", "content": task_description}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate plan
    print("Generating plan from LLM...")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = out[0][inputs['input_ids'].shape[1]:]
    plan = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"Generated plan: {plan}")
    
    # Execute plan
    interpreter.process_final_response(plan)
    
    return plan

# Example usage
if __name__ == "__main__":
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0, 0, 0.1]]),
        physics=Physics.PYB,
        ctrl_freq=30,
        gui=True,
        act=ActionType.PID
    )
    
    interpreter = DroneInterpreter(env, grid_shape=(10, 10, 3), cell_size=1.0)
    env.reset()
    
    task = """
    Head from (0, 0, 0) to (5, 5, 2).
    Grid shape: [10, 10, 3]
    No-fly zones: No-fly zone at (3,3) (all altitudes)
    Produce the plan at one-line.
    """
    
    plan = llm_plan_and_execute(env, interpreter, task)
    
    env.close()