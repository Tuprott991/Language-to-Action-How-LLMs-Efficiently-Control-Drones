"""
Drone Path Planning with Vertex AI and gym-pybullet-drones
"""

from google.oauth2.service_account import Credentials
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType
from interpreter import DroneInterpreter
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import numpy as np
import json

# ============================================
# Setup Vertex AI
# ============================================
credentials_path = "prusandbx-nprd-uat-kw1ozq-dcfe6900463a.json"
credentials = Credentials.from_service_account_file(
    credentials_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

PROJECT_ID = "prusandbx-nprd-uat-kw1ozq"
REGION = "asia-southeast1"

vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)

# ============================================
# System Prompt for Drone Planning
# ============================================
system_prompt = """You are an expert drone flight planner. Your task is to generate low-level action plans for drone navigation based on the given constraints.

**Output Format:**
Generate a single-line action plan using these commands:
- takeoff(altitude) - Take off to specified altitude in meters
- goto(x, y, z) - Navigate to grid coordinates (x, y, z)
- land() - Land at current position
- log('message') - Log a message
- delay(seconds) - Wait for specified seconds
- re_plan() - Request replanning

**Advanced Structures:**
- Conditional: if True { action1; action2; }
- Loop: [N] { action1; action2; } (repeat N times)

**Rules:**
1. Always start with takeoff() and end with land()
2. Avoid no-fly zones (all altitudes)
3. Navigate around obstacles at specific (x,y,z) positions
4. Use scanning patterns: [2] { log('scanning'); delay(1); }
5. Stay within grid boundaries
6. Use smooth paths with waypoints
7. Output MUST be a single line with semicolons between commands
8. IMPORTANT: Always complete the full path from start to end
9. End every command with a semicolon, including the last land()

**Example Input:**
Task: Fly from (0, 0, 0) to (5, 5, 2)
Grid shape: [10, 10, 3]
No-fly zones: No-fly zone at (2,2) (all altitudes)
Obstacles: Obstacle at (3,3,1)

**Example Output:**
takeoff(2.5); goto(0,0,0); goto(1,1,1); goto(2,1,1); goto(3,1,1); goto(4,2,1); goto(5,3,2); goto(5,4,2); goto(5,5,2); land();

Remember: Generate the COMPLETE path with ALL waypoints from start to destination, then land().

Now generate the action plan for the given task."""

# ============================================
# Initialize Vertex AI Model
# ============================================
def create_vertex_model():
    """Create and configure Vertex AI Generative Model."""
    model = GenerativeModel(
        model_name="gemini-2.5-flash",  # or "gemini-1.5-pro-002" for better quality
        system_instruction=system_prompt
    )
    return model

# ============================================
# Generate Plan from Vertex AI
# ============================================
def generate_plan_vertex(task_description, model):
    """
    Generate drone flight plan using Vertex AI.
    
    Args:
        task_description: Task prompt with grid info, obstacles, etc.
        model: Vertex AI GenerativeModel instance
    
    Returns:
        Generated action plan string
    """
    generation_config = GenerationConfig(
        temperature=0.4,  # Lower for more deterministic output
        top_p=0.9,
        top_k=40,
        max_output_tokens=800,  # Increased to allow full plans
        candidate_count=1,
    )
    
    print("Generating plan from Vertex AI...")
    response = model.generate_content(
        task_description,
        generation_config=generation_config,
    )
    
    plan = response.text.strip()
    print(f"\nGenerated plan:\n{plan}\n")
    
    return plan

# ============================================
# Generate Plan with Streaming
# ============================================
def generate_plan_vertex_streaming(task_description, model, interpreter):
    """
    Generate plan with streaming response.
    
    Args:
        task_description: Task prompt
        model: Vertex AI model
        interpreter: DroneInterpreter instance for processing chunks
    """
    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=500,
    )
    
    print("Generating plan (streaming)...")
    full_response = ""
    
    response_stream = model.generate_content(
        task_description,
        generation_config=generation_config,
        stream=True
    )
    
    for chunk in response_stream:
        if chunk.text:
            print(chunk.text, end='', flush=True)
            full_response += chunk.text
            # Process streaming chunks
            interpreter.process_stream_chunk(chunk.text)
    
    print("\n")
    return full_response

# ============================================
# Setup Environment
# ============================================
def setup_environment():
    """Create and initialize gym-pybullet-drones environment."""
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0, 0, 0.1]]),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=30,
        gui=True,  # Set to False for headless
        record=False
    )
    return env

# ============================================
# Format Task Prompt
# ============================================
def format_task_prompt(
    start_pos,
    end_pos,
    grid_shape=(10, 10, 3),
    no_fly_zones=None,
    obstacles=None,
    task_description=None
):
    """
    Format a task prompt for the LLM.
    
    Args:
        start_pos: (x, y, z) starting position
        end_pos: (x, y, z) ending position
        grid_shape: Grid dimensions
        no_fly_zones: List of (x, y) no-fly coordinates
        obstacles: List of (x, y, z) obstacle coordinates
        task_description: Optional custom task description
    """
    if task_description is None:
        task_description = f"Fly from {start_pos} to {end_pos}"
    
    prompt = f"{task_description}\n"
    prompt += f"Grid shape: {list(grid_shape)}\n"
    
    if no_fly_zones:
        zones_str = "; ".join([f"No-fly zone at ({x},{y}) (all altitudes)" 
                               for x, y in no_fly_zones])
        prompt += f"No-fly zones: {zones_str}\n"
    
    if obstacles:
        obs_str = "; ".join([f"Obstacle at ({x},{y},{z})" 
                            for x, y, z in obstacles])
        prompt += f"Abstract map: {obs_str}\n"
    
    prompt += "Produce the low-level action plan (one-line) as the output."
    
    return prompt

# ============================================
# Main Execution Functions
# ============================================
def execute_task_non_streaming(env, interpreter, model, task_prompt):
    """Execute a task without streaming."""
    print("=" * 60)
    print("TASK:", task_prompt.split('\n')[0])
    print("=" * 60)
    
    # Generate plan
    plan = generate_plan_vertex(task_prompt, model)
    
    # Execute plan
    print("Executing plan...")
    env.reset()
    interpreter.reset()
    interpreter.process_final_response(plan)
    
    # Get execution log
    log = interpreter.get_execution_log()
    print("\nExecution log:")
    for entry in log:
        print(f"  ✓ {entry}")
    
    return plan, log

def execute_task_streaming(env, interpreter, model, task_prompt):
    """Execute a task with streaming."""
    print("=" * 60)
    print("TASK:", task_prompt.split('\n')[0])
    print("=" * 60)
    
    # Generate and stream plan
    env.reset()
    interpreter.reset()
    plan = generate_plan_vertex_streaming(task_prompt, model, interpreter)
    
    # Execute queued actions
    print("\nExecuting streamed plan...")
    interpreter.execute_plan()
    
    # Get execution log
    log = interpreter.get_execution_log()
    print("\nExecution log:")
    for entry in log:
        print(f"  ✓ {entry}")
    
    return plan, log

# ============================================
# Example Usage
# ============================================
def main():
    # Setup
    print("Initializing environment...")
    env = setup_environment()
    interpreter = DroneInterpreter(
        env=env,
        grid_shape=(10, 10, 3),
        cell_size=0.3,
        control_freq=30
    )
    
    print("Initializing Vertex AI model...")
    model = create_vertex_model()
    
    # Example 1: Simple navigation
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple Navigation")
    print("="*60)
    
    task1 = format_task_prompt(
        start_pos=(0, 0, 0),
        end_pos=(5, 5, 2),
        no_fly_zones=[(2, 2), (3, 3)],
        obstacles=[(4, 4, 1)]
    )
    
    plan1, log1 = execute_task_non_streaming(env, interpreter, model, task1)
    
    # Example 2: Complex navigation with scanning
    print("\n" + "="*60)
    print("EXAMPLE 2: Survey Mission")
    print("="*60)
    
    task2 = format_task_prompt(
        start_pos=(0, 0, 0),
        end_pos=(8, 8, 2),
        grid_shape=(10, 10, 3),
        no_fly_zones=[(4, 4), (5, 5)],
        obstacles=[(3, 3, 1), (6, 6, 2)],
        task_description="Survey the area from (0, 0, 0) to (8, 8, 2), perform scanning"
    )
    
    plan2, log2 = execute_task_streaming(env, interpreter, model, task2)
    
    # Example 3: Custom task with replanning
    print("\n" + "="*60)
    print("EXAMPLE 3: Mission with Uncertainty")
    print("="*60)
    
    task3 = """
    Mission: Navigate from (1, 1, 0) to (7, 7, 2), replan if obstacles detected.
    Grid shape: [10, 10, 3]
    No-fly zones: No-fly zone at (4,4) (all altitudes); No-fly zone at (5,5) (all altitudes)
    Abstract map: Obstacle at (3,3,1); Obstacle at (6,6,2)
    Produce the low-level action plan (one-line) as the output.
    """
    
    plan3, log3 = execute_task_non_streaming(env, interpreter, model, task3)
    
    # Cleanup
    print("\n" + "="*60)
    print("All tasks completed!")
    print("="*60)
    env.close()

# ============================================
# Interactive Mode
# ============================================
def interactive_mode():
    """Run in interactive mode - user inputs tasks."""
    env = setup_environment()
    interpreter = DroneInterpreter(env, grid_shape=(10, 10, 3), cell_size=0.3)
    model = create_vertex_model()
    
    print("\n" + "="*60)
    print("INTERACTIVE DRONE PLANNER")
    print("="*60)
    print("Enter task descriptions or 'quit' to exit\n")
    
    while True:
        print("\nEnter task (or 'quit'):")
        print("Example: Fly from (0,0,0) to (5,5,2), avoid (2,2)")
        task = input("> ").strip()
        
        if task.lower() in ['quit', 'exit', 'q']:
            break
        
        if not task:
            continue
        
        # Parse simple task format or use as-is
        if "from" in task.lower() and "to" in task.lower():
            # Try to parse coordinates
            import re
            coords = re.findall(r'\((\d+),(\d+),(\d+)\)', task)
            if len(coords) >= 2:
                start = tuple(map(int, coords[0]))
                end = tuple(map(int, coords[1]))
                
                # Look for obstacles
                avoid_match = re.findall(r'avoid\s*\((\d+),(\d+)\)', task)
                no_fly = [tuple(map(int, m)) for m in avoid_match]
                
                task_prompt = format_task_prompt(start, end, no_fly_zones=no_fly)
            else:
                task_prompt = task
        else:
            task_prompt = task
        
        try:
            plan, log = execute_task_non_streaming(env, interpreter, model, task_prompt)
            print("\n✓ Task completed successfully!")
        except Exception as e:
            print(f"\n✗ Error: {e}")
    
    env.close()
    print("Goodbye!")

if __name__ == "__main__":
    # Choose mode
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        main()