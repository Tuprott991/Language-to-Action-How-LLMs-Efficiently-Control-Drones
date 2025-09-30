# interpreter.py - Compatible with gym-pybullet-drones

import re
import numpy as np
import time
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType

class DroneInterpreter:
    def __init__(self, env, grid_shape=(10, 10, 3), cell_size=0.03, control_freq=30):
        """
        env: gym_pybullet_drones environment instance
        grid_shape: (x_cells, y_cells, z_cells)
        cell_size: meters per grid cell
        control_freq: control frequency in Hz (should match env.CTRL_FREQ)
        """
        self.env = env
        self.grid_shape = grid_shape
        self.cell_size = cell_size
        self.control_freq = control_freq
        
        # Detect action type from environment
        # CtrlAviary uses PID, VelocityAviary uses VEL, etc.
        if hasattr(env, 'ACT_TYPE'):
            self.action_type = env.ACT_TYPE
        else:
            # Default based on class name
            env_class_name = env.__class__.__name__
            if 'Velocity' in env_class_name:
                self.action_type = ActionType.VEL
            elif 'RPM' in env_class_name:
                self.action_type = ActionType.RPM
            else:
                self.action_type = ActionType.PID  # Default for CtrlAviary
        
        self.partial_buffer = ""
        self.planned_actions = []
        self.execution_log = []
        
        # PID gains for position control
        self.Kp = np.array([1.0, 1.0, 2.0])  # Position gains
        self.Kd = np.array([0.5, 0.5, 1.0])  # Velocity gains
        
    def grid_to_world(self, gx, gy, gz):
        """Convert grid coordinates to world coordinates."""
        x = (gx - self.grid_shape[0] / 2) * self.cell_size
        y = (gy - self.grid_shape[1] / 2) * self.cell_size
        z = gz * self.cell_size
        return np.array([x, y, z])

    def process_stream_chunk(self, chunk: str):
        """Handle partial streaming of LLM response."""
        self.partial_buffer += chunk
        while ";" in self.partial_buffer:
            action_str, self.partial_buffer = self.partial_buffer.split(";", 1)
            action_str = action_str.strip()
            if action_str:
                self._parse_and_queue_action(action_str)

    def process_final_response(self, response: str):
        """Process full LLM plan."""
        self.planned_actions = []
        self.partial_buffer = ""
        
        actions = self._parse_response(response)
        self.planned_actions.extend(actions)
        
        self.execute_plan()

    def _parse_response(self, response: str):
        """Parse the full response string into individual actions."""
        actions = []
        parts = []
        current = ""
        brace_depth = 0
        
        for char in response:
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
            elif char == ';' and brace_depth == 0:
                if current.strip():
                    parts.append(current.strip())
                current = ""
                continue
            current += char
        
        if current.strip():
            parts.append(current.strip())
        
        for part in parts:
            parsed = self._parse_action_part(part)
            if parsed:
                actions.extend(parsed)
        
        return actions

    def _parse_action_part(self, part: str):
        """Parse a single action part."""
        actions = []
        
        # Handle if blocks
        if_match = re.match(r'if\s+True\s*\{(.*?)\}', part, re.DOTALL)
        if if_match:
            inner = if_match.group(1)
            inner_actions = self._parse_response(inner)
            return inner_actions
        
        # Handle loops
        loop_match = re.match(r'\[(\d+)\]\s*\{(.*?)\}', part, re.DOTALL)
        if loop_match:
            repeat = int(loop_match.group(1))
            inner = loop_match.group(2)
            inner_actions = self._parse_response(inner)
            for _ in range(repeat):
                actions.extend(inner_actions)
            return actions
        
        actions.append(part)
        return actions

    def _parse_and_queue_action(self, action: str):
        """Parse and queue a single action."""
        parsed = self._parse_action_part(action)
        self.planned_actions.extend(parsed)

    def execute_plan(self):
        """Execute all queued actions."""
        print(f"[Interpreter] Executing {len(self.planned_actions)} actions")
        for i, action in enumerate(self.planned_actions):
            print(f"[{i+1}/{len(self.planned_actions)}] {action}")
            self._handle_action(action)
        
        print("[Interpreter] Plan execution complete")
        self.planned_actions = []

    def _handle_action(self, action: str):
        """Interpret a single action string."""
        action = action.strip()
        
        if action.startswith("takeoff"):
            match = re.search(r'takeoff\s*\(\s*([\d.]+)\s*\)', action)
            if match:
                alt = float(match.group(1))
                self._cmd_takeoff(alt)
                self.execution_log.append(f"takeoff({alt})")

        elif action.startswith("goto"):
            match = re.search(r'goto\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', action)
            if match:
                gx, gy, gz = int(match.group(1)), int(match.group(2)), int(match.group(3))
                self._cmd_goto(gx, gy, gz)
                self.execution_log.append(f"goto({gx},{gy},{gz})")

        elif action.startswith("land"):
            self._cmd_land()
            self.execution_log.append("land()")

        elif action.startswith("log"):
            match = re.search(r"log\s*\(\s*['\"](.+?)['\"]\s*\)", action)
            if match:
                msg = match.group(1)
                print(f"[Drone Log] {msg}")
                self.execution_log.append(f"log('{msg}')")

        elif action.startswith("delay"):
            match = re.search(r'delay\s*\(\s*([\d.]+)\s*\)', action)
            if match:
                sec = float(match.group(1))
                time.sleep(sec)
                self.execution_log.append(f"delay({sec})")

        elif action.startswith("re_plan"):
            print("[Interpreter] Re-planning requested")
            self.execution_log.append("re_plan()")

        else:
            print(f"[Interpreter] Unknown action: {action}")

    def _get_current_state(self):
        """Get current drone state from environment."""
        try:
            # Method 1: Try _getDroneStateVector (most direct)
            if hasattr(self.env, '_getDroneStateVector'):
                state = self.env._getDroneStateVector(0)
                pos = state[0:3]
                vel = state[10:13]
                return pos, vel
        except Exception as e:
            pass
        
        try:
            # Method 2: Try getting from last observation
            if hasattr(self.env, '_computeObs'):
                obs = self.env._computeObs()
                # CtrlAviary returns dict with single key
                if isinstance(obs, dict):
                    # Get first (and only) drone's observation
                    obs_array = obs[0] if 0 in obs else list(obs.values())[0]
                else:
                    obs_array = obs
                
                # Flatten if needed
                if isinstance(obs_array, np.ndarray):
                    obs_flat = obs_array.flatten()
                    pos = obs_flat[0:3]
                    vel = obs_flat[3:6] if len(obs_flat) >= 6 else np.zeros(3)
                    return pos, vel
        except Exception as e:
            pass
        
        # Method 3: Access PyBullet directly
        try:
            import pybullet as p
            drone_id = self.env.DRONE_IDS[0]
            pos, _ = p.getBasePositionAndOrientation(drone_id)
            vel, _ = p.getBaseVelocity(drone_id)
            return np.array(pos), np.array(vel)
        except Exception as e:
            raise RuntimeError(f"Cannot get drone state from environment: {e}")

    def _compute_control_action(self, target_pos, target_vel=np.zeros(3)):
        """
        Compute control action for CtrlAviary.
        CtrlAviary uses DSLPIDControl which needs RPM commands computed from state.
        """
        current_pos, current_vel = self._get_current_state()
        
        if self.action_type == ActionType.PID:
            # Use the environment's built-in controller
            # We need to call the controller directly
            if hasattr(self.env, 'ctrl'):
                # Get RPMs from the controller
                rpm, _, _ = self.env.ctrl[0].computeControlFromState(
                    control_timestep=self.env.CTRL_TIMESTEP,
                    state=self.env._getDroneStateVector(0),
                    target_pos=target_pos,
                    target_rpy=np.array([0, 0, 0]),  # level flight
                    target_vel=target_vel,
                    target_rpy_rates=np.array([0, 0, 0])
                )
                return rpm
            else:
                # Fallback: manual computation
                # This is a simple PD controller that outputs "thrust"
                pos_error = target_pos - current_pos
                vel_error = target_vel - current_vel
                
                # Proportional-Derivative control
                desired_acc = self.Kp * pos_error + self.Kd * vel_error
                
                # Convert acceleration to "normalized thrust" (0 to 1)
                # Gravity compensation + desired acceleration
                gravity = 9.81
                mass = 0.027  # kg (CF2X mass)
                total_thrust = mass * (gravity + desired_acc[2])
                
                # Normalize and convert to RPM-like values
                # CtrlAviary expects something that gets clipped to [0, MAX_RPM]
                hover_rpm = 10000  # approximate hover RPM
                thrust_to_rpm = hover_rpm / (mass * gravity)
                
                base_rpm = total_thrust * thrust_to_rpm
                
                # Simple attitude control for x,y
                max_tilt = 0.3  # max tilt in radians
                roll = np.clip(desired_acc[1] / gravity, -max_tilt, max_tilt)
                pitch = np.clip(-desired_acc[0] / gravity, -max_tilt, max_tilt)
                
                # Distribute to 4 motors (very simplified)
                rpm = np.array([
                    base_rpm * (1 + pitch - roll),
                    base_rpm * (1 + pitch + roll),
                    base_rpm * (1 - pitch + roll),
                    base_rpm * (1 - pitch - roll)
                ])
                
                rpm = np.clip(rpm, 0, 20000)
                return rpm
                
        elif self.action_type == ActionType.VEL:
            pos_error = target_pos - current_pos
            desired_vel = self.Kp * pos_error
            max_vel = 2.0
            action = np.clip(desired_vel, -max_vel, max_vel)
            return action
            
        elif self.action_type == ActionType.RPM:
            action = np.array([10000, 10000, 10000, 10000])
            return action
            
        else:
            raise ValueError(f"Unsupported action type: {self.action_type}")

    def _cmd_takeoff(self, altitude: float):
        """Takeoff to specified altitude."""
        try:
            current_pos, _ = self._get_current_state()
            target = np.array([current_pos[0], current_pos[1], altitude])
            print(f"  Taking off to altitude {altitude}m")
            self._move_to_target(target, tolerance=0.1)
        except Exception as e:
            print(f"  Error in takeoff: {e}")

    def _cmd_goto(self, gx: int, gy: int, gz: int):
        """Navigate to grid position."""
        if not (0 <= gx < self.grid_shape[0] and 
                0 <= gy < self.grid_shape[1] and 
                0 <= gz < self.grid_shape[2]):
            print(f"  Warning: Grid position ({gx},{gy},{gz}) out of bounds")
            return
        
        target = self.grid_to_world(gx, gy, gz)
        print(f"  Moving to grid ({gx},{gy},{gz}) → world [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
        self._move_to_target(target, tolerance=0.15)

    def _cmd_land(self):
        """Land at current position."""
        try:
            current_pos, _ = self._get_current_state()
            target = np.array([current_pos[0], current_pos[1], 0.05])  # Land just above ground
            print(f"  Landing at ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
            self._move_to_target(target, tolerance=0.1)
        except Exception as e:
            print(f"  Error in landing: {e}")

    def _move_to_target(self, target_pos: np.ndarray, tolerance: float = 0.15, max_steps: int = 500):
        """
        Move to target position using gym-pybullet-drones step function.
        """
        reached_count = 0  # Count consecutive steps within tolerance
        required_count = 10  # Need to be stable for 10 steps
        
        for step in range(max_steps):
            current_pos, _ = self._get_current_state()
            
            # Check if reached
            distance = np.linalg.norm(current_pos - target_pos)
            if distance < tolerance:
                reached_count += 1
                if reached_count >= required_count:
                    print(f"    Reached target (distance: {distance:.3f}m)")
                    break
            else:
                reached_count = 0
            
            # Compute control action
            action = self._compute_control_action(target_pos)
            
            # CtrlAviary expects 2D array: (NUM_DRONES, 4)
            action_array = np.array([action])  # Shape: (1, 4)
            
            # Step environment
            try:
                result = self.env.step(action_array)
                
                # Handle different return formats
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                elif len(result) == 4:
                    obs, reward, done, info = result
                else:
                    raise ValueError(f"Unexpected step return length: {len(result)}")
                
                # Render for visualization
                if hasattr(self.env, 'render'):
                    self.env.render()
                
                # Add small delay for real-time visualization
                import time
                time.sleep(0.01)
                    
                if done:
                    print("    Environment episode ended")
                    break
                    
            except Exception as e:
                print(f"    Step error: {e}")
                import traceback
                traceback.print_exc()
                break
        else:
            current_pos, _ = self._get_current_state()
            distance = np.linalg.norm(current_pos - target_pos)
            print(f"    Warning: Max steps reached, distance: {distance:.3f}m")

    def get_execution_log(self):
        """Return the log of executed actions."""
        return self.execution_log

    def reset(self):
        """Reset interpreter state."""
        self.partial_buffer = ""
        self.planned_actions = []
        self.execution_log = []