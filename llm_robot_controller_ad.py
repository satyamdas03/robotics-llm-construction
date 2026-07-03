import pybullet as p
import pybullet_data
import time
import http.client
import json
import re  
import math 

# --- PART 1: THE "PLANNER" (OLLAMA LLM COMMUNICATION) ---
# (This part is unchanged and working)

def get_llm_plan(user_command):
    """
    Sends a command to the local Ollama LLM and gets a step-by-step plan back.
    """
    print(f"Sending command to LLM: '{user_command}'...")
    
    system_prompt = """
    You are a robot control AI. Your job is to convert a natural language command 
    into a JSON list of simple, sequential function calls.
    
    The available functions are:
    - move_to("target_name")
    - pickup("object_name")
    - drop("location_name")
    
    The available targets, objects, and locations are:
    - "red_ball" (object)
    - "blue_cube" (object)
    - "start_area" (location)
    - "drop_zone" (location)

    RULES:
    1. You MUST respond with *only* a valid JSON list of strings.
    2. The list MUST be the *only* thing in your response.
    3. DO NOT add any explanation, notes, or text before or after the list.
    4. You MUST use double quotes ("") for all strings. DO NOT use single quotes.
    5. Your entire response MUST start with [ and end with ].
    
    Example command: "Get the red ball and put it in the drop zone."
    Correct response: ["move_to(\"red_ball\")", "pickup(\"red_ball\")", "move_to(\"drop_zone\")", "drop(\"drop_zone\")"]
    """
    
    conn = http.client.HTTPConnection("localhost", 11434)
    
    payload = {
        "model": "llama3:8b", # Using llama3:8b
        "prompt": f"{system_prompt}\n\nCommand: \"{user_command}\"\nResponse:",
        "stream": False
    }
    
    llm_output_string = "" 
    
    try:
        conn.request("POST", "/api/generate", json.dumps(payload))
        response = conn.getresponse()
        
        if response.status != 200:
            print(f"Error from Ollama: {response.status} {response.reason}")
            return []
            
        response_body = response.read().decode('utf-8')
        conn.close()
        
        response_data = json.loads(response_body)
        llm_output_string = response_data.get('response', '[]')
        
        plan = re.findall(r'(\w+\(.*?\))', llm_output_string)
        
        if not plan:
            print(f"LLM Response did not contain any valid function calls: {llm_output_string}")
            return []
        
        print(f"LLM generated plan: {plan}")
        return plan
        
    except Exception as e:
        print(f"Error communicating with Ollama or parsing plan: {e}")
        print(f"Raw LLM Output was: {llm_output_string}")
        print("---")
        print("Is Ollama running? Have you run 'ollama pull llama3:8b'?")
        print("---")
        return []

# --- PART 2: THE "SIMULATOR & SKILLS" (PYBULLET ROBOTICS) ---
# (This section is HEAVILY modified)

object_ids = {
    "robot": None,
    "current_constraint": None
}

WORLD_KNOWLEDGE = {
    "red_ball": {"pos": [2, 2, 0.05], "id": None},
    "blue_cube": {"pos": [2, -2, 0.05], "id": None},
    "drop_zone": {"pos": [-2, 0, 0.01], "id": None},
    "start_area": {"pos": [0, 0, 0.05], "id": None}
}

def setup_simulation():
    print("Setting up simulation...")
    try:
        physicsClient = p.connect(p.GUI)
        print("Connected to new PyBullet GUI.")
    except p.error:
        physicsClient = p.connect(p.DIRECT)
        print("Could not connect to GUI, connected to DIRECT.")
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    robot_start_pos = WORLD_KNOWLEDGE["start_area"]["pos"]
    object_ids["robot"] = p.loadURDF("r2d2.urdf", robot_start_pos)
    
    ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
    ball_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
    WORLD_KNOWLEDGE["red_ball"]["id"] = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=ball_shape,
        baseVisualShapeIndex=ball_visual,
        basePosition=WORLD_KNOWLEDGE["red_ball"]["pos"]
    )
    
    cube_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[0, 0, 1, 1])
    WORLD_KNOWLEDGE["blue_cube"]["id"] = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=cube_shape,
        baseVisualShapeIndex=cube_visual,
        basePosition=WORLD_KNOWLEDGE["blue_cube"]["pos"]
    )

    zone_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01])
    zone_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01], rgbaColor=[0, 1, 0, 0.5])
    WORLD_KNOWLEDGE["drop_zone"]["id"] = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=zone_shape,
        baseVisualShapeIndex=zone_visual,
        basePosition=WORLD_KNOWLEDGE["drop_zone"]["pos"]
    )
    print("Simulation setup complete.")

# --- ROBOT "SKILLS" (NEW AND IMPROVED) ---

def get_current_pos(body_id):
    pos, _ = p.getBasePositionAndOrientation(body_id)
    return pos[0], pos[1]

def get_distance(objA_id, objB_id):
    posA = get_current_pos(objA_id)
    posB = get_current_pos(objB_id)
    return math.sqrt((posA[0] - posB[0])**2 + (posA[1] - posB[1])**2)

def move_to(target_name):
    """
    Moves the robot to the target's known location using velocity control.
    """
    print(f"SKILL: Navigating to '{target_name}'...")
    if target_name not in WORLD_KNOWLEDGE:
        print(f"Error: Unknown target '{target_name}'")
        return
        
    target_pos = WORLD_KNOWLEDGE[target_name]["pos"]
    robot_id = object_ids["robot"]
    
    speed = 2.0
    close_enough = 0.3 
    
    for _ in range(1000):
        current_pos, _ = p.getBasePositionAndOrientation(robot_id)
        
        dir_x = target_pos[0] - current_pos[0]
        dir_y = target_pos[1] - current_pos[1]
        distance = math.sqrt(dir_x**2 + dir_y**2)
        
        if distance < close_enough:
            p.resetBaseVelocity(robot_id, [0, 0, 0]) 
            print(f"Arrived at {target_name}.")
            return # E-STOP: We have arrived
        
        norm_x = dir_x / distance
        norm_y = dir_y / distance
        
        linear_velocity = [norm_x * speed, norm_y * speed, 0]
        
        target_yaw = math.atan2(dir_y, dir_x)
        target_orn = p.getQuaternionFromEuler([0, 0, target_yaw])
        
        p.resetBasePositionAndOrientation(robot_id, current_pos, target_orn)
        p.resetBaseVelocity(robot_id, linear_velocity, [0, 0, 0])
        
        p.stepSimulation()
        time.sleep(0.01)
        
    p.resetBaseVelocity(robot_id, [0, 0, 0]) 
    print("Navigation timed out.")

def pickup(object_name):
    """
    Picks up an object. If too far, it will navigate to it first.
    """
    print(f"SKILL: Attempting to pick up '{object_name}'...")
    if object_name not in WORLD_KNOWLEDGE:
        print(f"Error: Unknown object '{object_name}'")
        return
    if object_ids["current_constraint"] is not None:
        print("Robot is already holding an object.")
        return

    robot_id = object_ids["robot"]
    object_id = WORLD_KNOWLEDGE[object_name]["id"]
    
    # --- THIS IS THE FIX ---
    # Proximity check
    distance = get_distance(robot_id, object_id)
    if distance > 0.5: # Must be within 0.5 units
        print(f"Error: '{object_name}' is too far away ({distance:.2f} units). Moving to it first...")
        move_to(object_name) # CALL MOVE_TO AUTOMATICALLY
    # --- END FIX ---

    # Now we are close (or were already), so we can pick up.
    object_ids["current_constraint"] = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=-1, 
        childBodyUniqueId=object_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0.5],
        childFramePosition=[0, 0, 0]
    )
    print(f"Picked up {object_name}.")
    
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)

def drop(location_name):
    """
    Drops the currently held object. If not at the location,
    it will navigate there first.
    """
    print(f"SKILL: Attempting to drop at '{location_name}'...")
    if location_name not in WORLD_KNOWLEDGE:
        print(f"Error: Unknown location '{location_name}'")
        return
    if object_ids["current_constraint"] is None:
        print("Robot is not holding anything.")
        return
    
    robot_id = object_ids["robot"]
    drop_zone_id = WORLD_KNOWLEDGE[location_name]["id"]

    # --- THIS IS THE FIX ---
    # Proximity check
    distance = get_distance(robot_id, drop_zone_id)
    if distance > 0.5: # Must be within 0.5 units
        print(f"Error: Not at '{location_name}' ({distance:.2f} units). Moving to it first...")
        move_to(location_name) # CALL MOVE_TO AUTOMATICALLY
    # --- END FIX ---

    # Now we are at the drop zone, so we can drop.
    p.removeConstraint(object_ids["current_constraint"])
    object_ids["current_constraint"] = None
    print("Dropped object.")
    
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)

# --- PART 3: THE "EXECUTOR" (MAIN SCRIPT LOGIC) ---
# (This part is unchanged and working)

def parse_and_execute(plan):
    """
    Parses the list of function strings and calls the
    corresponding Python function.
    """
    if not plan:
        print("Plan is empty. Nothing to do.")
        return
        
    print(f"--- EXECUTING PLAN ---")
    
    for step in plan:
        try:
            function_name = step.split('(', 1)[0].strip()
            
            match = re.search(r'\((.*)\)', step)
            if not match:
                print(f"Could not find parameters in step: {step}")
                continue
            
            all_params_str = match.group(1)
            first_param_raw = all_params_str.split(',')[0].strip()
            parameter = re.sub(r'[\\"\']', '', first_param_raw)
            
            if function_name == "move_to":
                move_to(parameter)
            elif function_name == "pickup":
                pickup(parameter)
            elif function_name == "drop":
                drop(parameter)
            else:
                print(f"Unknown command in plan: {step}")
                
        except Exception as e:
            print(f"Error executing step '{step}': {e}")
            
    print("--- PLAN COMPLETE ---")


def main():
    setup_simulation()
    
    print("\n" + "="*30)
    print("  LLM ROBOT CONTROLLER - READY")
    print("="*30)
    print(f"  Using model: llama3:8b")
    print("  Robot skills are now smarter!")
    print("="*30)
    print("Try these commands:")
    print(" - Get the red ball.")
    print(" - Bring me the blue cube.")
    print(" - Get the red ball and put it in the drop zone.")
    print(" - Type 'exit' to quit.")
    print("="*30)
    
    try:
        while True:
            command = input("\nEnter command for robot: ")
            
            if command.lower() == 'exit':
                break
            
            if command:
                plan = get_llm_plan(command)
                parse_and_execute(plan)

    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()
        print("Simulation disconnected. Goodbye.")

if __name__ == "__main__":
    main()