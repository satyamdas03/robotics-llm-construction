import pybullet as p
import pybullet_data
import time
import http.client
import json
import re  # We definitely need this!

# --- PART 1: THE "PLANNER" (OLLAMA LLM COMMUNICATION) ---
# This part is working correctly.

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
        
        # This regex finds all function calls, e.g., "move_to(...)"
        # This part is working perfectly.
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
# (This part is unchanged and correct)

object_ids = {
    "robot": None,
    "red_ball": None,
    "blue_cube": None,
    "drop_zone": None,
    "current_constraint": None
}

WORLD_KNOWLEDGE = {
    "red_ball": [2, 2, 0.05],
    "blue_cube": [2, -2, 0.05],
    "drop_zone": [-2, 0, 0.01],
    "start_area": [0, 0, 0.05]
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
    
    object_ids["robot"] = p.loadURDF("r2d2.urdf", WORLD_KNOWLEDGE["start_area"])
    
    ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
    ball_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
    object_ids["red_ball"] = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=ball_shape,
        baseVisualShapeIndex=ball_visual,
        basePosition=WORLD_KNOWLEDGE["red_ball"]
    )
    
    cube_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[0, 0, 1, 1])
    object_ids["blue_cube"] = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=cube_shape,
        baseVisualShapeIndex=cube_visual,
        basePosition=WORLD_KNOWLEDGE["blue_cube"]
    )

    zone_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01])
    zone_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01], rgbaColor=[0, 1, 0, 0.5])
    object_ids["drop_zone"] = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=zone_shape,
        baseVisualShapeIndex=zone_visual,
        basePosition=WORLD_KNOWLEDGE["drop_zone"]
    )
    print("Simulation setup complete.")

# --- ROBOT "SKILLS" ---

def move_to(target_name):
    print(f"SKILL: Moving to '{target_name}'...")
    if target_name not in WORLD_KNOWLEDGE:
        print(f"Error: Unknown target '{target_name}'")
        return
        
    target_pos = WORLD_KNOWLEDGE[target_name]
    p.resetBasePositionAndOrientation(
        object_ids["robot"],
        posObj=target_pos,
        ornObj=[0, 0, 0, 1]
    )
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)

def pickup(object_name):
    print(f"SKILL: Picking up '{object_name}'...")
    if object_name not in object_ids:
        print(f"Error: Unknown object '{object_name}'")
        return
    if object_ids["current_constraint"] is not None:
        print("Robot is already holding an object.")
        return

    object_id = object_ids[object_name]
    
    object_ids["current_constraint"] = p.createConstraint(
        parentBodyUniqueId=object_ids["robot"],
        parentLinkIndex=-1,
        childBodyUniqueId=object_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0.5],
        childFramePosition=[0, 0, 0]
    )
    print(f"Picked up {object_name}.")
    time.sleep(1)

def drop(location_name):
    print(f"SKILL: Dropping object at '{location_name}'...")
    if object_ids["current_constraint"] is None:
        print("Robot is not holding anything.")
        return
        
    p.removeConstraint(object_ids["current_constraint"])
    object_ids["current_constraint"] = None
    print("Dropped object.")
    time.sleep(1)

# --- PART 3: THE "EXECUTOR" (MAIN SCRIPT LOGIC) ---

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
            # Get the function name (e.G., "move_to")
            function_name = step.split('(', 1)[0].strip()
            
            # --- THIS IS THE FIX ---
            # 1. Find everything inside the parentheses
            match = re.search(r'\((.*)\)', step)
            if not match:
                print(f"Could not find parameters in step: {step}")
                continue
            
            # 2. Get all params, e.g., \"start_area\" OR \"drop_zone\", \"red_ball\"
            all_params_str = match.group(1)
            
            # 3. Take only the first parameter
            # e.g., \"start_area\"
            first_param_raw = all_params_str.split(',')[0].strip()
            
            # 4. Aggressively clean it:
            # This regex [\\"\'] matches a backslash, a double quote, OR a single quote.
            # We replace them all with an empty string.
            parameter = re.sub(r'[\\"\']', '', first_param_raw)
            
            # Now, 'parameter' will be clean, e.g., "start_area"
            
            # Call the correct function
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
    print("="*30)
    print("Try these commands:")
    print(" - Get the red ball.")
    print(" - Bring me the blue cube.")
    print(" - Get the red ball and put it in the drop zone.")
    print(" - Type 'exit' to quit.")
    print("="*30)
    
    try:
        while True:
            # Keep the simulation running so the window doesn't freeze
            p.stepSimulation()
            time.sleep(0.01) 
            
            command = input("\nEnter command for robot: ")
            
            if command.lower() == 'exit':
                break
            
            if command: # Only run if the command is not empty
                # 1. Get plan from LLM
                plan = get_llm_plan(command)
                
                # 2. Execute plan in PyBullet
                parse_and_execute(plan)

    except KeyboardInterrupt:
        pass
    finally:
        # Check if we are still connected before disconnecting
        if p.isConnected():
            p.disconnect()
        print("Simulation disconnected. Goodbye.")

if __name__ == "__main__":
    main()