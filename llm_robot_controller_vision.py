import pybullet as p
import pybullet_data
import time
import http.client
import json
import re  
import math 
import numpy as np

# --- PART 1: THE "PLANNER" (OLLAMA LLM COMMUNICATION) ---

def get_world_state_text():
    """
    Scans the PyBullet world and generates a text report of the
    current state for the LLM.
    """
    print("Generating world state report...")
    
    state_report = "\n--- CURRENT WORLD STATE ---\n"
    
    # 1. Get Robot's Location
    robot_id = object_ids["robot"]
    robot_pos = get_current_pos(robot_id)
    robot_location = "an unknown area"
    
    for name, info in WORLD_KNOWLEDGE.items():
        if info["type"] == "location":
            loc_id = info.get("id")
            if loc_id is not None:
                if get_distance(robot_id, loc_id) < 0.5:
                    robot_location = name
                    break
    state_report += f"Robot is at: {robot_location}\n"
    
    # 2. Check what robot is holding
    held_item = object_ids["held_object_name"]
    if held_item:
        state_report += f"Robot is holding: {held_item}\n"
    else:
        state_report += "Robot is holding: nothing\n"
        
    # 3. Get location of all *other* objects
    for name, info in WORLD_KNOWLEDGE.items():
        if info["type"] == "object" and name != held_item:
            obj_id = info.get("id")
            if obj_id is not None:
                obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
                state_report += f"- {name} is at: ({obj_pos[0]:.1f}, {obj_pos[1]:.1f})\n"
            
    state_report += "-----------------------------\n"
    return state_report

def get_llm_plan(user_command):
    """
    Sends a command (and world state) to the local LLM and gets a plan back.
    """
    print(f"Sending command to LLM: '{user_command}'...")
    
    world_state = get_world_state_text()
    
    # --- NEW: We are now explicitly asking for a JSON list of OBJECTS ---
    # --- AND teaching the AI about "original position" ---
    system_prompt = """
    You are a robot controller. You convert a Command into a JSON list of objects.
    You MUST obey the Current World STATE and all Rules.
    
    FUNCTIONS (Use this format):
    - {"function": "move_to", "target": "target_name"}
    - {"function": "pickup", "target": "object_name"}
    - {"function": "drop", "target": "location_name"}
    
    TARGETS:
    - "red_ball" (object)
    - "blue_cube" (object)
    - "start_area" (location)
    - "drop_zone" (location)
    - "red_ball_spawn" (location, the original position of the red ball)
    - "blue_cube_spawn" (location, the original position of the blue cube)

    RULES:
    1. Read the "CURRENT WORLD STATE" to know the robot's status.
    2. Your plan MUST logically follow the "Command". Do NOT add extra, un-commanded actions.
    3. If the robot is holding an object, it MUST `drop` it before it can `pickup` another.
    4. "Bring me" or "give me" an object means: `pickup` the object, `move_to("start_area")`, and `drop("start_area")`.
    5. "Return an object" or "put an object back" means: `pickup` the object, `move_to` its "spawn" location, and `drop` it there.
    6. You MUST respond with *only* the JSON list of objects. NO other text.
    7. Your response MUST start with [ and end with ].
    
    Example:
    --- CURRENT WORLD STATE ---
    Robot is at: drop_zone
    Robot is holding: red_ball
    - blue_cube is at: (2.0, -2.0)
    -----------------------------
    Command: "Return the red ball to its original position."
    Correct response: [{"function": "move_to", "target": "red_ball_spawn"}, {"function": "drop", "target": "red_ball_spawn"}]
    """
    
    conn = http.client.HTTPConnection("localhost", 11434)
    
    payload = {
        "model": "llama3:8b",
        "prompt": f"{system_prompt}\n{world_state}\nCommand: \"{user_command}\"\nResponse:",
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
        
        match = re.search(r'\[.*\]', llm_output_string, re.DOTALL)
        if not match:
            print(f"LLM Response did not contain a JSON list: {llm_output_string}")
            return []
        
        plan_json_string = match.group(0)
        plan = json.loads(plan_json_string)
        
        print(f"LLM generated plan: {plan}")
        return plan
        
    except Exception as e:
        print(f"Error communicating with Ollama or parsing JSON plan: {e}")
        print(f"Raw LLM Output was: {llm_output_string}")
        print("---")
        print("Is Ollama running? Have you run 'ollama pull llama3:8b'?")
        print("---")
        return []

# --- PART 2: THE "SIMULATOR & SKILLS" (PYBULLET ROBOTICS) ---

object_ids = {
    "robot": None,
    "current_constraint": None,
    "held_object_name": None 
}

# --- NEW: Added spawn points ---
WORLD_KNOWLEDGE = {
    "red_ball":        {"pos": [2, 2, 0.05],   "id": None, "type": "object"},
    "blue_cube":       {"pos": [2, -2, 0.05],  "id": None, "type": "object"},
    "drop_zone":       {"pos": [-2, 0, 0.01],  "id": None, "type": "location"},
    "start_area":      {"pos": [0, 0, 0.01],   "id": None, "type": "location"},
    "red_ball_spawn":  {"pos": [2, 2, 0.01],   "id": None, "type": "location"},
    "blue_cube_spawn": {"pos": [2, -2, 0.01],  "id": None, "type": "location"}
}

CAM_IMG_WIDTH = 320
CAM_IMG_HEIGHT = 200

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
    
    start_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01])
    start_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01], rgbaColor=[0.5, 0.5, 0.5, 0.5]) # Gray
    WORLD_KNOWLEDGE["start_area"]["id"] = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=start_shape,
        baseVisualShapeIndex=start_visual,
        basePosition=WORLD_KNOWLEDGE["start_area"]["pos"]
    )
    
    # --- NEW: Add visual markers for spawn points (faint purple) ---
    spawn_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.005])
    spawn_visual_red = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.005], rgbaColor=[1, 0, 1, 0.3])
    spawn_visual_blue = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.005], rgbaColor=[0, 1, 1, 0.3])

    WORLD_KNOWLEDGE["red_ball_spawn"]["id"] = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=spawn_shape,
        baseVisualShapeIndex=spawn_visual_red,
        basePosition=WORLD_KNOWLEDGE["red_ball_spawn"]["pos"]
    )
    WORLD_KNOWLEDGE["blue_cube_spawn"]["id"] = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=spawn_shape,
        baseVisualShapeIndex=spawn_visual_blue,
        basePosition=WORLD_KNOWLEDGE["blue_cube_spawn"]["pos"]
    )
    
    print("Simulation setup complete.")

# --- ROBOT "SKILLS" ---

def get_current_pos(body_id):
    pos, _ = p.getBasePositionAndOrientation(body_id)
    return pos[0], pos[1]

def get_distance(objA_id, objB_id):
    posA = get_current_pos(objA_id)
    posB = get_current_pos(objB_id)
    return math.sqrt((posA[0] - posB[0])**2 + (posA[1] - posB[1])**2)

def get_camera_image_and_find(target_id):
    robot_id = object_ids["robot"]
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    
    rot_matrix = p.getMatrixFromQuaternion(orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    
    camera_offset = [0.2, 0, 0.1] 
    camera_pos = pos + rot_matrix.dot(camera_offset) 
    target_offset = [1.0, 0, 0] 
    target_pos = pos + rot_matrix.dot(target_offset)

    view_matrix = p.computeViewMatrix(camera_pos, target_pos, [0, 0, 1])
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=float(CAM_IMG_WIDTH) / CAM_IMG_HEIGHT,
        nearVal=0.1,
        farVal=100.0
    )
    
    (_, _, _, _, segImg) = p.getCameraImage(
        width=CAM_IMG_WIDTH,
        height=CAM_IMG_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL 
    )
    
    if target_id in segImg:
        return True, segImg 
    else:
        return False, segImg 

def find_object(target_name):
    print(f"SKILL: Finding '{target_name}'...")
    
    if object_ids["held_object_name"] == target_name:
        print(f"Already holding {target_name}.")
        return True
    
    target_id = WORLD_KNOWLEDGE[target_name]["id"]
    robot_id = object_ids["robot"]
    
    distance = get_distance(robot_id, target_id)
    if distance < 0.5:
        print(f"Object {target_name} is already at the robot's feet.")
        return True
    
    for i in range(20):
        found, _ = get_camera_image_and_find(target_id)
        if found:
            print(f"Found {target_name}!")
            return True
            
        current_pos, current_orn = p.getBasePositionAndOrientation(robot_id)
        yaw = p.getEulerFromQuaternion(current_orn)[2]
        next_yaw = yaw + (math.pi * 2 / 20)
        next_orn = p.getQuaternionFromEuler([0, 0, next_yaw])
        
        p.resetBasePositionAndOrientation(robot_id, current_pos, next_orn)
        
        for _ in range(10):
            p.stepSimulation()
            time.sleep(0.01)
            
    print(f"Could not find {target_name} after spinning 360 degrees.")
    return False

def _navigate_to_coords(target_pos):
    """
    Internal helper function to navigate to a known (x, y) coordinate.
    """
    robot_id = object_ids["robot"]
    
    # --- NEW: Proportional Speed Control ---
    # We will slow down as we get closer to prevent overshoot.
    close_enough = 0.3 # How close we need to be to stop
    
    for _ in range(1000):
        current_pos, _ = p.getBasePositionAndOrientation(robot_id)
        
        dir_x = target_pos[0] - current_pos[0]
        dir_y = target_pos[1] - current_pos[1]
        distance = math.sqrt(dir_x**2 + dir_y**2)
        
        if distance < close_enough:
            p.resetBaseVelocity(robot_id, [0, 0, 0]) 
            return True 
        
        # --- P-Controller for Speed ---
        # Speed is proportional to distance, with a max of 2.0 and a min of 0.4
        # This prevents jittering and helps it "settle" at the target.
        speed = min(2.0, max(0.4, distance * 1.5))
        # --- End P-Controller ---
        
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
    return False

def move_to(target_name):
    print(f"SKILL: Navigating to '{target_name}'...")
    if target_name not in WORLD_KNOWLEDGE:
        print(f"Error: Unknown target '{target_name}'")
        return
    
    target_info = WORLD_KNOWLEDGE[target_name]
    
    if target_info["type"] == "location":
        print(f"{target_name} is a location. Moving to known coordinates.")
        arrived = _navigate_to_coords(target_info["pos"])
        if arrived:
            print(f"Arrived at {target_name}.")
            
    elif target_info["type"] == "object":
        print(f"{target_name} is an object. Trying to find it...")
        found = find_object(target_name)
        
        if found:
            if object_ids["held_object_name"] == target_name:
                print(f"Already holding {target_name}, no need to move.")
                return
                
            target_id = target_info["id"]
            target_pos_tuple, _ = p.getBasePositionAndOrientation(target_id)
            print(f"Object {target_name} located. Moving to its position.")
            arrived = _navigate_to_coords(target_pos_tuple)
            if arrived:
                print(f"Arrived at {target_name}.")
        else:
            print(f"Failed to move to {target_name} (could not find it).")

def pickup(object_name):
    print(f"SKILL: Attempting to pick up '{object_name}'...")
    if object_name not in WORLD_KNOWLEDGE:
        print(f"Error: Unknown object '{object_name}'")
        return
    # --- NEW: Check type ---
    if WORLD_KNOWLEDGE[object_name]["type"] != "object":
        print(f"Error: Cannot 'pickup' a location: {object_name}")
        return
        
    if object_ids["current_constraint"] is not None:
        print(f"Robot is already holding an object ({object_ids['held_object_name']}).")
        return

    robot_id = object_ids["robot"]
    object_id = WORLD_KNOWLEDGE[object_name]["id"]
    
    distance = get_distance(robot_id, object_id)
    if distance > 0.5:
        print(f"Error: '{object_name}' is too far away ({distance:.2f} units). Moving to it first...")
        move_to(object_name)
    
    # Re-check distance after moving
    distance = get_distance(robot_id, object_id)
    if distance > 0.5:
        print(f"Failed to get close enough to {object_name}. Pickup failed.")
        return

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
    object_ids["held_object_name"] = object_name 
    print(f"Picked up {object_name}.")
    
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)

def drop(location_name):
    print(f"SKILL: Attempting to drop at '{location_name}'...")
    if location_name not in WORLD_KNOWLEDGE:
        print(f"Error: Unknown location '{location_name}'")
        return
    # --- NEW: Check type ---
    target_info = WORLD_KNOWLEDGE[location_name]
    if target_info["type"] != "location":
        print(f"Error: Cannot 'drop' at an object. '{location_name}' is not a location.")
        return
        
    if object_ids["current_constraint"] is None:
        print("Robot is not holding anything.")
        return
    
    robot_id = object_ids["robot"]
    drop_zone_id = target_info["id"] # Use the info we already fetched

    distance = get_distance(robot_id, drop_zone_id)
    if distance > 0.5:
        print(f"Error: Not at '{location_name}' ({distance:.2f} units). Moving to it first...")
        move_to(location_name)
    
    distance = get_distance(robot_id, drop_zone_id)
    if distance > 0.5:
        print(f"Failed to get close enough to {location_name}. Drop failed.")
        return

    p.removeConstraint(object_ids["current_constraint"])
    object_ids["current_constraint"] = None
    object_ids["held_object_name"] = None 
    print("Dropped object.")
    
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)

# --- PART 3: THE "EXECUTOR" (MAIN SCRIPT LOGIC) ---
# (This part is unchanged and working)

def parse_and_execute(plan):
    """
    Parses the new plan (a list of dicts) and calls the
    corresponding Python function.
    """
    if not plan:
        print("Plan is empty. Nothing to do.")
        return
        
    print(f"--- EXECUTING PLAN ---")
    
    for step in plan:
        try:
            if not isinstance(step, dict):
                print(f"Error: Plan step is not a valid object: {step}")
                continue
                
            function_name = step.get("function")
            parameter = step.get("target")
            
            if not function_name or not parameter:
                print(f"Error: Plan step is malformed: {step}")
                continue
            
            if function_name == "move_to":
                move_to(parameter)
            elif function_name == "pickup":
                pickup(parameter)
            elif function_name == "drop":
                drop(parameter)
            else:
                print(f"Unknown command in plan: {function_name}")
                
        except Exception as e:
            print(f"Error executing step '{step}': {e}")
            
    print("--- PLAN COMPLETE ---")


def main():
    setup_simulation()
    
    print("\n" + "="*30)
    print("  LLM ROBOT CONTROLLER - READY")
    print("="*30)
    print(f"  Using model: llama3:8b")
    print("  Robot has MEMORY, CAMERA, and P-CONTROL!")
    print("="*30)
    print("Try these commands:")
    print(" - Get the red ball.")
    print(" - Bring me the blue cube. (It should drop the red ball first!)")
    print(" - Get the red ball and put it in the drop zone.")
    print(" - Return the red ball to its original position.")
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