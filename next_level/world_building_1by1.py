import pybullet as p
import pybullet_data
import time
import http.client
import json
import re  
import math 
import numpy as np
import cv2 # Import OpenCV

# --- PART 1: THE "PLANNER" (OLLAMA LLM COMMUNICATION) ---

def get_location_name(body_id):
    """
    Finds the human-readable location name of a body.
    """
    for name, info in WORLD_KNOWLEDGE.items():
        if info["type"] == "location":
            loc_id = info.get("id")
            if loc_id is not None:
                if get_distance(body_id, loc_id) < 0.5:
                    return name
    return "an unknown area"

def get_world_state_text(robot_name):
    """
    Scans the PyBullet world and generates a text report of the
    current state for a SPECIFIC robot.
    """
    print(f"Generating world state report for {robot_name}...")
    
    state_report = "\n--- CURRENT WORLD STATE ---\n"
    
    # 1. Get Robot's Location
    robot_id = ROBOT_STATE[robot_name]["id"]
    state_report += f"Robot is at: {get_location_name(robot_id)}\n"
    
    # 2. Check what robot is holding
    held_item = ROBOT_STATE[robot_name]["held_object_name"]
    if held_item:
        state_report += f"Robot is holding: {held_item}\n"
    else:
        state_report += "Robot is holding: nothing\n"
        
    state_report += "Object locations are UNKNOWN unless seen.\n"
    state_report += "-----------------------------\n"
    return state_report

def get_llm_plan(robot_name, user_command, failure_info=None):
    """
    Sends a command to the LLM for a specific robot.
    """
    world_state = get_world_state_text(robot_name) 
    
    system_prompt = """
    You are a robot controller. You convert a Command into a JSON list of objects.
    You MUST obey the Current World STATE and all Rules.
    
    FUNCTIONS (Use this format):
    - {"function": "move_to", "target": "target_name"}
    - {"function": "pickup", "target": "object_name"}
    - {"function": "drop", "target": "location_name"}
    - {"function": "return_object", "target": "object_name"}
    
    TARGETS:
    - "red_ball" (object)
    - "blue_cube" (object)
    - "yellow_cylinder" (object)
    - "start_area" (location)
    - "drop_zone" (location)

    RULES:
    1. Read the "CURRENT WORLD STATE".
    2. Object locations are UNKNOWN. To find an object, you MUST use `move_to("object_name")`.
    3. If the robot is holding an object, it MUST `drop` it before it can `pickup` another.
    4. "Bring me" or "give me" an object means: `pickup` the object, `move_to("start_area")`, and `drop("start_area")`.
    5. "Return an object" or "put an object back" means: call `return_object("object_name")`.
    6. You MUST respond with *only* the JSON list of objects. NO other text.
    """
    
    prompt_text = f"{system_prompt}\n{world_state}\n"
    
    if failure_info:
        print(f"\n!!! [{robot_name}] REPLANNING REQUIRED: {failure_info} !!!\n")
        prompt_text += f"PREVIOUS PLAN FAILED!\nFailure Reason: {failure_info}\n"
        prompt_text += f"The ORIGINAL COMMAND was: \"{user_command}\"\n"
        prompt_text += "Generate a NEW, FULL plan from scratch to achieve the original command.\nResponse:"
    else:
        print(f"[{robot_name}] Sending command to LLM: '{user_command}'...")
        prompt_text += f"Command: \"{user_command}\"\nResponse:"

    conn = http.client.HTTPConnection("localhost", 11434)
    
    payload = {
        "model": "llama3:8b",
        "prompt": prompt_text,
        "stream": False
    }
    
    llm_output_string = "" 
    
    try:
        conn.request("POST", "/api/generate", json.dumps(payload))
        response = conn.getresponse()
        
        if response.status != 200:
            print(f"[{robot_name}] Error from Ollama: {response.status} {response.reason}")
            return []
            
        response_body = response.read().decode('utf-8')
        conn.close()
        
        response_data = json.loads(response_body)
        llm_output_string = response_data.get('response', '[]')
        
        match = re.search(r'\[.*\]', llm_output_string, re.DOTALL)
        if not match:
            print(f"[{robot_name}] LLM Response did not contain a JSON list: {llm_output_string}")
            return []
        
        plan_json_string = match.group(0)
        plan = json.loads(plan_json_string)
        
        print(f"[{robot_name}] LLM generated plan: {plan}")
        return plan
        
    except Exception as e:
        print(f"[{robot_name}] Error communicating with Ollama or parsing JSON plan: {e}")
        return []

# --- PART 2: THE "SIMULATOR & SKILLS" (PYBULLET ROBOTICS) ---

ROBOT_STATE = {
    "robot_0": {"id": None, "held_object_name": None, "current_constraint": None, "color": [1, 0, 0, 1]}, # Red
    "robot_1": {"id": None, "held_object_name": None, "current_constraint": None, "color": [0, 1, 0, 1]}, # Green
    "robot_2": {"id": None, "held_object_name": None, "current_constraint": None, "color": [0, 0, 1, 1]}  # Blue
}

WORLD_KNOWLEDGE = {
    "red_ball":        {"pos": [2, 2, 0.05],   "id": None, "type": "object"},
    "blue_cube":       {"pos": [2, -2, 0.05],  "id": None, "type": "object"},
    "yellow_cylinder": {"pos": [0, 3, 0.05],   "id": None, "type": "object"}, 
    "drop_zone":       {"pos": [-2, 0, 0.01],  "id": None, "type": "location"},
    "start_area":      {"pos": [0, 0, 0.01],   "id": None, "type": "location"},
    "red_ball_spawn":  {"pos": [2, 2, 0.01],   "id": None, "type": "location"},
    "blue_cube_spawn": {"pos": [2, -2, 0.01],  "id": None, "type": "location"},
    "yellow_cylinder_spawn": {"pos": [0, 3, 0.01],   "id": None, "type": "location"} 
}

CAM_IMG_WIDTH = 320
CAM_IMG_HEIGHT = 200

# Color Ranges
COLOR_RANGES = {
    "red_ball": {
        'lower1': np.array([0, 100, 70]), 'upper1': np.array([10, 255, 255]),
        'lower2': np.array([170, 100, 70]), 'upper2': np.array([180, 255, 255])
    },
    "blue_cube": {
        'lower1': np.array([100, 100, 70]), 'upper1': np.array([140, 255, 255]),
        'lower2': None, 'upper2': None
    },
    "yellow_cylinder": {
        'lower1': np.array([20, 100, 70]), 'upper1': np.array([35, 255, 255]),
        'lower2': None, 'upper2': None
    }
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
    
    # Spread the robots out
    start_positions = [[-1, 0, 0.05], [0, 0, 0.05], [1, 0, 0.05]]
    
    for i, robot_name in enumerate(ROBOT_STATE.keys()):
        start_pos = start_positions[i]
        robot_id = p.loadURDF("r2d2.urdf", start_pos)
        ROBOT_STATE[robot_name]["id"] = robot_id
        p.changeVisualShape(robot_id, -1, rgbaColor=ROBOT_STATE[robot_name]["color"])

    # Setup Objects
    ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
    ball_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
    WORLD_KNOWLEDGE["red_ball"]["id"] = p.createMultiBody(
        baseMass=0.1, baseCollisionShapeIndex=ball_shape,
        baseVisualShapeIndex=ball_visual, basePosition=WORLD_KNOWLEDGE["red_ball"]["pos"]
    )
    
    cube_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[0, 0, 1, 1])
    WORLD_KNOWLEDGE["blue_cube"]["id"] = p.createMultiBody(
        baseMass=0.1, baseCollisionShapeIndex=cube_shape,
        baseVisualShapeIndex=cube_visual, basePosition=WORLD_KNOWLEDGE["blue_cube"]["pos"]
    )
    
    cyl_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.2)
    cyl_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.1, length=0.2, rgbaColor=[1, 1, 0, 1])
    WORLD_KNOWLEDGE["yellow_cylinder"]["id"] = p.createMultiBody(
        baseMass=0.1, baseCollisionShapeIndex=cyl_shape,
        baseVisualShapeIndex=cyl_visual, basePosition=WORLD_KNOWLEDGE["yellow_cylinder"]["pos"]
    )

    # Setup Locations
    zone_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01])
    zone_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01], rgbaColor=[0, 1, 0, 0.5])
    WORLD_KNOWLEDGE["drop_zone"]["id"] = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=zone_shape,
        baseVisualShapeIndex=zone_visual, basePosition=WORLD_KNOWLEDGE["drop_zone"]["pos"]
    )
    
    start_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01])
    start_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01], rgbaColor=[0.5, 0.5, 0.5, 0.5])
    WORLD_KNOWLEDGE["start_area"]["id"] = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=start_shape,
        baseVisualShapeIndex=start_visual, basePosition=WORLD_KNOWLEDGE["start_area"]["pos"]
    )
    
    # Spawn markers
    spawn_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.005])
    spawn_visual_red = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.005], rgbaColor=[1, 0, 1, 0.3])
    spawn_visual_blue = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.005], rgbaColor=[0, 1, 1, 0.3])
    spawn_visual_yellow = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.005], rgbaColor=[1, 1, 0, 0.3])

    WORLD_KNOWLEDGE["red_ball_spawn"]["id"] = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=spawn_shape,
        baseVisualShapeIndex=spawn_visual_red, basePosition=WORLD_KNOWLEDGE["red_ball_spawn"]["pos"]
    )
    WORLD_KNOWLEDGE["blue_cube_spawn"]["id"] = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=spawn_shape,
        baseVisualShapeIndex=spawn_visual_blue, basePosition=WORLD_KNOWLEDGE["blue_cube_spawn"]["pos"]
    )
    WORLD_KNOWLEDGE["yellow_cylinder_spawn"]["id"] = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=spawn_shape,
        baseVisualShapeIndex=spawn_visual_yellow, basePosition=WORLD_KNOWLEDGE["yellow_cylinder_spawn"]["pos"]
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

def get_camera_image_and_find(robot_name, target_name):
    """
    Takes a snapshot from the specified robot's camera and uses OpenCV
    to find the target object by its color.
    """
    if target_name not in COLOR_RANGES:
        print(f"[{robot_name}] Error: No color definition for {target_name}")
        return False, None

    robot_id = ROBOT_STATE[robot_name]["id"] 
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    
    rot_matrix = p.getMatrixFromQuaternion(orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    
    # Aggressive Camera Angle
    camera_offset = [0.2, 0, 0.3] 
    target_offset = [1.5, 0, 0.0] 
    
    camera_pos = pos + rot_matrix.dot(camera_offset) 
    target_pos = pos + rot_matrix.dot(target_offset)

    view_matrix = p.computeViewMatrix(camera_pos, target_pos, [0, 0, 1])
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=90.0,
        aspect=float(CAM_IMG_WIDTH) / CAM_IMG_HEIGHT,
        nearVal=0.1,
        farVal=100.0
    )
    
    # Get the RGB image
    (_, _, rgbImg, _, _) = p.getCameraImage(
        width=CAM_IMG_WIDTH,
        height=CAM_IMG_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL 
    )
    
    # --- FIX 3: FORCE RESHAPE AND TYPE ---
    # PyBullet returns a flat list. We must reshape it to (H, W, 4).
    height = CAM_IMG_HEIGHT
    width = CAM_IMG_WIDTH
    
    rgbImg = np.array(rgbImg, dtype=np.uint8)
    
    # Check if we need to reshape (common PyBullet behavior)
    if len(rgbImg.shape) == 1:
        rgbImg = rgbImg.reshape((height, width, 4))
        
    # Now it is guaranteed to be (H, W, 4) and uint8
    
    # --- OpenCV Processing ---
    try:
        # RGBA to BGR (OpenCV default)
        img_bgr = cv2.cvtColor(rgbImg, cv2.COLOR_RGBA2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        colors = COLOR_RANGES[target_name]
        
        mask1 = cv2.inRange(img_hsv, colors['lower1'], colors['upper1'])
        
        if colors['lower2'] is not None:
            mask2 = cv2.inRange(img_hsv, colors['lower2'], colors['upper2'])
            mask = mask1 | mask2
        else:
            mask = mask1
            
        if np.sum(mask) > 50: 
            return True, mask 
        else:
            return False, mask
    except Exception as e:
        print(f"[{robot_name}] Vision Error: {e}")
        print(f"Debug: Img Shape: {rgbImg.shape}, Type: {rgbImg.dtype}")
        return False, None

def find_object(robot_name, target_name):
    print(f"[{robot_name}] SKILL: Finding '{target_name}'...")
    
    if ROBOT_STATE[robot_name]["held_object_name"] == target_name:
        print(f"[{robot_name}] Already holding {target_name}.")
        return True
    
    target_id = WORLD_KNOWLEDGE[target_name]["id"]
    robot_id = ROBOT_STATE[robot_name]["id"]
    
    distance = get_distance(robot_id, target_id)
    if distance < 0.5:
        print(f"[{robot_name}] Object {target_name} is already at my feet.")
        return True
    
    for i in range(20):
        found, _ = get_camera_image_and_find(robot_name, target_name) 
        if found:
            print(f"[{robot_name}] Found {target_name}!")
            return True
            
        current_pos, current_orn = p.getBasePositionAndOrientation(robot_id)
        yaw = p.getEulerFromQuaternion(current_orn)[2]
        next_yaw = yaw + (math.pi * 2 / 20)
        next_orn = p.getQuaternionFromEuler([0, 0, next_yaw])
        
        p.resetBasePositionAndOrientation(robot_id, current_pos, next_orn)
        
        for _ in range(10):
            p.stepSimulation()
            time.sleep(0.01)
            
    print(f"[{robot_name}] Could not find {target_name} after spinning 360 degrees.")
    return False

def _navigate_to_coords(robot_name, target_pos):
    robot_id = ROBOT_STATE[robot_name]["id"]
    
    # "Jitter" Bug Fix
    close_enough = 0.4 
    
    for _ in range(1500): 
        current_pos, _ = p.getBasePositionAndOrientation(robot_id)
        
        dir_x = target_pos[0] - current_pos[0]
        dir_y = target_pos[1] - current_pos[1]
        distance = math.sqrt(dir_x**2 + dir_y**2)
        
        if distance < close_enough:
            p.resetBaseVelocity(robot_id, [0, 0, 0]) 
            return True 
        
        speed = min(2.0, max(0.4, distance * 1.5))
        
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
    print(f"[{robot_name}] Navigation timed out.")
    return False

def move_to(robot_name, target_name):
    print(f"[{robot_name}] SKILL: Navigating to '{target_name}'...")
    if target_name not in WORLD_KNOWLEDGE:
        print(f"[{robot_name}] Error: Unknown target '{target_name}'")
        return False
    
    target_info = WORLD_KNOWLEDGE[target_name]
    
    if target_info["type"] == "location":
        print(f"[{robot_name}] {target_name} is a location. Moving to known coordinates.")
        if _navigate_to_coords(robot_name, target_info["pos"]):
            print(f"[{robot_name}] Arrived at {target_name}.")
            return True
        return False
            
    elif target_info["type"] == "object":
        print(f"[{robot_name}] {target_name} is an object. Trying to find it...")
        if find_object(robot_name, target_name):
            if ROBOT_STATE[robot_name]["held_object_name"] == target_name:
                print(f"[{robot_name}] Already holding {target_name}, no need to move.")
                return True
                
            target_id = target_info["id"]
            target_pos_tuple, _ = p.getBasePositionAndOrientation(target_id)
            print(f"[{robot_name}] Object {target_name} located. Moving to its position.")
            if _navigate_to_coords(robot_name, target_pos_tuple):
                print(f"[{robot_name}] Arrived at {target_name}.")
                return True
        else:
            print(f"[{robot_name}] Failed to move to {target_name} (could not find it).")
            return False
    return False

def pickup(robot_name, object_name):
    print(f"[{robot_name}] SKILL: Attempting to pick up '{object_name}'...")
    if object_name not in WORLD_KNOWLEDGE:
        print(f"[{robot_name}] Error: Unknown object '{object_name}'")
        return False
    if WORLD_KNOWLEDGE[object_name]["type"] != "object":
        print(f"[{robot_name}] Error: Cannot 'pickup' a location: {object_name}")
        return False
        
    if ROBOT_STATE[robot_name]["current_constraint"] is not None:
        print(f"[{robot_name}] Robot is already holding an object ({ROBOT_STATE[robot_name]['held_object_name']}).")
        return False

    robot_id = ROBOT_STATE[robot_name]["id"]
    object_id = WORLD_KNOWLEDGE[object_name]["id"]
    
    pickup_distance = 0.45 # Jitter Fix
    
    distance = get_distance(robot_id, object_id)
    if distance > pickup_distance:
        print(f"[{robot_name}] Error: '{object_name}' is too far away ({distance:.2f} units). Moving to it first...")
        if not move_to(robot_name, object_name):
            return False
    
    distance = get_distance(robot_id, object_id)
    if distance > pickup_distance:
        print(f"[{robot_name}] Failed to get close enough to {object_name} (distance: {distance:.2f}). Pickup failed.")
        return False

    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=-1, 
        childBodyUniqueId=object_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0.5],
        childFramePosition=[0, 0, 0]
    )
    ROBOT_STATE[robot_name]["current_constraint"] = constraint_id
    ROBOT_STATE[robot_name]["held_object_name"] = object_name 
    print(f"[{robot_name}] Picked up {object_name}.")
    
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)
    return True

def drop(robot_name, location_name):
    print(f"[{robot_name}] SKILL: Attempting to drop at '{location_name}'...")
    if location_name not in WORLD_KNOWLEDGE:
        print(f"[{robot_name}] Error: Unknown location '{location_name}'")
        return False
        
    target_info = WORLD_KNOWLEDGE[location_name]
    if target_info["type"] != "location":
        print(f"[{robot_name}] Error: Cannot 'drop' at an object. '{location_name}' is not a location.")
        return False
        
    if ROBOT_STATE[robot_name]["current_constraint"] is None:
        print(f"[{robot_name}] Robot is not holding anything.")
        return False
    
    robot_id = ROBOT_STATE[robot_name]["id"]
    drop_zone_id = target_info["id"] 

    drop_distance = 0.45 # Jitter Fix

    distance = get_distance(robot_id, drop_zone_id)
    if distance > drop_distance:
        print(f"[{robot_name}] Error: Not at '{location_name}' ({distance:.2f} units). Moving to it first...")
        if not move_to(robot_name, location_name):
            return False
    
    distance = get_distance(robot_id, drop_zone_id)
    if distance > drop_distance:
        print(f"[{robot_name}] Failed to get close enough to {location_name}. Drop failed.")
        return False

    p.removeConstraint(ROBOT_STATE[robot_name]["current_constraint"])
    ROBOT_STATE[robot_name]["current_constraint"] = None
    ROBOT_STATE[robot_name]["held_object_name"] = None 
    print(f"[{robot_name}] Dropped object.")
    
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)
    return True

def return_object(robot_name, object_name):
    """
    Smarter skill to return an object to its spawn point.
    """
    print(f"[{robot_name}] SKILL: Attempting to return '{object_name}'...")
    if object_name not in WORLD_KNOWLEDGE or WORLD_KNOWLEDGE[object_name]["type"] != "object":
        print(f"[{robot_name}] Error: Unknown object {object_name}")
        return False

    if ROBOT_STATE[robot_name]["held_object_name"] != object_name:
        print(f"[{robot_name}] Error: Not holding {object_name}. Attempting to pick it up first.")
        if not pickup(robot_name, object_name):
            print(f"[{robot_name}] Failed to pick up {object_name}.")
            return False
            
    spawn_name = f"{object_name}_spawn"
    if spawn_name not in WORLD_KNOWLEDGE:
        print(f"[{robot_name}] Error: No spawn location defined for {object_name}.")
        return False
        
    print(f"[{robot_name}] Moving to {spawn_name} to drop {object_name}.")
    if move_to(robot_name, spawn_name):
        return drop(robot_name, spawn_name)
    
    return False

# --- PART 3: THE "EXECUTOR" (MAIN SCRIPT LOGIC) ---

def parse_and_execute(robot_name, plan):
    if not plan:
        print(f"[{robot_name}] Plan is empty. Nothing to do.")
        return True, None
        
    print(f"--- [{robot_name}] EXECUTING PLAN ---")
    
    for step in plan:
        try:
            if not isinstance(step, dict):
                print(f"[{robot_name}] Error: Plan step is not a valid object: {step}")
                continue
                
            function_name = step.get("function")
            parameter = step.get("target")

            if not function_name or not parameter:
                print(f"[{robot_name}] Error: Plan step is malformed: {step}")
                continue
            
            success = False
            if function_name == "move_to":
                success = move_to(robot_name, parameter)
            elif function_name == "pickup":
                success = pickup(robot_name, parameter)
            elif function_name == "drop":
                success = drop(robot_name, parameter)
            elif function_name == "return_object": 
                success = return_object(robot_name, parameter)
            else:
                print(f"[{robot_name}] Unknown command in plan: {function_name}")
                success = True 
            
            if not success:
                print(f"!!! [{robot_name}] STEP FAILED: {function_name}({parameter}) !!!")
                return False, step 
                
        except Exception as e:
            print(f"[{robot_name}] Error executing step '{step}': {e}")
            return False, step
            
    print(f"--- [{robot_name}] PLAN COMPLETE ---")
    return True, None


def main():
    setup_simulation()
    
    print("\n" + "="*30)
    print("  MULTI-ROBOT CONTROLLER - READY")
    print("="*30)
    print(f"  Using model: llama3:8b")
    print("  Mode: Sequential Foreman")
    print("  VISION: Now using OpenCV!")
    print("="*30)

    # Initial commands for each robot
    commands = {
        "robot_0": "Bring me the red ball",
        "robot_1": "Bring me the blue cube",
        "robot_2": "Bring me the yellow cylinder"
    }
    
    # Keep track of failures to enable replanning
    failures = {
        "robot_0": None,
        "robot_1": None,
        "robot_2": None
    }

    # Restart Ollama check
    print("Ensure Ollama is running with 'ollama run llama3:8b' before starting.")
    input("Press Enter to start the simulation loop...")

    while True:
        for robot_name in commands.keys():
            # Skip if command is None (finished)
            if commands[robot_name] is None:
                continue
                
            # 1. Get Plan (with failure info if needed)
            plan = get_llm_plan(robot_name, commands[robot_name], failures[robot_name])
            
            # 2. Execute Plan
            success, error_message = parse_and_execute(robot_name, plan)
            
            if success:
                print(f"[{robot_name}] Task complete! Waiting for new command...")
                # Mark as done for this demo
                commands[robot_name] = None 
                failures[robot_name] = None
            else:
                print(f"[{robot_name}] Task failed. Will replan next cycle.")
                failures[robot_name] = error_message
                
        p.stepSimulation()
        time.sleep(0.01)

if __name__ == "__main__":
    main()