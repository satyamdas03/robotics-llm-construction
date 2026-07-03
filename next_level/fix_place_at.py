# Script to revert to the known-good version and apply the fix correctly

# Copy the working version from before our botched edit
import shutil

# Make a safety backup first
shutil.copy('world_building_construction.py', 'world_building_construction_broken.py')

# Restore from our earlier restoration
with open('world_building_construction.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the place_at function with corrected version
place_at_start = content.find('def place_at(robot_name, target_coords_str):')
if place_at_start == -1:
    print("ERROR: Could not find place_at function!")
    import sys
    sys.exit(1)

# Find the end of place_at (start of return_object)
return_object_start = content.find('def return_object(robot_name, object_name):', place_at_start)
if return_object_start == -1:
    print("ERROR: Could not find return_object function!")
    import sys
    sys.exit(1)

# Extract everything before place_at and after place_at
before_place_at = content[:place_at_start]
after_place_at = content[return_object_start:]

# New place_at function with robot stabilization
new_place_at = '''def place_at(robot_name, target_coords_str):
    print(f"[{robot_name}] SKILL: Attempting to place at '{target_coords_str}'...")
    
    try:
        coords = [float(x.strip()) for x in target_coords_str.split(',')]
        if len(coords) != 3:
            print(f"[{robot_name}] Error: Invalid coordinates '{target_coords_str}'")
            return False
    except ValueError:
        print(f"[{robot_name}] Error: Could not parse coordinates '{target_coords_str}'")
        return False

    if ROBOT_STATE[robot_name]["current_constraint"] is None:
        print(f"[{robot_name}] Error: Cannot place. Robot is not holding anything.")
        return False
        
    held_obj_name = ROBOT_STATE[robot_name]["held_object_name"]
    if not held_obj_name:
        print(f"[{robot_name}] Error: State says holding nothing (name is None).")
        return False
        
    object_id = WORLD_KNOWLEDGE[held_obj_name]["id"]
    robot_id = ROBOT_STATE[robot_name]["id"]

    # --- OFFSET PLACEMENT LOGIC ---
    # 1. Calculate Standoff Position (0.5m away from target)
    current_pos, _ = p.getBasePositionAndOrientation(robot_id)
    dx = coords[0] - current_pos[0]
    dy = coords[1] - current_pos[1]
    angle_to_target = math.atan2(dy, dx)
    
    # Target position for the ROBOT BODY (0.5m away from block target)
    standoff_dist = 0.5
    stand_x = coords[0] - math.cos(angle_to_target) * standoff_dist
    stand_y = coords[1] - math.sin(angle_to_target) * standoff_dist
    stand_pos = [stand_x, stand_y, 0]

    print(f"[{robot_name}] Moving to Standoff Position {stand_pos} (Offset Placement)...")
    
    if _navigate_to_coords(robot_name, stand_pos, threshold=0.05):
        print(f"[{robot_name}] Arrived at Standoff. Aligning to face target...")
        
        # 2. Force Orientation to face target exactly
        target_orn = p.getQuaternionFromEuler([0, 0, angle_to_target])
        p.resetBasePositionAndOrientation(robot_id, [stand_x, stand_y, 0], target_orn)
        
        # Stop any residual motion
        p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
        for _ in range(20): p.stepSimulation(); time.sleep(0.01)

        print(f"[{robot_name}] Aligned. Freezing robot for stable placement...")
        
        # FIX: Freeze robot to prevent tipping
        original_dynamics = p.getDynamicsInfo(robot_id, -1)
        original_mass = original_dynamics[0]
        p.changeDynamics(robot_id, -1, mass=0)  # Make immovable
        p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
        
        print(f"[{robot_name}] Robot stabilized. Lowering object gently...")
        
        constraint_id = ROBOT_STATE[robot_name]["current_constraint"]
        
        start_height = 0.6
        target_relative_height = coords[2] + 0.02
        
        steps = 100 
        for i in range(steps):
            h = start_height - (start_height - target_relative_height) * (i / steps)
            
            p.removeConstraint(constraint_id)
            constraint_id = p.createConstraint(
                parentBodyUniqueId=robot_id,
                parentLinkIndex=-1, 
                childBodyUniqueId=object_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0.5, 0, h],
                childFramePosition=[0, 0, 0]
            )
            ROBOT_STATE[robot_name]["current_constraint"] = constraint_id
            
            # FIX: Keep robot locked during lowering
            p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
            
            p.stepSimulation()
            time.sleep(0.01)
        
        # FIX: Teleport block to exact target position
        print(f"[{robot_name}] Correcting block position to exact target...")
        level_orn = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(object_id, coords, level_orn)
        
        # Stabilize with correct position
        for _ in range(30): 
            p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
            p.stepSimulation()
            time.sleep(0.01)

        # Release constraint
        p.removeConstraint(constraint_id)
        ROBOT_STATE[robot_name]["current_constraint"] = None
        ROBOT_STATE[robot_name]["held_object_name"] = None 
        print(f"[{robot_name}] Placed object at {coords}.")
        
        # FIX: Restore robot mobility for retreat
        p.changeDynamics(robot_id, -1, mass=original_mass)
        
        # BACKUP MANEUVER
        print(f"[{robot_name}] Backing up...")
        for _ in range(40):
            bx = -math.cos(angle_to_target) * 0.2
            by = -math.sin(angle_to_target) * 0.2
            p.resetBaseVelocity(robot_id, [bx, by, 0], [0, 0, 0])
            p.stepSimulation()
            time.sleep(0.01)
        
        # Retreat to safe zone
        print(f"[{robot_name}] Retreating to safe zone...")
        retreat_map = {
            "robot_0": [-2, -2, 0],
            "robot_1": [-2, 0, 0],
            "robot_2": [-2, 2, 0]
        }
        safe_pos = retreat_map.get(robot_name, [-3, 0, 0])
        _navigate_to_coords(robot_name, safe_pos, threshold=0.5)
        
        return True
    else:
        print(f"[{robot_name}] Failed to reach standoff coordinates.")
        return False

'''

# Reconstruct the file
new_content = before_place_at + new_place_at + after_place_at

# Write it back
with open('world_building_construction.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("File fixed successfully!")
print(f"Applied robot stabilization fix to place_at function")
