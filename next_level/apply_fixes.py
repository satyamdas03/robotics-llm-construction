# Script to apply extended stabilization and retreat fixes to place_at function

with open('world_building_construction.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the place_at function
place_at_start = None
for i, line in enumerate(lines):
    if 'def place_at(robot_name, target_coords_str):' in line:
        place_at_start = i
        break

if place_at_start is None:
    print("ERROR: Could not find place_at function")
    import sys
    sys.exit(1)

# Find specific sections to replace:
# 1. Find "# Stabilize with correct position" section
# 2. Replace until we hit "return True" (in the main if block)

# Find the stabilization section
stabilize_start = None
for i in range(place_at_start, len(lines)):
    if '# Stabilize with correct position' in lines[i]:
        stabilize_start = i
        break

if stabilize_start is None:
    print("ERROR: Could not find stabilization section")
    import sys
    sys.exit(1)

# Find the end (look for the return True in the main if block, after the retreat)
retreat_end = None
for i in range(stabilize_start, min(stabilize_start + 100, len(lines))):
    # Look for "return True" that's at the same indentation level as the if _navigate_to_coords block
    if 'return True' in lines[i] and lines[i].startswith('        return'):
        retreat_end = i
        break

if retreat_end is None:
    print("ERROR: Could not find end of retreat section")
    import sys
    sys.exit(1)

print(f"Found stabilization section at line {stabilize_start}")
print(f"Found retreat end at line {retreat_end}")

# New stabilization and retreat code
new_code = '''        # FIX: Increased stabilization with both robot and block frozen
        for _ in range(50):  # Increased from 30
            p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
            p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])  # Also freeze block
            p.stepSimulation()
            time.sleep(0.01)

        # Release constraint
        p.removeConstraint(constraint_id)
        ROBOT_STATE[robot_name]["current_constraint"] = None
        ROBOT_STATE[robot_name]["held_object_name"] = None 
        print(f"[{robot_name}] Placed object at {coords}.")
        
        # FIX: Increase block friction to prevent sliding/falling
        p.changeDynamics(object_id, -1, 
            lateralFriction=2.0,      # Increased from default
            spinningFriction=0.5,
            rollingFriction=0.5)
        
        # FIX: Keep robot frozen even after release for settling
        print(f"[{robot_name}] Waiting for block to settle...")
        for _ in range(50):  # NEW: Additional settling time
            p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
            p.stepSimulation()
            time.sleep(0.01)
        
        # FIX: NOW restore robot mobility for retreat
        p.changeDynamics(robot_id, -1, mass=original_mass)
        
        # FIX: Improved retreat - move perpendicular first to clear stack area
        print(f"[{robot_name}] Moving sideways to clear stack...")
        perp_angle = angle_to_target + math.pi/2  # 90 degrees perpendicular
        for _ in range(30):
            sx = math.cos(perp_angle) * 0.15
            sy = math.sin(perp_angle) * 0.15
            p.resetBaseVelocity(robot_id, [sx, sy, 0], [0, 0, 0])
            p.stepSimulation()
            time.sleep(0.01)
        
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
'''

# Reconstruct the file
new_lines = lines[:stabilize_start] + [new_code] + lines[retreat_end+1:]

# Write it back
with open('world_building_construction.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Successfully applied extended stabilization and retreat fixes!")
print(f"Modified lines {stabilize_start} to {retreat_end}")
print(f"Old line count: {len(lines)}, New line count: {len(new_lines)}")
