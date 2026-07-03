# Plan: Uncertainty-Aware Physics-Coupled Multi-Robot Planner

## Goal
Build the first local 8B-model multi-robot construction planner that uses fast PyBullet physics rollouts to resolve uncertainty before execution. The simulator becomes an active reasoning module rather than a passive evaluator.

## Base Code
Start from `next_level/world_building_construction_COMPLETE.py` (808 lines, most complete stacking demo). Create a new evolution: `next_level/world_building_construction_uncertainty.py` so the working version remains untouched.

## Core Design

### 1. Uncertainty Scoring
Primary: **temperature ensemble**. Generate the same plan 3 times with `temperature=0.8`. Steps that vary across samples get higher uncertainty. Steps that are identical get low uncertainty.

Secondary: **self-reported confidence**. Prompt the model to also output `uncertainty` (0.0-1.0) per step. Use it for display, but trust the ensemble for routing.

Threshold: steps with ensemble uncertainty ≥ 0.5 are routed to a physics rollout before execution.

### 2. World Snapshot / Rollback
Implement `save_world_state()` and `load_world_state()` that capture:
- PyBullet `saveState()` / `restoreState()` for all body poses, velocities, and constraints
- `ROBOT_STATE` dict (held_object_name, current_constraint) per robot
- Any `p.changeDynamics` overrides (e.g., mass=0 from place_at freezing)

Unit test: randomize world, save, perturb, restore, diff positions/velocities.

### 3. Branch Simulator
Implement `simulate_branch(robot_name, plan_steps, horizon_seconds=0.5)`:
- Save current world state
- For each plan step, run the same skill function but in a fast DIRECT-mode client or by temporarily disabling GUI sleeps
- Return: success boolean, failure step, final robot/object positions, collision count, max contact force
- Restore world state before returning

Execution path: uncertain steps are validated first. If the branch fails, the failure text is fed back to the LLM for replanning. If it succeeds, the step is executed for real.

### 4. Planner Output Schema Extension
Current plan format:
```json
[{"function": "move_to", "target": "block_red"}, ...]
```

New plan format (backward-compatible parser):
```json
[{"function": "move_to", "target": "block_red", "uncertainty": 0.2}, ...]
```

The parser accepts both old and new formats. Missing `uncertainty` defaults to 0.0.

### 5. Integration into Executor
Modify `parse_and_execute(robot_name, plan)`:
- Iterate plan steps
- For each step, compute ensemble uncertainty
- If uncertainty < 0.5: execute immediately
- If uncertainty ≥ 0.5: call `simulate_branch`. On failure, return failure info and trigger replan loop in `main()`
- On success, execute the real step

### 6. Demo Task
Command: *"Build a red-blue alternating wall using all robots."*
The system must:
- Decompose the wall into a block sequence
- Allocate blocks to robots
- Tag uncertain placements (top rows, collisions, heavy blocks)
- Validate uncertain placements in PyBullet
- Execute validated placements with `place_at`

### 7. Metrics & Logging
Log to a JSONL file per run:
- Plan acceptance rate
- Number of rollouts performed
- Rollout failures avoided
- End-to-end success
- Per-step uncertainty distribution
- Total time

## File Changes

### New files
- `next_level/world_building_construction_uncertainty.py` — main implementation
- `next_level/test_snapshot.py` — unit tests for save/load world state
- `next_level/experiments/uncertainty_calibration.py` — compare self-reported vs ensemble uncertainty
- `next_level/experiments/metrics_2026-07-03.jsonl` — run logs

### Read-only analysis of existing files
- `world_building_construction_COMPLETE.py` — base code (read)
- `world_building_construction.py` — compare for any extra fixes to port

## First-Week Sprint

| Day | Deliverable |
|-----|-------------|
| 1 | Extend planner output to carry optional `uncertainty`; update parser for backward compatibility. Generate 10 plans and plot uncertainty distribution. |
| 2 | Implement deterministic `save_world_state()` / `load_world_state()` covering body poses, velocities, constraints, and robot mental state. Unit test with `test_snapshot.py`. |
| 3 | Build `simulate_branch()` in DIRECT mode; return success/failure and outcome metrics. |
| 4 | Wire uncertainty threshold into executor; failed branches trigger replan with failure text. |
| 5 | Calibrate confidence: compare self-reported vs ensemble; switch to whichever correlates better with rollout outcomes. |
| 6 | Run the red-blue wall demo; record screen and logs. |
| 7 | Clean code, write README section, commit. |

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| llama3:8b self-reported confidence is unreliable | Use temperature ensemble as the routing signal, not self-reported confidence |
| Snapshot misses active constraints or dynamics overrides | Save all constraint IDs and `changeDynamics` state; unit test exhaustively |
| Multi-robot collisions during rollouts are missed | For wall demo, sequence robots; later add collision-aware path checks |
| JSON schema drift breaks parser | Parser accepts both old and new formats; default uncertainty = 0.0 |
| Ollama latency slows ensemble 3× | Reuse connection, run generations sequentially; cache world-state prompt text |
| Rollout in same GUI client is slow or visually jarring | Run rollouts in `p.DIRECT` secondary client, not the GUI client |

## Success Criteria
- [ ] A natural-language command produces a multi-robot block wall.
- [ ] At least 30% of placement steps trigger a physics rollout.
- [ ] Rollout failures are detected and trigger successful replans ≥ 50% of the time.
- [ ] Demo runs fully on the local Windows laptop with no cloud API calls.
- [ ] Screen recording + JSONL metrics exist for the final run.

## Why This Direction
This is the shortest path from the existing codebase to a genuinely novel result. It reuses the multi-robot controller, `place_at` skill, OpenCV vision, and Ollama planner, but adds the missing loop that turns the simulator into an active reasoning module. The empirical claim is concrete and defensible: first local 8B-model multi-robot construction planner with physics rollouts for uncertainty resolution.
