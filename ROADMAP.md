# Roadmap — Revolutionary Next Directions

Generated: 2026-07-03 via multi-agent brainstorming workflow.

## #1 Recommendation: PAL — Push-Align-Latch (score 36)

**Pitch:** Add a first-class `push_to` primitive so the R2-D2 robots can nudge, slide, and tap blocks into position using controlled contact, not only pick-and-place.

**Why revolutionary:** Most LLM block-construction demos treat blocks as free-flying pose targets. PAL treats contact and non-prehensile pushing as a core skill, letting the local uncertainty-ensemble LLM planner query PyBullet as a dynamics oracle for multi-robot pushing and gap-filling — skills impossible with pure pick-and-place.

**Caveman first step:** Extend the existing JSON skill schema with `push_to` fields (`block_id`, `target_xy`, `approach_angle`, `contact_speed`, `push_distance`) and implement the primitive in PyBullet by driving the robot toward the block at constant velocity, monitoring `getContactPoints` force, stopping on force threshold or push distance, then backing off.

**Biggest risk:** Pushing is far more sensitive to contact geometry, friction, approach angle, and speed than teleport-style pick-and-place; the R2-D2 holonomic base is not designed for sustained controlled contact, so pushes may topple walls or the rollout loop may reject most attempts, and the existing string-equality ensemble will misfire on near-identical continuous push parameters.

**Demo idea:** A tight red-blue-red wall where the middle blue block must be slid horizontally between two red pillars because vertical clearance is too small to place it.

## #2: Plan Archaeology — Fossilize Winning Subplans (score 35)

**Pitch:** After each construction episode, archive successful JSON skill sub-sequences as reusable composite "fossil" skills with preconditions, and seed future LLM prompts with the best matching fossils while injecting past failures as "tar pit" negative examples.

**Why revolutionary:** Most LLM planners either replan from scratch or require cloud fine-tuning and external vector databases to improve. This keeps learning entirely local, interpretable, and physics-grounded.

**Caveman first step:** Create `memory/fossils.jsonl`, `memory/tarpits.jsonl`, and `skills/fossils.json`, then run a small `fossilize.py` post-episode script that scans logs, extracts successful contiguous sub-sequences longer than one primitive, and registers each as a composite skill entry with precondition, steps, and success_count.

**Biggest risk:** Signature overfitting: coarse discretized state matching plus L-infinity tolerance will falsely recall fossils in superficially similar but mechanically different states.

## #3: Caveman Voxel Belief Maps (VBM) (score 34)

**Pitch:** Give each robot a cheap shared 3D belief grid instead of treating every HSV detection as ground truth, so the LLM plans from ranked, confidence-weighted block estimates and can issue explicit verification steps when uncertainty is high.

**Why revolutionary:** Most perception pipelines treat vision as a binary predicate. VBM turns monocular HSV masks into a probabilistic, multi-robot world model and makes perception uncertainty a first-class planning input.

**Caveman first step:** Add a fixed-resolution numpy voxel grid over the known table storing color_id, occupancy_confidence, color_confidence, and last_seen timestamp per robot; back-project each HSV blob through the camera ray to the table plane and merge maps via confidence-weighted averaging.

**Biggest risk:** The LLM may ignore confidence thresholds and keep issuing pickup commands; UDP multicast is overkill since the demo runs in one Python process — use a shared dict.

## Honorable mentions

- **Monte Carlo Physics Certificate (MCPC)** — statistical certifier for `place_at`. Well-scoped, but undermined until `place_at` is made physically honest.
- **Counterfactual Physics Tribunal (CPT)** — LLM prosecutor/defender/judge debate. Novel but too brittle for local LLMs right now.
