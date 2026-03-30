# Visualization & Analysis

## Simulation Rendering (SMB / Zelda)

Re-render simulation GIFs from existing runs. Non-SMB/Zelda directories are skipped.

```bash
uv run python render_simulation.py runs/some_smb_run/
uv run python render_simulation.py runs/sweep/*smb*/trial_*/   # Glob multiple runs
uv run python render_simulation.py --no-render-stats runs/some_smb_run/  # Without stats overlay
```

## Trajectory Visualization (SMB)

Static PNG showing player and enemy movement trajectories.

```bash
uv run python render_trajectory.py runs/some_smb_run/
uv run python render_trajectory.py runs/run1/ runs/run2/ --scale 3 --subsample 5
```

Output `trajectory.png` shows:
- **Spawn positions**: semi-transparent sprites at starting locations
- **Movement paths**: dashed lines — red for Mario, unique colors per enemy
- **Kill markers**: red X at stomp/fire/shell kills (fall kills excluded)

## Per-Tool-Call Edit Animation

`per_tool_edit_process.gif` is auto-generated at the end of each run. Shows each tool call's tile changes highlighted (tool-by-tool), unlike `run_animation.gif` which shows one frame per step.

```bash
uv run python render_per_tool_call_edit.py runs/some_run/
uv run python render_per_tool_call_edit.py runs/some_run/ --all-steps     # Include rejected (with rollback)
uv run python render_per_tool_call_edit.py runs/some_run/ --frame-duration 300  # Faster playback
uv run python render_per_tool_call_edit.py sweeps/some_sweep/**/trial_*/  # Multiple runs
```
