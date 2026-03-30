#!/usr/bin/env python3
"""Render per-tool-call edit animation from a ToolUsePCG run.

Unlike run_animation.gif (which shows one frame per step), this script
renders each individual tool call within each step, so you can see the
level being built up tool-by-tool.

Usage:
    uv run python render_per_tool_call_edit.py runs/some_run/
    uv run python render_per_tool_call_edit.py runs/some_run/ --all-steps
    uv run python render_per_tool_call_edit.py runs/some_run/ --frame-duration 300
    uv run python render_per_tool_call_edit.py runs/some_run/ --output custom.gif
    uv run python render_per_tool_call_edit.py runs/run1/ runs/run2/
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from toolusepcg.config import load_config, get_problem_type
from toolusepcg.factory import LevelFactory
from toolusepcg.level import Level
from toolusepcg.logging.trace import TraceLogger, create_edit_overlay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_trace(trace_path: Path) -> list[dict]:
    entries = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Main rendering logic
# ---------------------------------------------------------------------------
def render_edits_gif(
    run_dir: Path,
    *,
    all_steps: bool = False,
    frame_duration: int = 400,
    hold_duration: int = 1500,
    output: str | None = None,
) -> Path | None:
    """Render per-tool-call animation GIF for a run.

    When all_steps is False, delegates to TraceLogger.render_per_tool_edit_gif().
    When all_steps is True, includes rejected steps with rollback.

    Args:
        run_dir: Path to run directory containing config.yaml and trace.jsonl.
        all_steps: If True, include rejected steps too.
        frame_duration: Duration per tool-call frame in ms.
        hold_duration: Duration for initial/final frames in ms.
        output: Custom output filename (default: per_tool_edit_process.gif).

    Returns:
        Path to saved GIF, or None on failure.
    """
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        config_path = run_dir / "config_resolved.yaml"
    trace_path = run_dir / "trace.jsonl"

    if not config_path.exists() or not trace_path.exists():
        print(f"  SKIP: missing config.yaml or trace.jsonl in {run_dir}")
        return None

    config = load_config(str(config_path))
    problem_type = get_problem_type(config.env.name)
    factory = LevelFactory(config)
    env = factory.env

    logger = TraceLogger(config, run_dir=str(run_dir), render_stats=True)
    logger.set_env(env)

    # Simple case: accepted steps only — delegate to TraceLogger method
    if not all_steps and output is None:
        return logger.render_per_tool_edit_gif(frame_duration, hold_duration)

    # Extended case: custom output name or --all-steps with rollback
    border_offset = logger._get_border_offset()
    entries = _load_trace(trace_path)

    initial_entry = next((e for e in entries if e["event"] == "initial_level"), None)
    if initial_entry is None:
        print(f"  SKIP: no initial_level in trace.jsonl")
        return None

    current_level = Level.from_string(initial_entry["level"], problem_type=problem_type)
    current_metrics = initial_entry.get("metrics", {})

    frames: list[Image.Image] = []
    durations: list[int] = []

    def add_frame(img: Image.Image, duration: int):
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (30, 30, 30))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")
        frames.append(img)
        durations.append(duration)

    def render_with_stats(level_data, metrics):
        base_img = logger._render_level(level_data)
        if base_img is None:
            return None
        return logger._compose_with_stats(base_img, metrics)

    # Initial level frame
    init_img = render_with_stats(current_level.data, current_metrics)
    if init_img is None:
        print(f"  SKIP: failed to render initial level")
        return None
    add_frame(init_img, hold_duration)

    # Process steps
    for entry in entries:
        if entry.get("event") != "step":
            continue

        accepted = entry.get("accepted", False)
        tool_results = entry.get("tool_results", [])

        if not accepted and not all_steps:
            continue
        if not tool_results:
            continue

        level_before_step = current_level.copy()
        metrics_during = entry.get("metrics_before", current_metrics)
        metrics_after = entry.get("metrics_after", metrics_during)
        name_tiles = current_level.name_tiles

        for tr in tool_results:
            result = tr.get("result", {})
            if not result.get("success"):
                continue
            diff = result.get("data", {}).get("diff", [])
            if not diff:
                continue

            for d in diff:
                tile_val = name_tiles.get(d["new"])
                if tile_val is not None:
                    current_level.data[d["y"], d["x"]] = tile_val

            base_img = logger._render_level(current_level.data)
            if base_img is None:
                continue

            edit_img = create_edit_overlay(base_img, diff, border_offset=border_offset)
            img_with_stats = logger._compose_with_stats(edit_img, metrics_during)
            add_frame(img_with_stats, frame_duration)

        result_img = render_with_stats(current_level.data, metrics_after)
        if result_img is not None:
            add_frame(result_img, frame_duration + 200)

        current_metrics = metrics_after

        if not accepted:
            current_level = level_before_step

    # Final level frame
    final_img = render_with_stats(current_level.data, current_metrics)
    if final_img is not None:
        add_frame(final_img, hold_duration)

    if len(frames) < 2:
        print(f"  SKIP: not enough frames to create GIF")
        return None

    max_w = max(f.width for f in frames)
    max_h = max(f.height for f in frames)
    normalized = []
    for f in frames:
        if f.size != (max_w, max_h):
            canvas = Image.new("RGB", (max_w, max_h), (30, 30, 30))
            canvas.paste(f, (0, 0))
            normalized.append(canvas)
        else:
            normalized.append(f)

    gif_name = output or "per_tool_edit_process.gif"
    gif_path = run_dir / gif_name
    normalized[0].save(
        gif_path,
        save_all=True,
        append_images=normalized[1:],
        duration=durations,
        loop=0,
    )
    return gif_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Render per-tool-call edit animation from ToolUsePCG runs."
    )
    parser.add_argument(
        "run_dirs", nargs="+", type=Path,
        help="Run directories containing config.yaml and trace.jsonl",
    )
    parser.add_argument(
        "--all-steps", action="store_true",
        help="Include rejected steps too (with rollback)",
    )
    parser.add_argument(
        "--frame-duration", type=int, default=400,
        help="Duration per tool-call frame in ms (default: 400)",
    )
    parser.add_argument(
        "--hold-duration", type=int, default=1500,
        help="Duration for initial/final frames in ms (default: 1500)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output filename (default: per_tool_edit_process.gif)",
    )

    args = parser.parse_args()

    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        print(f"Processing {run_dir} ...")
        gif_path = render_edits_gif(
            run_dir,
            all_steps=args.all_steps,
            frame_duration=args.frame_duration,
            hold_duration=args.hold_duration,
            output=args.output,
        )
        if gif_path:
            total_tools = 0
            entries = _load_trace(run_dir / "trace.jsonl")
            for e in entries:
                if e.get("event") == "step":
                    if e.get("accepted", False) or args.all_steps:
                        for tr in e.get("tool_results", []):
                            if tr.get("result", {}).get("success") and tr.get("result", {}).get("data", {}).get("diff"):
                                total_tools += 1
            print(f"  OK: {gif_path} ({total_tools} tool-call frames)")
        else:
            print(f"  FAILED")


if __name__ == "__main__":
    main()
