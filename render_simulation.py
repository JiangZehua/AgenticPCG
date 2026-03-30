#!/usr/bin/env python3
"""Render simulation GIF from an existing finished run (SMB or Zelda).

Re-evaluates the final level to obtain simulation data, then produces a
simulation.gif showing the playthrough.

Usage:
    uv run python render_simulation.py runs/20250101_model_smb/
    uv run python render_simulation.py runs/20250101_model_zelda/
    uv run python render_simulation.py runs/sweep/model_*/trial_*/
"""

import argparse
import sys
from pathlib import Path

from toolusepcg.config import load_config, get_problem_type
from toolusepcg.evaluator import StatsEvaluator
from toolusepcg.factory import LevelFactory
from toolusepcg.level import Level
from toolusepcg.logging.trace import TraceLogger


def render_for_run(run_dir: Path, render_stats: bool = True) -> bool:
    """Render simulation GIF for a single run directory.

    Args:
        run_dir: Path to a run directory containing config.yaml and final_level.txt.
        render_stats: Whether to overlay metrics/targets on each frame.

    Returns:
        True if GIF was produced, False otherwise.
    """
    config_path = run_dir / "config.yaml"
    level_path = run_dir / "final_level.txt"

    if not config_path.exists():
        print(f"  SKIP: no config.yaml in {run_dir}")
        return False

    if not level_path.exists():
        print(f"  SKIP: no final_level.txt in {run_dir}")
        return False

    # Load config and check problem type
    config = load_config(str(config_path))
    problem_type = get_problem_type(config.env.name)

    if problem_type not in ("smb", "zelda"):
        print(f"  SKIP: problem type is '{problem_type}', not smb or zelda")
        return False

    # Create environment and evaluate
    factory = LevelFactory(config)
    env = factory.env
    evaluator = StatsEvaluator(config, env)

    level = Level.from_string(level_path.read_text(), problem_type=problem_type)
    result = evaluator.evaluate(level)
    info = result.info or {}

    if problem_type == "smb":
        locations = info.get("locations", [])
        if not locations:
            print(f"  SKIP: no simulation locations (level may be unsolvable)")
            return False

        logger = TraceLogger(config, run_dir=str(run_dir), render_sim=True, render_stats=render_stats)
        logger.set_env(env)
        metrics = result.metrics if render_stats else None
        gif_path = logger.render_simulation_gif(level.data, info, metrics=metrics)

        if gif_path:
            print(f"  OK: {gif_path} ({len(locations)} frames)")
            return True

    elif problem_type == "zelda":
        pk_path = info.get("pk_path", [])
        kd_path = info.get("kd_path", [])
        if not pk_path or not kd_path:
            print(f"  SKIP: no solution path (level may be unsolvable)")
            return False

        logger = TraceLogger(config, run_dir=str(run_dir), render_sim=True, render_stats=render_stats)
        logger.set_env(env)
        metrics = result.metrics if render_stats else None
        gif_path = logger.render_zelda_simulation_gif(level.data, info, metrics=metrics)

        if gif_path:
            total_steps = len(pk_path) + len(kd_path) - 1
            print(f"  OK: {gif_path} ({total_steps} path steps)")
            return True

    print(f"  FAIL: render returned None")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Render simulation GIF (SMB/Zelda) from existing finished run(s)."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more run directories containing config.yaml and final_level.txt",
    )
    parser.add_argument(
        "--render-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay metrics/targets on each frame (default: on, use --no-render-stats to disable)",
    )
    args = parser.parse_args()

    success = 0
    total = len(args.run_dirs)

    for run_dir in args.run_dirs:
        print(f"[{run_dir}]")
        try:
            if render_for_run(run_dir, render_stats=args.render_stats):
                success += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    if total > 1:
        print(f"\nRendered {success}/{total} simulation GIFs.")

    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
