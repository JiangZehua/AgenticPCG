"""Resume state reconstruction from a previous run's trace data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import get_problem_type


@dataclass
class ResumeState:
    """Reconstructed state from a previous run at a specific step."""

    config_dict: dict
    """Fully-resolved config (without ``runtime`` key)."""

    runtime_args: dict
    """The ``runtime`` key from saved config."""

    level_str: str
    """ASCII level at the resume step."""

    problem_type: str
    """Problem type inferred from config (e.g. 'binary', 'smb')."""

    resume_step: int
    """Step number resumed from."""

    total_changes: int
    """Cumulative accepted tiles changed through resume_step."""

    current_score: float
    """Score after last accepted step (or initial score)."""

    current_metrics: dict = field(default_factory=dict)
    """Metrics after last accepted step (or initial metrics)."""


def load_resume_state(
    run_dir: str | Path,
    resume_step: int | None = None,
) -> ResumeState:
    """Load resume state from a previous run directory.

    Parses ``config.yaml`` and ``trace.jsonl`` to reconstruct the optimizer
    state at a given step.

    Args:
        run_dir: Path to the previous run directory.
        resume_step: Step number to resume from.  If ``None``, resumes from
            the last recorded step.

    Returns:
        ResumeState with reconstructed level, score, metrics, and config.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If resume_step is beyond the available steps.
    """
    run_dir = Path(run_dir)

    # --- Load config.yaml ---------------------------------------------------
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {run_dir}")

    with open(config_path) as f:
        saved_config = yaml.safe_load(f) or {}

    runtime_args = saved_config.pop("runtime", {})
    config_dict = saved_config

    # Infer problem type
    problem_type = get_problem_type(config_dict.get("env", {}).get("name", "binary-v0"))

    # --- Parse trace.jsonl ---------------------------------------------------
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"No trace.jsonl found in {run_dir}")

    level_str: str | None = None
    current_score: float = 0.0
    current_metrics: dict = {}
    total_changes: int = 0
    max_step_seen: int = 0

    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            event = entry.get("event")

            if event == "initial_level":
                level_str = entry["level"]
                current_metrics = entry.get("metrics", {})
                current_score = 0.0
                continue

            if event == "step":
                step_num = entry.get("step_num", 0)
                max_step_seen = max(max_step_seen, step_num)

                # If resume_step specified, skip steps beyond it
                if resume_step is not None and step_num > resume_step:
                    continue

                if entry.get("accepted", False):
                    # Extract level from tool results
                    new_level = _extract_level_from_step(entry.get("tool_results", []))
                    if new_level is not None:
                        level_str = new_level
                    current_score = entry.get("score_after", current_score)
                    current_metrics = entry.get("metrics_after", current_metrics)
                    total_changes += entry.get("num_tiles_changed", 0)

    # Determine effective resume step
    if resume_step is None:
        effective_step = max_step_seen
    else:
        if resume_step > max_step_seen:
            raise ValueError(
                f"--resume-step {resume_step} is beyond the last recorded step "
                f"({max_step_seen}) in {run_dir}"
            )
        effective_step = resume_step

    if level_str is None:
        raise ValueError(f"No level data found in {trace_path}")

    return ResumeState(
        config_dict=config_dict,
        runtime_args=runtime_args,
        level_str=level_str,
        problem_type=problem_type,
        resume_step=effective_step,
        total_changes=total_changes,
        current_score=current_score,
        current_metrics=current_metrics,
    )


def _extract_level_from_step(tool_results: list[dict]) -> str | None:
    """Extract the level string from a step's tool results.

    Iterates tool results in reverse to find the last successful edit that
    produced a ``new_level``.

    Args:
        tool_results: List of tool result dicts from a trace step event.

    Returns:
        The level string, or None if no edit produced a new level.
    """
    for tr in reversed(tool_results):
        result = tr.get("result", {})
        if result.get("success") and result.get("data", {}).get("new_level"):
            return result["data"]["new_level"]
    return None
