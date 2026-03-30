#!/usr/bin/env python3
"""Experiment launcher with parallel trial execution.

This script orchestrates multi-trial experiments by spawning parallel
worker processes that each run a single trial via main.py.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrialConfig:
    """Configuration for a single trial."""
    trial_id: int
    config_path: str
    run_dir: Path
    max_steps: int
    seed: int
    target_overrides: dict[str, Any] | None = None
    verbose: bool = False
    debug: bool = False
    model: str | None = None
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int | None = None
    render_stats: bool = True
    render_sim: bool = False
    edit_tools: list[str] | None = None
    init: str | None = None
    max_tool_calls: int | None = None
    epsilon: float | None = None
    sa_temp: float | None = None
    sa_decay: float | None = None
    conv_length: int | None = None
    conv_filter: str | None = None
    change_percentage: float | None = None
    smb_solver: str | None = None


def run_single_trial(trial_config: TrialConfig) -> dict:
    """Run a single trial as a subprocess.

    Args:
        trial_config: Configuration for this trial.

    Returns:
        Dictionary with trial results.
    """
    trial_dir = trial_config.run_dir / f"trial_{trial_config.trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        sys.executable, "main.py",
        "--config", trial_config.config_path,
        "--run-dir", str(trial_dir),
        "--max-steps", str(trial_config.max_steps),
        "--seed", str(trial_config.seed),
    ]

    if not trial_config.verbose:
        cmd.append("--quiet")

    if trial_config.debug:
        cmd.append("--debug")

    if trial_config.model:
        cmd.extend(["--model", trial_config.model])

    if trial_config.provider:
        cmd.extend(["--provider", trial_config.provider])

    if trial_config.base_url:
        cmd.extend(["--base-url", trial_config.base_url])

    if trial_config.api_key:
        cmd.extend(["--api-key", trial_config.api_key])

    if trial_config.max_tokens:
        cmd.extend(["--max-tokens", str(trial_config.max_tokens)])

    if not trial_config.render_stats:
        cmd.append("--no-render-stats")

    if trial_config.render_sim:
        cmd.append("--render-sim")

    if trial_config.edit_tools is not None:
        cmd.extend(["--edit-tools"] + trial_config.edit_tools)

    if trial_config.init is not None:
        cmd.extend(["--init", trial_config.init])

    if trial_config.max_tool_calls is not None:
        cmd.extend(["--max-tool-calls", str(trial_config.max_tool_calls)])

    if trial_config.epsilon is not None:
        cmd.extend(["--epsilon", str(trial_config.epsilon)])

    if trial_config.sa_temp is not None:
        cmd.extend(["--sa-temp", str(trial_config.sa_temp)])

    if trial_config.sa_decay is not None:
        cmd.extend(["--sa-decay", str(trial_config.sa_decay)])

    if trial_config.conv_length is not None:
        cmd.extend(["--conv-length", str(trial_config.conv_length)])

    if trial_config.conv_filter is not None:
        cmd.extend(["--conv-filter", trial_config.conv_filter])

    if trial_config.change_percentage is not None:
        cmd.extend(["--change-percentage", str(trial_config.change_percentage)])

    if trial_config.smb_solver is not None:
        cmd.extend(["--smb-solver", trial_config.smb_solver])

    # Add target overrides
    if trial_config.target_overrides:
        for metric, value in trial_config.target_overrides.items():
            cmd.extend(["--target", metric, str(value)])

    # Run subprocess
    # Timeout scales with max_steps: ~2 min per step + 5 min buffer
    timeout_seconds = trial_config.max_steps * 120 + 300
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        # Parse summary if available
        summary_path = trial_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        else:
            summary = {}

        return {
            "trial_id": trial_config.trial_id,
            "success": result.returncode == 0,
            "run_dir": str(trial_dir),
            "summary": summary,
            "seed": trial_config.seed,
            "target_overrides": trial_config.target_overrides,
            "returncode": result.returncode,
            "stderr": result.stderr[-1000:] if result.stderr else "",  # Last 1000 chars
        }

    except subprocess.TimeoutExpired:
        return {
            "trial_id": trial_config.trial_id,
            "success": False,
            "run_dir": str(trial_dir),
            "summary": {},
            "seed": trial_config.seed,
            "target_overrides": trial_config.target_overrides,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "trial_id": trial_config.trial_id,
            "success": False,
            "run_dir": str(trial_dir),
            "summary": {},
            "seed": trial_config.seed,
            "target_overrides": trial_config.target_overrides,
            "error": str(e),
        }


class ProgressTracker:
    """Tracks and displays progress for parallel trials."""

    def __init__(self, total_trials: int, update_interval: float = 5.0):
        self.total = total_trials
        self.completed = 0
        self.failed = 0
        self.running = 0
        self.update_interval = update_interval
        self.start_time = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start the progress display thread."""
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the progress display thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def trial_started(self):
        """Mark a trial as started."""
        with self._lock:
            self.running += 1

    def trial_completed(self, success: bool):
        """Mark a trial as completed."""
        with self._lock:
            self.running -= 1
            if success:
                self.completed += 1
            else:
                self.failed += 1

    def _display_loop(self):
        """Background loop that displays progress updates."""
        while not self._stop_event.is_set():
            self._print_status()
            self._stop_event.wait(self.update_interval)

    def _print_status(self):
        """Print current status."""
        with self._lock:
            elapsed = time.time() - self.start_time if self.start_time else 0
            done = self.completed + self.failed
            pending = self.total - done - self.running

            status_parts = [
                f"[{elapsed:.0f}s]",
                f"Running: {self.running}",
                f"Done: {done}/{self.total}",
            ]
            if self.failed > 0:
                status_parts.append(f"(Failed: {self.failed})")
            if pending > 0:
                status_parts.append(f"Pending: {pending}")

            print(f"\r{' | '.join(status_parts)}", end="", flush=True)


class ExperimentLauncher:
    """Orchestrates parallel experiment execution."""

    def __init__(
        self,
        config_path: str,
        experiment_type: str,
        num_trials: int,
        max_workers: int,
        max_steps: int,
        base_seed: int,
        sampling_seed: int | None,
        run_dir: Path | None,
        verbose: bool,
        debug: bool = False,
        model: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        render_stats: bool = True,
        render_sim: bool = False,
        edit_tools: list[str] | None = None,
        init: str | None = None,
        max_tool_calls: int | None = None,
        epsilon: float | None = None,
        sa_temp: float | None = None,
        sa_decay: float | None = None,
        conv_length: int | None = None,
        conv_filter: str | None = None,
        change_percentage: float | None = None,
        smb_solver: str | None = None,
    ):
        self.config_path = config_path
        self.experiment_type = experiment_type
        self.num_trials = num_trials
        self.max_workers = max_workers
        self.max_steps = max_steps
        self.base_seed = base_seed
        self.sampling_seed = sampling_seed
        self.verbose = verbose
        self.debug = debug
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.render_stats = render_stats
        self.render_sim = render_sim
        self.edit_tools = edit_tools
        self.init = init
        self.max_tool_calls = max_tool_calls
        self.epsilon = epsilon
        self.sa_temp = sa_temp
        self.sa_decay = sa_decay
        self.conv_length = conv_length
        self.conv_filter = conv_filter
        self.change_percentage = change_percentage
        self.smb_solver = smb_solver

        # Load config to get problem type and settings
        from toolusepcg.config import load_config, get_problem_type, get_control_ranges, MetricRange

        self.config = load_config(config_path)
        self.problem_type = get_problem_type(self.config.env.name)

        # Set up run directory
        if run_dir:
            self.run_dir = Path(run_dir)
        else:
            from toolusepcg.config import sanitize_model_name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use model override if provided, otherwise fall back to config default
            model_str = self.model if self.model else self.config.llm.model
            model_name = sanitize_model_name(model_str)
            self.run_dir = Path("runs") / f"{timestamp}_{model_name}_{self.problem_type}_{experiment_type}"

        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sampling RNG
        import random
        self.rng = random.Random(sampling_seed)

        # Get control ranges for controllability experiments
        self.control_ranges = {}
        if experiment_type == "controllability":
            # Check config for custom ranges
            if (self.config.experiment.controllability and
                self.config.experiment.controllability.ranges):
                for param, range_dict in self.config.experiment.controllability.ranges.items():
                    self.control_ranges[param] = MetricRange(
                        min_value=range_dict.get('min_value', 0),
                        max_value=range_dict.get('max_value', 100),
                        step=range_dict.get('step', 1),
                    )
            else:
                self.control_ranges = get_control_ranges(
                    self.problem_type,
                    self.config.env.height,
                    self.config.env.width,
                )

        self.trial_results: list[dict] = []

        # Save experiment-level args (after control_ranges is initialized)
        self._save_experiment_args()

    def _save_experiment_args(self) -> None:
        """Save experiment-level arguments to experiment_config.yaml.

        Saves the effective config that actually ran: experiment orchestration
        args plus the fully resolved optimization config. Omits null values
        and false-by-default flags to keep the output clean.
        """
        import yaml
        from toolusepcg.config import config_to_dict

        args_dict = {
            "experiment_type": self.experiment_type,
            "num_trials": self.num_trials,
            "max_workers": self.max_workers,
            "max_steps": self.max_steps,
            "base_seed": self.base_seed,
            "sampling_seed": self.sampling_seed,
            "problem_type": self.problem_type,
        }

        # Only include non-null, non-default-false values
        if self.render_stats is not True:
            args_dict["render_stats"] = self.render_stats
        if self.render_sim:
            args_dict["render_sim"] = self.render_sim
        if self.verbose:
            args_dict["verbose"] = self.verbose
        if self.debug:
            args_dict["debug"] = self.debug
        if self.init:
            args_dict["init"] = self.init

        # Include control ranges for controllability experiments
        if self.experiment_type == "controllability" and self.control_ranges:
            args_dict["control_ranges"] = {
                name: {"min_value": r.min_value, "max_value": r.max_value, "step": r.step}
                for name, r in self.control_ranges.items()
            }

        # Include the resolved per-trial config (env, targets, scoring, etc.)
        args_dict["resolved_config"] = config_to_dict(self.config)

        with open(self.run_dir / "experiment_config.yaml", "w") as f:
            yaml.dump(args_dict, f, default_flow_style=False)

    def _sample_control_params(self) -> dict[str, int | float]:
        """Sample control parameters for controllability experiment."""
        sampled = {}
        for param_name, param_range in self.control_ranges.items():
            min_val = param_range.min_value
            max_val = param_range.max_value
            step = param_range.step

            if step == 1 and isinstance(min_val, int) and isinstance(max_val, int):
                sampled[param_name] = self.rng.randint(int(min_val), int(max_val))
            else:
                num_steps = int((max_val - min_val) / step) + 1
                step_idx = self.rng.randint(0, num_steps - 1)
                sampled[param_name] = min_val + step_idx * step

        return sampled

    def _prepare_trial_configs(self) -> list[TrialConfig]:
        """Prepare configurations for all trials."""
        configs = []

        for trial_id in range(self.num_trials):
            # Diversity: same seed for all trials (same init level, diversity from LLM)
            # Quality/Controllability: systematic seeds
            if self.experiment_type == "diversity":
                trial_seed = self.base_seed
            else:
                trial_seed = self.base_seed + trial_id

            # Determine target overrides for controllability
            target_overrides = None
            if self.experiment_type == "controllability":
                target_overrides = self._sample_control_params()

            configs.append(TrialConfig(
                trial_id=trial_id,
                config_path=self.config_path,
                run_dir=self.run_dir,
                max_steps=self.max_steps,
                seed=trial_seed,
                target_overrides=target_overrides,
                verbose=self.verbose,
                debug=self.debug,
                model=self.model,
                provider=self.provider,
                base_url=self.base_url,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                render_stats=self.render_stats,
                render_sim=self.render_sim,
                edit_tools=self.edit_tools,
                init=self.init,
                max_tool_calls=self.max_tool_calls,
                epsilon=self.epsilon,
                sa_temp=self.sa_temp,
                sa_decay=self.sa_decay,
                conv_length=self.conv_length,
                conv_filter=self.conv_filter,
                change_percentage=self.change_percentage,
                smb_solver=self.smb_solver,
            ))

        return configs

    def run(self) -> dict:
        """Run all trials in parallel and aggregate results."""
        trial_configs = self._prepare_trial_configs()

        print(f"Starting {self.experiment_type} experiment")
        print(f"  Problem type: {self.problem_type}")
        print(f"  Num trials: {self.num_trials}")
        print(f"  Max workers: {self.max_workers}")
        print(f"  Sampling seed: {self.sampling_seed}")
        print(f"  Run dir: {self.run_dir}")

        if self.experiment_type == "controllability":
            print(f"\nControl parameter ranges:")
            for param, r in self.control_ranges.items():
                print(f"  {param}: [{r.min_value}, {r.max_value}] (step={r.step})")

        print("-" * 60)

        # Run trials in parallel with progress tracking
        completed = 0
        failed = 0

        # Initialize progress tracker (updates every 5 seconds)
        tracker = ProgressTracker(self.num_trials, update_interval=5.0)

        print(f"Submitting {self.num_trials} trials to {self.max_workers} workers...")
        print("Progress updates every 5 seconds (trial completions shown immediately)")
        print()

        tracker.start()

        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for tc in trial_configs:
                    future = executor.submit(run_single_trial, tc)
                    futures[future] = tc
                    tracker.trial_started()

                for future in as_completed(futures):
                    trial_config = futures[future]
                    try:
                        result = future.result()
                        self.trial_results.append(result)

                        if result["success"]:
                            completed += 1
                            tracker.trial_completed(success=True)
                            score = result["summary"].get("final_score", "N/A")
                            # Clear the progress line and print completion message
                            print(f"\r\033[K✓ Trial {result['trial_id']+1}/{self.num_trials} complete "
                                  f"(score={score})")
                        else:
                            failed += 1
                            tracker.trial_completed(success=False)
                            error = result.get("error", result.get("stderr", "unknown"))
                            print(f"\r\033[K✗ Trial {result['trial_id']+1}/{self.num_trials} FAILED: {error[:100]}")

                    except Exception as e:
                        failed += 1
                        tracker.trial_completed(success=False)
                        print(f"\r\033[K✗ Trial {trial_config.trial_id+1}/{self.num_trials} EXCEPTION: {e}")
        finally:
            tracker.stop()

        print()
        print("-" * 60)
        print(f"Completed: {completed}/{self.num_trials}, Failed: {failed}")

        # Aggregate results
        summary = self._aggregate_results()

        # Save summary
        summary_path = self.run_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        self._print_summary(summary)

        return summary

    def _aggregate_results(self) -> dict:
        """Aggregate results across all trials."""
        successful_results = [r for r in self.trial_results if r["success"]]

        if not successful_results:
            return {
                "experiment_type": self.experiment_type,
                "total_trials": self.num_trials,
                "successful_trials": 0,
                "failed_trials": self.num_trials,
                "avg_score": 0.0,
                "trials": self.trial_results,
                "timestamp": datetime.now().isoformat(),
            }

        # Base aggregation
        final_scores = [r["summary"].get("final_score", 0) for r in successful_results]
        avg_final_score = sum(final_scores) / len(final_scores)

        summary = {
            "experiment_type": self.experiment_type,
            "total_trials": self.num_trials,
            "successful_trials": len(successful_results),
            "failed_trials": self.num_trials - len(successful_results),
            "avg_optimization_score": avg_final_score,
            "run_dir": str(self.run_dir),
            "trials": self.trial_results,
            "timestamp": datetime.now().isoformat(),
        }

        # Type-specific aggregation
        if self.experiment_type == "quality":
            summary.update(self._aggregate_quality(successful_results))
        elif self.experiment_type == "diversity":
            summary.update(self._aggregate_diversity(successful_results))
        elif self.experiment_type == "controllability":
            summary.update(self._aggregate_controllability(successful_results))

        return summary

    def _make_eval_env(self):
        """Create a PCG benchmark environment matching the factory's configuration.

        Uses LevelFactory so tile-based variants (e.g. smbtile) get the
        correct content space instead of the slice-based default (smb-v0).
        """
        from toolusepcg.factory import LevelFactory
        return LevelFactory(self.config).env

    def _aggregate_quality(self, results: list[dict]) -> dict:
        """Aggregate quality and diversity experiment results.

        Computes per-trial quality scores and cross-trial diversity in a
        single pass over the final levels.
        """
        # For quality, we need to compute quality scores using the PCG benchmark
        # This requires loading the final levels and evaluating them
        env = self._make_eval_env()
        quality_scores = []
        solvable_count = 0
        solvable_infos = []

        for result in results:
            trial_dir = Path(result["run_dir"])
            final_level_path = trial_dir / "final_level.txt"

            if final_level_path.exists():
                # Load and evaluate level
                from toolusepcg.level import Level
                with open(final_level_path) as f:
                    level = Level.from_string(f.read(), problem_type=self.problem_type)

                try:
                    info = env.info(level.data)
                    _, quality_score, _ = env.quality(info)
                    quality_scores.append(quality_score)
                    result["quality_score"] = quality_score

                    solvable = self._is_level_solvable(self.problem_type, info)
                    result["solvable"] = solvable
                    if solvable:
                        solvable_count += 1
                        solvable_infos.append(info)
                except Exception as e:
                    print(f"Warning: Failed to compute quality for trial {result['trial_id']}: {e}")

        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            pass_threshold = 1.0
            passed = sum(1 for q in quality_scores if q >= pass_threshold)
        else:
            avg_quality = 0.0
            passed = 0

        solvable_rate = solvable_count / len(results) if results else 0.0

        # Compute cross-trial diversity from solvable levels
        diversity_score = 0.0
        pairwise_stats = {}

        if len(solvable_infos) > 1:
            try:
                div_result = env.diversity(solvable_infos)
                diversity_score = float(div_result[0])
                if len(div_result) > 2:
                    diversity_matrix = div_result[2]
                    matrix = np.array(diversity_matrix)
                    triu_indices = np.triu_indices(matrix.shape[0], k=1)
                    pairwise_scores = matrix[triu_indices]
                    if len(pairwise_scores) > 0:
                        pairwise_stats = {
                            "min": float(np.min(pairwise_scores)),
                            "max": float(np.max(pairwise_scores)),
                            "avg": float(np.mean(pairwise_scores)),
                            "std": float(np.std(pairwise_scores)),
                        }
            except Exception as e:
                print(f"Warning: Failed to compute diversity: {e}")

        return {
            "avg_score": avg_quality,
            "quality_scores": quality_scores,
            "pass_rate": passed / len(results) if results else 0.0,
            "min_quality": min(quality_scores) if quality_scores else 0.0,
            "max_quality": max(quality_scores) if quality_scores else 0.0,
            "solvable_count": solvable_count,
            "solvable_rate": solvable_rate,
            "diversity_score": diversity_score,
            "pairwise_stats": pairwise_stats,
        }

    @staticmethod
    def _is_level_solvable(problem_type: str, info: dict) -> bool:
        """Check if a level is solvable based on problem-type-specific signals."""
        if problem_type == "binary":
            return bool(info.get("path", 0) > 0)
        elif problem_type == "binarydoor":
            return bool(info.get("door_path", 0) > 0)
        elif problem_type == "zelda":
            return bool(info.get("player_key", 0) > 0 and info.get("key_door", 0) > 0)
        elif problem_type == "sokoban":
            return bool(len(info.get("solution", [])) > 0)
        elif problem_type == "loderunner":
            gold = info.get("gold", 0)
            return bool(gold > 0 and info.get("collected_gold", 0) >= gold)
        elif problem_type == "smb":
            return bool(info.get("complete", 0.0) >= 1.0)
        # Unknown problem type: assume solvable
        return True

    def _aggregate_controllability(self, results: list[dict]) -> dict:
        """Aggregate controllability experiment results.

        Only solvable trials contribute to controllability scores. Unsolvable
        trials get controllability_score=0 and are excluded from avg_score /
        exact_match averages. The solvable count and rate are reported
        separately.
        """
        env = self._make_eval_env()
        ctrl_scores = []
        exact_matches = 0
        solvable_count = 0

        per_param_stats = {p: {"scores": [], "exact": 0} for p in self.control_ranges.keys()}

        for result in results:
            trial_dir = Path(result["run_dir"])
            final_level_path = trial_dir / "final_level.txt"
            control_params = result.get("target_overrides", {})

            if final_level_path.exists() and control_params:
                from toolusepcg.level import Level
                with open(final_level_path) as f:
                    level = Level.from_string(f.read(), problem_type=self.problem_type)

                try:
                    info = env.info(level.data)
                    _, ctrl_score, _ = env.controlability(info, control_params)
                    solvable = self._is_level_solvable(self.problem_type, info)
                    result["solvable"] = solvable

                    if solvable:
                        solvable_count += 1
                        ctrl_scores.append(ctrl_score)
                        result["controllability_score"] = ctrl_score

                        # Check exact match and per-param stats
                        final_metrics = result["summary"].get("final_metrics", {})
                        all_match = True

                        for param, target_val in control_params.items():
                            achieved_val = final_metrics.get(param, 0)

                            # Per-param score
                            if target_val > 0:
                                param_score = 1.0 - min(1.0, abs(achieved_val - target_val) / target_val)
                            else:
                                param_score = 1.0 if achieved_val == 0 else 0.0

                            per_param_stats[param]["scores"].append(param_score)

                            if int(achieved_val) == int(target_val):
                                per_param_stats[param]["exact"] += 1
                            else:
                                all_match = False

                        if all_match:
                            exact_matches += 1
                            result["exact_match"] = True
                        else:
                            result["exact_match"] = False
                    else:
                        # Unsolvable: exclude from controllability averages
                        result["controllability_score"] = 0.0
                        result["exact_match"] = False

                except Exception as e:
                    print(f"Warning: Failed to compute controllability for trial {result['trial_id']}: {e}")

        # Compute averages (only over solvable trials)
        avg_ctrl = sum(ctrl_scores) / len(ctrl_scores) if ctrl_scores else 0.0
        exact_rate = exact_matches / len(results) if results else 0.0
        solvable_rate = solvable_count / len(results) if results else 0.0

        per_param_summary = {}
        for param, stats in per_param_stats.items():
            per_param_summary[param] = {
                "avg_score": sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0,
                "exact_matches": stats["exact"],
            }

        return {
            "avg_score": avg_ctrl,
            "controllability_scores": ctrl_scores,
            "exact_match_rate": exact_rate,
            "exact_matches": exact_matches,
            "solvable_count": solvable_count,
            "solvable_rate": solvable_rate,
            "per_param_stats": per_param_summary,
        }

    def _aggregate_diversity(self, results: list[dict]) -> dict:
        """Aggregate diversity experiment results (cross-trial evaluation).

        All trials use the same seed so they start from the same initial level.
        Diversity measures how different the LLM's outputs are from identical
        starting conditions. Only solvable levels are included.
        """
        env = self._make_eval_env()
        infos = []
        solvable_count = 0
        total_levels = 0

        for result in results:
            trial_dir = Path(result["run_dir"])
            final_level_path = trial_dir / "final_level.txt"

            if final_level_path.exists():
                from toolusepcg.level import Level
                with open(final_level_path) as f:
                    level = Level.from_string(f.read(), problem_type=self.problem_type)

                try:
                    info = env.info(level.data)
                    total_levels += 1
                    solvable = self._is_level_solvable(self.problem_type, info)
                    result["solvable"] = solvable

                    if solvable:
                        solvable_count += 1
                        infos.append(info)
                except Exception as e:
                    print(f"Warning: Failed to get info for trial {result['trial_id']}: {e}")

        # Compute diversity across solvable levels
        diversity_score = 0.0
        pairwise_stats = {}

        if len(infos) > 1:
            try:
                div_result = env.diversity(infos)
                diversity_score = float(div_result[0])
                if len(div_result) > 2:
                    diversity_matrix = div_result[2]
                    matrix = np.array(diversity_matrix)
                    triu_indices = np.triu_indices(matrix.shape[0], k=1)
                    pairwise_scores = matrix[triu_indices]
                    if len(pairwise_scores) > 0:
                        pairwise_stats = {
                            "min": float(np.min(pairwise_scores)),
                            "max": float(np.max(pairwise_scores)),
                            "avg": float(np.mean(pairwise_scores)),
                            "std": float(np.std(pairwise_scores)),
                        }
            except Exception as e:
                print(f"Warning: Failed to compute diversity: {e}")

        threshold = 0.4
        passed = diversity_score >= threshold

        solvable_rate = solvable_count / len(results) if results else 0.0

        return {
            "avg_score": diversity_score,
            "diversity_score": diversity_score,
            "threshold": threshold,
            "passed": passed,
            "num_levels": len(infos),
            "total_levels": total_levels,
            "solvable_count": solvable_count,
            "solvable_rate": solvable_rate,
            "pairwise_stats": pairwise_stats,
        }

    def _print_summary(self, summary: dict) -> None:
        """Print experiment summary."""
        print("\n" + "=" * 60)
        print(f"{self.experiment_type.upper()} EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Total trials: {summary['total_trials']}")
        print(f"Successful: {summary['successful_trials']}")
        print(f"Failed: {summary['failed_trials']}")
        print(f"Average score: {summary.get('avg_score', 0):.3f}")

        if self.experiment_type == "quality":
            print(f"Pass rate: {summary.get('pass_rate', 0):.1%}")
            print(f"Min quality: {summary.get('min_quality', 0):.3f}")
            print(f"Max quality: {summary.get('max_quality', 0):.3f}")
            print(f"Diversity score: {summary.get('diversity_score', 0):.3f}")
            if "pairwise_stats" in summary and summary["pairwise_stats"]:
                ps = summary["pairwise_stats"]
                print(f"Pairwise diversity: min={ps.get('min', 0):.3f}, max={ps.get('max', 0):.3f}, avg={ps.get('avg', 0):.3f}")

        elif self.experiment_type == "diversity":
            print(f"Diversity score: {summary.get('diversity_score', 0):.3f}")
            print(f"Threshold: {summary.get('threshold', 0):.3f}")
            print(f"Passed: {'Yes' if summary.get('passed', False) else 'No'}")
            print(f"Solvable: {summary.get('solvable_count', 0)}/{summary.get('total_levels', 0)}")
            if "pairwise_stats" in summary and summary["pairwise_stats"]:
                ps = summary["pairwise_stats"]
                print(f"Pairwise diversity: min={ps.get('min', 0):.3f}, max={ps.get('max', 0):.3f}, avg={ps.get('avg', 0):.3f}")

        elif self.experiment_type == "controllability":
            print(f"Exact match rate: {summary.get('exact_match_rate', 0):.1%}")
            if "per_param_stats" in summary:
                print("\nPer-parameter statistics:")
                for param, stats in summary["per_param_stats"].items():
                    print(f"  {param}:")
                    print(f"    Avg score: {stats.get('avg_score', 0):.3f}")
                    print(f"    Exact matches: {stats.get('exact_matches', 0)}/{summary['successful_trials']}")

        print(f"\nResults saved to: {self.run_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel experiment trials for PCG optimization."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["quality", "diversity", "controllability"],
        required=True,
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of trials to run (default: 10)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel workers (default: 4)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum optimization steps per trial (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--experiment-seed",
        type=int,
        default=None,
        help="Seed for experiment sampling (default: same as --seed)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output directory for experiment (default: auto-generated)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output for each trial",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Disable verbose step output (overrides --verbose)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: save full LLM prompts/responses to debug.log in each trial directory",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["portkey", "openai", "google"],
        default=None,
        help="LLM provider: 'portkey' (default), 'openai' (for vLLM, TGI, etc.), or 'google' (Google Genai SDK)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (overrides config)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="LLM API base URL (overrides config, e.g., http://localhost:8000/v1 for vLLM)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API key (overrides config; for vLLM, defaults to 'EMPTY' if not set)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max completion tokens (overrides config; lower for vLLM where max_tokens + prompt must fit in context)",
    )
    parser.add_argument(
        "--render-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay metrics/targets on rendered level images (default: True, use --no-render-stats to disable)",
    )
    parser.add_argument(
        "--render-sim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render simulation GIF + trajectory for SMB, simulation GIF for Zelda (default: True, use --no-render-sim to disable)",
    )
    parser.add_argument(
        "--edit-tools",
        nargs="*",
        default=None,
        metavar="TOOL",
        help="Edit tools to provide to the agent (e.g., place_single_tile place_line place_patch)",
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        help="Level initialization strategy: 'random' (default), 'empty' (all passable tiles), "
             "'filled' (all blocking tiles), 'weighted' (weighted random: common tiles favored, "
             "special tiles rare), or a file path to an ASCII level file",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="Max tool calls per optimization step (overrides config optimizer.max_tool_calls_per_step)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Probability of accepting a worse solution in greedy strategy (overrides config optimizer.epsilon_accept_worse)",
    )
    parser.add_argument(
        "--sa-temp",
        type=float,
        default=None,
        help="Simulated annealing initial temperature (enables SA strategy when set; e.g. 10.0)",
    )
    parser.add_argument(
        "--sa-decay",
        type=float,
        default=None,
        help="Simulated annealing decay rate per step (default: 0.95); only used when --sa-temp is set",
    )
    parser.add_argument(
        "--conv-length",
        type=int,
        default=None,
        help="Number of conversation turns to keep (1 = no history, default from config)",
    )
    parser.add_argument(
        "--conv-filter",
        type=str,
        choices=["all", "accepted"],
        default=None,
        help="Which steps to include in conversation history: 'all' or 'accepted' (default from config)",
    )
    parser.add_argument(
        "--change-percentage",
        type=float,
        default=None,
        help="Change budget multiplier for termination (budget = multiplier * height * width; overrides config termination.change_budget_multiplier)",
    )
    parser.add_argument(
        "--smb-solver",
        type=str,
        choices=["auto", "astar"],
        default=None,
        help="SMB simulation solver: 'auto' (default: heuristic then A*) or 'astar' (A* only)",
    )

    args = parser.parse_args()

    sampling_seed = args.experiment_seed if args.experiment_seed is not None else args.seed

    # --quiet overrides --verbose
    verbose = args.verbose and not args.quiet

    launcher = ExperimentLauncher(
        config_path=args.config,
        experiment_type=args.experiment,
        num_trials=args.num_trials,
        max_workers=args.max_workers,
        max_steps=args.max_steps,
        base_seed=args.seed,
        sampling_seed=sampling_seed,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        verbose=verbose,
        debug=args.debug,
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        render_stats=args.render_stats,
        render_sim=args.render_sim,
        edit_tools=args.edit_tools,
        init=args.init,
        max_tool_calls=args.max_tool_calls,
        epsilon=args.epsilon,
        sa_temp=args.sa_temp,
        sa_decay=args.sa_decay,
        conv_length=args.conv_length,
        conv_filter=args.conv_filter,
        change_percentage=args.change_percentage,
        smb_solver=args.smb_solver,
    )

    launcher.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
