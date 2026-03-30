#!/usr/bin/env python3
"""Sweep runner for comparing multiple models across problems and experiment types.

Orchestrates parallel execution of run_experiment.py across all combinations
of models, problem types, tool configurations, init strategies, tool call limits,
change percentages, conversation lengths, and experiment types defined in a sweep
config YAML. Automatically distributes max_workers across concurrent combinations
to maximize resource utilization.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class ModelSpec:
    """Specification for an LLM model."""
    name: str
    model: str
    max_tokens: int


@dataclass
class ProblemSpec:
    """Specification for a problem type."""
    name: str
    config: str


@dataclass
class ToolSpec:
    """Specification for an edit tool configuration."""
    name: str
    edit_tools: list[str] | None = None


@dataclass
class InitSpec:
    """Specification for a level initialization strategy."""
    name: str
    init: str  # "random", "empty", "filled", "weighted"


@dataclass
class ToolCallsSpec:
    """Specification for max tool calls per step."""
    name: str
    max_tool_calls: int


@dataclass
class ChangePctSpec:
    """Specification for a change budget multiplier."""
    name: str
    change_percentage: float


@dataclass
class AnnealingSpec:
    """Specification for the optimizer acceptance strategy."""
    name: str
    strategy: str = "greedy"
    epsilon_accept_worse: float | None = None
    annealing_initial_temp: float | None = None
    annealing_decay_rate: float | None = None


@dataclass
class ConvLengthSpec:
    """Specification for conversation history length."""
    name: str
    conv_length: int


@dataclass
class SweepConfig:
    """Full sweep configuration."""
    provider: str
    num_trials: int
    max_steps: int
    max_workers: int
    seed: int
    render_stats: bool
    render_sim: bool
    debug: bool
    base_url: str | None
    api_key: str | None
    conv_filter: str | None
    smb_solver: str | None
    experiment_seed: int | None
    models: list[ModelSpec]
    problems: list[ProblemSpec]
    tools: list[ToolSpec]
    inits: list[InitSpec]
    tool_calls: list[ToolCallsSpec]
    change_percentages: list[ChangePctSpec]
    annealings: list[AnnealingSpec]
    conv_lengths: list[ConvLengthSpec]
    experiments: list[str]


class SweepRunner:
    """Runs model x problem x tools x init x tool_calls x change_pct x annealing x conv_length x experiment combinations with auto-parallelism.

    Distributes max_workers across concurrent combinations. With num_trials=2
    and max_workers=10, runs 5 combinations concurrently (each with 2 workers)
    instead of 1 at a time wasting 8 worker slots.
    """

    VALID_EXPERIMENTS = {"quality", "diversity", "controllability"}

    def __init__(self, config_path: str, cli_overrides: dict):
        self.config_path = config_path
        self.cli_overrides = cli_overrides
        self.sweep_config = self._load_config(config_path)
        self.combinations = self._generate_combinations()

        # Apply CLI overrides
        if cli_overrides.get("num_trials") is not None:
            self.sweep_config.num_trials = cli_overrides["num_trials"]
        if cli_overrides.get("max_workers") is not None:
            self.sweep_config.max_workers = cli_overrides["max_workers"]
        if cli_overrides.get("max_steps") is not None:
            self.sweep_config.max_steps = cli_overrides["max_steps"]
        if cli_overrides.get("seed") is not None:
            self.sweep_config.seed = cli_overrides["seed"]

        # Set up sweep directory
        if cli_overrides.get("run_dir"):
            self.sweep_dir = Path(cli_overrides["run_dir"])
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = cli_overrides.get("name") or Path(config_path).stem
            self.sweep_dir = Path("runs") / f"{timestamp}_{suffix}"

        self.sweep_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, path: str) -> SweepConfig:
        """Load and validate sweep configuration from YAML."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        defaults = raw.get("defaults", {})

        # Parse models
        models = []
        for m in raw.get("models", []):
            models.append(ModelSpec(
                name=m["name"],
                model=m["model"],
                max_tokens=m.get("max_tokens", 65536),
            ))

        # Parse problems
        problems = []
        for p in raw.get("problems", []):
            problems.append(ProblemSpec(
                name=p["name"],
                config=p["config"],
            ))

        # Parse tools (default to single "default" entry if omitted)
        tools = []
        for t in raw.get("tools", []):
            tools.append(ToolSpec(
                name=t["name"],
                edit_tools=t.get("edit_tools"),
            ))
        if not tools:
            tools = [ToolSpec(name="default", edit_tools=None)]

        # Parse inits (default to single "random" entry if omitted)
        inits = []
        for i in raw.get("inits", []):
            inits.append(InitSpec(
                name=i["name"],
                init=i.get("init", i["name"]),  # default init value to the name itself
            ))
        if not inits:
            inits = [InitSpec(name="random", init="random")]

        # Parse tool_calls (default to single "5" entry if omitted)
        tool_calls_list = []
        for tc in raw.get("tool_calls", []):
            tool_calls_list.append(ToolCallsSpec(
                name=tc["name"],
                max_tool_calls=tc["max_tool_calls"],
            ))
        if not tool_calls_list:
            tool_calls_list = [ToolCallsSpec(name="20", max_tool_calls=20)]

        # Parse change_percentages (default to single "2.0" entry if omitted)
        change_pct_list = []
        for cp in raw.get("change_percentages", []):
            change_pct_list.append(ChangePctSpec(
                name=cp["name"],
                change_percentage=cp.get("change_percentage", float(cp["name"])),
            ))
        if not change_pct_list:
            change_pct_list = [ChangePctSpec(name="2.0", change_percentage=2.0)]

        # Parse annealings (default to single "greedy" entry if omitted)
        annealings_list = []
        for a in raw.get("annealings", []):
            annealings_list.append(AnnealingSpec(
                name=a["name"],
                strategy=a.get("strategy", "greedy"),
                epsilon_accept_worse=a.get("epsilon_accept_worse"),
                annealing_initial_temp=a.get("annealing_initial_temp"),
                annealing_decay_rate=a.get("annealing_decay_rate"),
            ))
        if not annealings_list:
            annealings_list = [AnnealingSpec(name="greedy", strategy="greedy")]

        # Parse conv_lengths (default to single entry from defaults.conv_length if omitted)
        conv_lengths_list = []
        for cl in raw.get("conv_lengths", []):
            conv_lengths_list.append(ConvLengthSpec(
                name=cl["name"],
                conv_length=cl.get("conv_length", int(cl["name"])),
            ))
        if not conv_lengths_list:
            default_cl = defaults.get("conv_length")
            if default_cl is not None:
                conv_lengths_list = [ConvLengthSpec(name=str(default_cl), conv_length=default_cl)]
            else:
                conv_lengths_list = [ConvLengthSpec(name="1", conv_length=1)]

        # Parse experiments
        experiments = raw.get("experiments", [])

        config = SweepConfig(
            provider=defaults.get("provider", "portkey"),
            num_trials=defaults.get("num_trials", 10),
            max_steps=defaults.get("max_steps", 100),
            max_workers=defaults.get("max_workers", 4),
            seed=defaults.get("seed", 42),
            render_stats=defaults.get("render_stats", True),
            render_sim=defaults.get("render_sim", True),
            debug=defaults.get("debug", False),
            base_url=defaults.get("base_url"),
            api_key=defaults.get("api_key"),
            conv_filter=defaults.get("conv_filter"),
            smb_solver=defaults.get("smb_solver"),
            experiment_seed=defaults.get("experiment_seed"),
            models=models,
            problems=problems,
            tools=tools,
            inits=inits,
            tool_calls=tool_calls_list,
            change_percentages=change_pct_list,
            annealings=annealings_list,
            conv_lengths=conv_lengths_list,
            experiments=experiments,
        )

        # Validate
        self._validate_config(config)
        return config

    def _validate_config(self, config: SweepConfig):
        """Validate sweep config upfront."""
        if not config.models:
            raise ValueError("No models defined in sweep config")
        if not config.problems:
            raise ValueError("No problems defined in sweep config")
        if not config.experiments:
            raise ValueError("No experiments defined in sweep config")

        for exp in config.experiments:
            if exp not in self.VALID_EXPERIMENTS:
                raise ValueError(
                    f"Invalid experiment type '{exp}'. "
                    f"Must be one of: {', '.join(sorted(self.VALID_EXPERIMENTS))}"
                )

        for problem in config.problems:
            config_path = Path(problem.config)
            if not config_path.exists():
                raise ValueError(
                    f"Config file not found for problem '{problem.name}': {problem.config}"
                )

        # Check for duplicate names within each dimension (would cause directory collisions)
        dimensions = {
            "models": [m.name for m in config.models],
            "problems": [p.name for p in config.problems],
            "tools": [t.name for t in config.tools],
            "inits": [i.name for i in config.inits],
            "tool_calls": [tc.name for tc in config.tool_calls],
            "change_percentages": [cp.name for cp in config.change_percentages],
            "annealings": [a.name for a in config.annealings],
            "conv_lengths": [cl.name for cl in config.conv_lengths],
        }
        for dim_name, names in dimensions.items():
            seen = {}
            for idx, name in enumerate(names):
                if name in seen:
                    raise ValueError(
                        f"Duplicate name '{name}' in {dim_name} "
                        f"(entries {seen[name]} and {idx}). "
                        f"Each entry must have a unique name to avoid "
                        f"directory collisions."
                    )
                seen[name] = idx

    def _generate_combinations(self) -> list[tuple[ModelSpec, ProblemSpec, ToolSpec, InitSpec, ToolCallsSpec, ChangePctSpec, AnnealingSpec, ConvLengthSpec, str]]:
        """Generate all (model, problem, tools, init, tool_calls, change_pct, annealing, conv_length, experiment) combinations."""
        combos = []
        for model in self.sweep_config.models:
            for problem in self.sweep_config.problems:
                for tools in self.sweep_config.tools:
                    for init in self.sweep_config.inits:
                        for tc in self.sweep_config.tool_calls:
                            for cp in self.sweep_config.change_percentages:
                                for annealing in self.sweep_config.annealings:
                                    for cl in self.sweep_config.conv_lengths:
                                        for experiment in self.sweep_config.experiments:
                                            combos.append((model, problem, tools, init, tc, cp, annealing, cl, experiment))
        return combos

    def _combo_dir(self, model: ModelSpec, problem: ProblemSpec, tools: ToolSpec, init: InitSpec, tc: ToolCallsSpec, cp: ChangePctSpec, annealing: AnnealingSpec, cl: ConvLengthSpec, experiment: str) -> Path:
        """Get output directory for a specific combination."""
        return self.sweep_dir / f"{model.name}_{problem.name}_{tools.name}_{init.name}_tc{tc.name}_cp{cp.name}_{annealing.name}_cl{cl.name}_{experiment}"

    def _is_completed(self, combo_dir: Path) -> bool:
        """Check if a combination has already been completed (with a valid JSON summary)."""
        summary_path = combo_dir / "experiment_summary.json"
        if not summary_path.exists():
            return False
        try:
            with open(summary_path) as f:
                json.load(f)
            return True
        except json.JSONDecodeError:
            return False

    def _build_command(
        self,
        model: ModelSpec,
        problem: ProblemSpec,
        tools: ToolSpec,
        init: InitSpec,
        tc: ToolCallsSpec,
        cp: ChangePctSpec,
        annealing: AnnealingSpec,
        cl: ConvLengthSpec,
        experiment: str,
        combo_dir: Path,
        workers_per_combo: int | None = None,
    ) -> list[str]:
        """Build the subprocess command for run_experiment.py."""
        workers = workers_per_combo if workers_per_combo is not None else self.sweep_config.max_workers
        cmd = [
            sys.executable, "run_experiment.py",
            "--config", problem.config,
            "--experiment", experiment,
            "--model", model.model,
            "--provider", self.sweep_config.provider,
            "--max-tokens", str(model.max_tokens),
            "--num-trials", str(self.sweep_config.num_trials),
            "--max-workers", str(workers),
            "--max-steps", str(self.sweep_config.max_steps),
            "--seed", str(self.sweep_config.seed),
            "--run-dir", str(combo_dir),
            "--quiet",
            "--max-tool-calls", str(tc.max_tool_calls),
            "--change-percentage", str(cp.change_percentage),
        ]

        if self.sweep_config.base_url is not None:
            cmd.extend(["--base-url", self.sweep_config.base_url])

        if self.sweep_config.api_key is not None:
            cmd.extend(["--api-key", self.sweep_config.api_key])

        if self.sweep_config.experiment_seed is not None:
            cmd.extend(["--experiment-seed", str(self.sweep_config.experiment_seed)])

        if not self.sweep_config.render_stats:
            cmd.append("--no-render-stats")

        if self.sweep_config.render_sim:
            cmd.append("--render-sim")

        if init.init != "random":
            cmd.extend(["--init", init.init])

        if tools.edit_tools is not None:
            cmd.extend(["--edit-tools"] + tools.edit_tools)

        if annealing.epsilon_accept_worse is not None:
            cmd.extend(["--epsilon", str(annealing.epsilon_accept_worse)])

        if annealing.strategy == "annealing":
            if annealing.annealing_initial_temp is not None:
                cmd.extend(["--sa-temp", str(annealing.annealing_initial_temp)])
            if annealing.annealing_decay_rate is not None:
                cmd.extend(["--sa-decay", str(annealing.annealing_decay_rate)])

        cmd.extend(["--conv-length", str(cl.conv_length)])

        if self.sweep_config.conv_filter is not None:
            cmd.extend(["--conv-filter", self.sweep_config.conv_filter])

        if self.sweep_config.smb_solver is not None:
            cmd.extend(["--smb-solver", self.sweep_config.smb_solver])

        if self.cli_overrides.get("verbose"):
            # Replace --quiet with --verbose
            cmd.remove("--quiet")
            cmd.append("--verbose")

        if self.cli_overrides.get("debug") or self.sweep_config.debug:
            cmd.append("--debug")

        return cmd

    def _run_combination(
        self,
        model: ModelSpec,
        problem: ProblemSpec,
        tools: ToolSpec,
        init: InitSpec,
        tc: ToolCallsSpec,
        cp: ChangePctSpec,
        annealing: AnnealingSpec,
        cl: ConvLengthSpec,
        experiment: str,
        combo_dir: Path,
        workers_per_combo: int | None = None,
    ) -> dict:
        """Run a single combination and return its result."""
        cmd = self._build_command(model, problem, tools, init, tc, cp, annealing, cl, experiment, combo_dir, workers_per_combo)

        # Timeout: max_steps * 150s per step + 30min buffer
        timeout_seconds = self.sweep_config.max_steps * 150 + 1800

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            elapsed = time.time() - start_time

            # Check for experiment_summary.json
            summary_path = combo_dir / "experiment_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                return {
                    "success": True,
                    "model": model.name,
                    "problem": problem.name,
                    "tools": tools.name,
                    "init": init.name,
                    "tool_calls": tc.name,
                    "change_percentage": cp.name,
                    "annealing": annealing.name,
                    "conv_length": cl.name,
                    "experiment": experiment,
                    "summary": summary,
                    "elapsed_seconds": elapsed,
                    "returncode": result.returncode,
                }
            else:
                return {
                    "success": False,
                    "model": model.name,
                    "problem": problem.name,
                    "tools": tools.name,
                    "init": init.name,
                    "tool_calls": tc.name,
                    "change_percentage": cp.name,
                    "annealing": annealing.name,
                    "conv_length": cl.name,
                    "experiment": experiment,
                    "error": "experiment_summary.json not found after run",
                    "elapsed_seconds": elapsed,
                    "returncode": result.returncode,
                    "stderr": result.stderr[-500:] if result.stderr else "",
                }

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "model": model.name,
                "problem": problem.name,
                "tools": tools.name,
                "init": init.name,
                "tool_calls": tc.name,
                "change_percentage": cp.name,
                "annealing": annealing.name,
                "conv_length": cl.name,
                "experiment": experiment,
                "error": f"timeout after {timeout_seconds}s",
                "elapsed_seconds": elapsed,
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "model": model.name,
                "problem": problem.name,
                "tools": tools.name,
                "init": init.name,
                "tool_calls": tc.name,
                "change_percentage": cp.name,
                "annealing": annealing.name,
                "conv_length": cl.name,
                "experiment": experiment,
                "error": str(e),
                "elapsed_seconds": elapsed,
            }

    def _collect_results(self) -> dict:
        """Collect results from all completed combination directories."""
        results = {}
        for model, problem, tools, init, tc, cp, annealing, cl, experiment in self.combinations:
            combo_dir = self._combo_dir(model, problem, tools, init, tc, cp, annealing, cl, experiment)
            summary_path = combo_dir / "experiment_summary.json"
            key = f"{model.name}_{problem.name}_{tools.name}_{init.name}_tc{tc.name}_cp{cp.name}_{annealing.name}_cl{cl.name}_{experiment}"
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        results[key] = {
                            "success": True,
                            "model": model.name,
                            "problem": problem.name,
                            "tools": tools.name,
                            "init": init.name,
                            "tool_calls": tc.name,
                            "change_percentage": cp.name,
                            "annealing": annealing.name,
                            "conv_length": cl.name,
                            "experiment": experiment,
                            "summary": json.load(f),
                        }
                except json.JSONDecodeError:
                    pass  # Truncated file from a previous crashed run; treat as missing
        return results

    def _generate_markdown(self, results: dict, failures: list[dict]) -> str:
        """Generate markdown comparison tables from results.

        Tables use model-rows x problem-columns layout. When there are multiple
        tool configs, init strategies, tool_calls, or annealing configs, each
        experiment section has sub-tables for each combination.
        """
        sc = self.sweep_config
        model_names = [m.name for m in sc.models]
        problem_names = [p.name for p in sc.problems]
        tool_names = [t.name for t in sc.tools]
        init_names = [i.name for i in sc.inits]
        tc_names = [tc.name for tc in sc.tool_calls]
        cp_names = [cp.name for cp in sc.change_percentages]
        annealing_names = [a.name for a in sc.annealings]
        cl_names = [cl.name for cl in sc.conv_lengths]

        lines = []
        lines.append("# Sweep Results")
        lines.append("")
        lines.append(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Trials per experiment**: {sc.num_trials}")
        lines.append(f"- **Max steps**: {sc.max_steps}")
        lines.append(f"- **Seed**: {sc.seed}")
        if len(tool_names) > 1:
            lines.append(f"- **Tool configs**: {', '.join(tool_names)}")
        if len(init_names) > 1:
            lines.append(f"- **Init strategies**: {', '.join(init_names)}")
        if len(tc_names) > 1:
            lines.append(f"- **Tool calls/step**: {', '.join(tc_names)}")
        if len(cp_names) > 1:
            lines.append(f"- **Change percentages**: {', '.join(cp_names)}")
        if len(annealing_names) > 1:
            lines.append(f"- **Annealing configs**: {', '.join(annealing_names)}")
        if len(cl_names) > 1:
            lines.append(f"- **Conv lengths**: {', '.join(cl_names)}")
        lines.append("")

        single_tool = len(tool_names) == 1
        single_init = len(init_names) == 1
        single_tc = len(tc_names) == 1
        single_cp = len(cp_names) == 1
        single_anneal = len(annealing_names) == 1
        single_cl = len(cl_names) == 1

        # One section per experiment type
        for experiment in sc.experiments:
            lines.append(f"## {experiment.capitalize()}")
            lines.append("")

            for tname in tool_names:
                for iname in init_names:
                    for tcname in tc_names:
                        for cpname in cp_names:
                            for aname in annealing_names:
                                for clname in cl_names:
                                    # Show sub-heading when there are multiple tools, inits, tool_calls, change_percentages, annealings, or conv_lengths
                                    if not (single_tool and single_init and single_tc and single_cp and single_anneal and single_cl):
                                        heading_parts = []
                                        if not single_tool:
                                            heading_parts.append(f"Tools: {tname}")
                                        if not single_init:
                                            heading_parts.append(f"Init: {iname}")
                                        if not single_tc:
                                            heading_parts.append(f"ToolCalls: {tcname}")
                                        if not single_cp:
                                            heading_parts.append(f"ChangePct: {cpname}")
                                        if not single_anneal:
                                            heading_parts.append(f"Annealing: {aname}")
                                        if not single_cl:
                                            heading_parts.append(f"ConvLen: {clname}")
                                        lines.append(f"### {' / '.join(heading_parts)}")
                                        lines.append("")

                                    # Header row
                                    header = "| Model |"
                                    separator = "|-------|"
                                    for pname in problem_names:
                                        header += f" {pname} |"
                                        separator += "------|"
                                    header += " Avg |"
                                    separator += "------|"
                                    lines.append(header)
                                    lines.append(separator)

                                    # Data rows
                                    for mname in model_names:
                                        row = f"| {mname} |"
                                        values_for_avg = []

                                        for pname in problem_names:
                                            key = f"{mname}_{pname}_{tname}_{iname}_tc{tcname}_cp{cpname}_{aname}_cl{clname}_{experiment}"
                                            entry = results.get(key)

                                            if entry and entry.get("success"):
                                                summary = entry["summary"]
                                                cell = self._format_cell(experiment, summary)
                                                avg_val = self._extract_avg_value(experiment, summary)
                                                if avg_val is not None:
                                                    values_for_avg.append(avg_val)
                                            else:
                                                cell = "FAIL"

                                            row += f" {cell} |"

                                        # Avg column
                                        if values_for_avg:
                                            avg = sum(values_for_avg) / len(values_for_avg)
                                            row += f" {avg:.3f} |"
                                        else:
                                            row += " - |"

                                        lines.append(row)

                                    lines.append("")

        # Failures section
        if failures:
            lines.append("## Failures")
            lines.append("")
            for f in failures:
                parts = [f"**{f['model']}**", f['problem']]
                if not single_tool:
                    parts.append(f['tools'])
                if not single_init:
                    parts.append(f.get('init', 'random'))
                if not single_tc:
                    parts.append(f"tc{f.get('tool_calls', '20')}")
                if not single_cp:
                    parts.append(f"cp{f.get('change_percentage', '2.0')}")
                if not single_anneal:
                    parts.append(f.get('annealing', 'greedy'))
                if not single_cl:
                    parts.append(f"cl{f.get('conv_length', '1')}")
                parts.append(f['experiment'])
                lines.append(f"- {' / '.join(parts)}: {f.get('error', 'unknown')}")
            lines.append("")

        return "\n".join(lines)

    def _has_meaningful_optimization(self, summary: dict) -> bool:
        """Check if any trial actually ran optimization steps successfully.

        If every trial had 0 accepted and 0 rejected steps, all steps were
        errors (e.g. LLM API failures). The scores in that case just reflect
        the unoptimized initial level and are not meaningful for comparison.
        """
        for trial in summary.get("trials", []):
            trial_summary = trial.get("summary", {})
            accepted = trial_summary.get("accepted_steps", 0)
            rejected = trial_summary.get("rejected_steps", 0)
            if accepted + rejected > 0:
                return True
        return False

    def _format_cell(self, experiment: str, summary: dict) -> str:
        """Format a table cell based on experiment type."""
        if not self._has_meaningful_optimization(summary):
            return "nan"

        if experiment == "quality":
            avg_score = summary.get("avg_score", 0)
            successful = summary.get("successful_trials", 0)
            total = summary.get("total_trials", 0)
            solvable = summary.get("solvable_count", 0)
            return f"{avg_score:.3f} ({successful}/{total}, {solvable} solved)"

        elif experiment == "controllability":
            avg_score = summary.get("avg_score", 0)
            exact_rate = summary.get("exact_match_rate", 0)
            solvable = summary.get("solvable_count", 0)
            total = summary.get("total_trials", 0)
            return f"{avg_score:.3f} / {exact_rate:.0%} ({solvable}/{total} solved)"

        elif experiment == "diversity":
            diversity_score = summary.get("diversity_score", 0)
            solvable = summary.get("solvable_count", 0)
            total = summary.get("total_trials", 0)
            return f"{diversity_score:.3f} ({solvable}/{total} solved)"

        return "?"

    def _extract_avg_value(self, experiment: str, summary: dict) -> float | None:
        """Extract the primary numeric value for averaging across problems."""
        if not self._has_meaningful_optimization(summary):
            return None

        if experiment == "quality":
            return summary.get("avg_score")
        elif experiment == "controllability":
            return summary.get("avg_score")
        elif experiment == "diversity":
            return summary.get("diversity_score")
        return None

    def _save_resolved_config(self) -> None:
        """Save the effective sweep config after CLI overrides."""
        sc = self.sweep_config
        # Only include CLI overrides that were explicitly passed (not argparse defaults)
        explicit_overrides = {
            k: v for k, v in self.cli_overrides.items()
            if v is not None and v is not False
        }
        resolved = {
            "provider": sc.provider,
            "num_trials": sc.num_trials,
            "max_steps": sc.max_steps,
            "max_workers": sc.max_workers,
            "seed": sc.seed,
            "render_stats": sc.render_stats,
            "render_sim": sc.render_sim,
            "debug": sc.debug,
            "models": [
                {"name": m.name, "model": m.model, "max_tokens": m.max_tokens}
                for m in sc.models
            ],
            "problems": [
                {"name": p.name, "config": p.config}
                for p in sc.problems
            ],
            "tools": [
                {"name": t.name, **({"edit_tools": t.edit_tools} if t.edit_tools is not None else {})}
                for t in sc.tools
            ],
            "inits": [
                {"name": i.name, "init": i.init}
                for i in sc.inits
            ],
            "tool_calls": [
                {"name": tc.name, "max_tool_calls": tc.max_tool_calls}
                for tc in sc.tool_calls
            ],
            "change_percentages": [
                {"name": cp.name, "change_percentage": cp.change_percentage}
                for cp in sc.change_percentages
            ],
            "annealings": [
                {"name": a.name, "strategy": a.strategy,
                 **({"epsilon_accept_worse": a.epsilon_accept_worse} if a.epsilon_accept_worse is not None else {}),
                 **({"annealing_initial_temp": a.annealing_initial_temp} if a.annealing_initial_temp is not None else {}),
                 **({"annealing_decay_rate": a.annealing_decay_rate} if a.annealing_decay_rate is not None else {})}
                for a in sc.annealings
            ],
            "conv_lengths": [
                {"name": cl.name, "conv_length": cl.conv_length}
                for cl in sc.conv_lengths
            ],
            "experiments": sc.experiments,
            "base_url": sc.base_url,
            "api_key": sc.api_key,
            "conv_filter": sc.conv_filter,
            "smb_solver": sc.smb_solver,
            "experiment_seed": sc.experiment_seed,
        }
        if explicit_overrides:
            resolved["cli_overrides"] = explicit_overrides
        with open(self.sweep_dir / "sweep_config.yaml", "w") as f:
            yaml.dump(resolved, f, default_flow_style=False)

    def dry_run(self):
        """Print all combinations without executing."""
        sc = self.sweep_config
        concurrent_combos, workers_per_combo = self._compute_parallelism()
        print(f"Sweep config: {self.config_path}")
        print(f"Models: {len(sc.models)}")
        print(f"Problems: {len(sc.problems)}")
        print(f"Tools: {len(sc.tools)}")
        print(f"Inits: {len(sc.inits)}")
        print(f"Tool calls: {len(sc.tool_calls)}")
        print(f"Change pct: {len(sc.change_percentages)}")
        print(f"Annealings: {len(sc.annealings)}")
        print(f"Conv lens:  {len(sc.conv_lengths)}")
        print(f"Experiments: {len(sc.experiments)}")
        print(f"Total combinations: {len(self.combinations)}")
        print(f"Trials per experiment: {sc.num_trials}")
        print(f"Max steps: {sc.max_steps}")
        print(f"Max workers: {sc.max_workers}")
        print(f"Parallelism: {concurrent_combos} concurrent combos × {workers_per_combo} workers each")
        print(f"Seed: {sc.seed}")
        print(f"Output dir: {self.sweep_dir}")
        print()

        for i, (model, problem, tools, init, tc, cp, annealing, cl, experiment) in enumerate(self.combinations, 1):
            combo_dir = self._combo_dir(model, problem, tools, init, tc, cp, annealing, cl, experiment)
            completed = self._is_completed(combo_dir)
            status = " [DONE]" if completed else ""
            print(f"  {i:3d}. {model.name:20s} x {problem.name:12s} x {tools.name:12s} x {init.name:10s} x tc{tc.name:5s} x cp{cp.name:5s} x {annealing.name:10s} x cl{cl.name:4s} x {experiment:18s}{status}")

        # Count already completed
        done = sum(
            1 for m, p, t, i, tc, cp, a, cl, e in self.combinations
            if self._is_completed(self._combo_dir(m, p, t, i, tc, cp, a, cl, e))
        )
        if done > 0:
            print(f"\n{done}/{len(self.combinations)} already completed (will be skipped)")

    def _compute_parallelism(self) -> tuple[int, int]:
        """Compute how many combinations to run concurrently and workers per combo.

        Distributes max_workers across concurrent combinations so that
        total concurrent trials ≈ max_workers. For example:
        - num_trials=2, max_workers=10 → 5 concurrent combos × 2 workers each
        - num_trials=10, max_workers=10 → 1 concurrent combo × 10 workers each
        - num_trials=1, max_workers=10 → 10 concurrent combos × 1 worker each
        """
        sc = self.sweep_config
        workers_per_combo = min(sc.num_trials, sc.max_workers)
        concurrent_combos = max(1, sc.max_workers // workers_per_combo)
        return concurrent_combos, workers_per_combo

    def run(self):
        """Run all combinations with auto-parallelism and resumability."""
        sc = self.sweep_config
        total = len(self.combinations)
        concurrent_combos, workers_per_combo = self._compute_parallelism()

        print("=" * 70)
        print("SWEEP RUNNER")
        print("=" * 70)
        print(f"Models:      {', '.join(m.name for m in sc.models)}")
        print(f"Problems:    {', '.join(p.name for p in sc.problems)}")
        print(f"Tools:       {', '.join(t.name for t in sc.tools)}")
        print(f"Inits:       {', '.join(i.name for i in sc.inits)}")
        print(f"Tool calls:  {', '.join(tc.name for tc in sc.tool_calls)}")
        print(f"Change pct:  {', '.join(cp.name for cp in sc.change_percentages)}")
        print(f"Conv lens:   {', '.join(cl.name for cl in sc.conv_lengths)}")
        print(f"Experiments: {', '.join(sc.experiments)}")
        print(f"Total:       {total} combinations")
        print(f"Trials:      {sc.num_trials} per experiment")
        print(f"Max steps:   {sc.max_steps}")
        print(f"Parallelism: {concurrent_combos} concurrent combos × {workers_per_combo} workers each (max_workers={sc.max_workers})")
        print(f"Output:      {self.sweep_dir}")
        print("=" * 70)

        # Save resolved sweep config (after CLI overrides) so we know what actually ran
        self._save_resolved_config()

        # Track results and failures
        all_results = {}
        failures = []
        skipped = 0
        sweep_start = time.time()
        print_lock = threading.Lock()

        # Separate completed from pending combinations
        pending_combos = []
        for model, problem, tools, init, tc, cp, annealing, cl, experiment in self.combinations:
            combo_dir = self._combo_dir(model, problem, tools, init, tc, cp, annealing, cl, experiment)
            if self._is_completed(combo_dir):
                skipped += 1
                print(f"SKIP {model.name} x {problem.name} x {tools.name} x {init.name} x tc{tc.name} x cp{cp.name} x {annealing.name} x cl{cl.name} x {experiment} (already completed)")
                summary_path = combo_dir / "experiment_summary.json"
                with open(summary_path) as f:
                    summary = json.load(f)
                key = f"{model.name}_{problem.name}_{tools.name}_{init.name}_tc{tc.name}_cp{cp.name}_{annealing.name}_cl{cl.name}_{experiment}"
                all_results[key] = {
                    "success": True,
                    "model": model.name,
                    "problem": problem.name,
                    "tools": tools.name,
                    "init": init.name,
                    "tool_calls": tc.name,
                    "change_percentage": cp.name,
                    "annealing": annealing.name,
                    "conv_length": cl.name,
                    "experiment": experiment,
                    "summary": summary,
                }
            else:
                pending_combos.append((model, problem, tools, init, tc, cp, annealing, cl, experiment))

        if not pending_combos:
            print("\nAll combinations already completed.")
        else:
            print(f"\nRunning {len(pending_combos)} pending combinations "
                  f"({concurrent_combos} concurrent, {workers_per_combo} workers each)...")

            completed_count = skipped

            try:
                with ThreadPoolExecutor(max_workers=concurrent_combos) as executor:
                    futures = {}
                    for model, problem, tools, init, tc, cp, annealing, cl, experiment in pending_combos:
                        combo_dir = self._combo_dir(model, problem, tools, init, tc, cp, annealing, cl, experiment)
                        future = executor.submit(
                            self._run_combination,
                            model, problem, tools, init, tc, cp, annealing, cl, experiment, combo_dir,
                            workers_per_combo,
                        )
                        futures[future] = (model, problem, tools, init, tc, cp, annealing, cl, experiment)

                    for future in as_completed(futures):
                        model, problem, tools, init, tc, cp, annealing, cl, experiment = futures[future]
                        completed_count += 1

                        try:
                            result = future.result()
                        except Exception as e:
                            result = {
                                "success": False,
                                "model": model.name,
                                "problem": problem.name,
                                "tools": tools.name,
                                "init": init.name,
                                "tool_calls": tc.name,
                                "change_percentage": cp.name,
                                "annealing": annealing.name,
                                "conv_length": cl.name,
                                "experiment": experiment,
                                "error": str(e),
                            }

                        key = f"{model.name}_{problem.name}_{tools.name}_{init.name}_tc{tc.name}_cp{cp.name}_{annealing.name}_cl{cl.name}_{experiment}"
                        with print_lock:
                            if result["success"]:
                                all_results[key] = result
                                elapsed = result.get("elapsed_seconds", 0)
                                avg_score = result["summary"].get("avg_score", 0)
                                print(f"[{completed_count}/{total}] OK   {model.name} x {problem.name} x {tools.name} x {init.name} x tc{tc.name} x cp{cp.name} x {annealing.name} x cl{cl.name} x {experiment}"
                                      f"  avg_score={avg_score:.3f} ({elapsed:.0f}s)")
                            else:
                                failures.append(result)
                                error = result.get("error", "unknown")
                                print(f"[{completed_count}/{total}] FAIL {model.name} x {problem.name} x {tools.name} x {init.name} x tc{tc.name} x cp{cp.name} x {annealing.name} x cl{cl.name} x {experiment}"
                                      f": {error}")

            except KeyboardInterrupt:
                print("\n\nSweep interrupted by user. Generating partial results...")

        sweep_elapsed = time.time() - sweep_start

        # Collect any results we may have missed (from previous runs)
        all_results.update(self._collect_results())

        # Save machine-readable results
        sweep_results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config_path,
            "num_models": len(sc.models),
            "num_problems": len(sc.problems),
            "num_tools": len(sc.tools),
            "num_inits": len(sc.inits),
            "num_tool_calls": len(sc.tool_calls),
            "num_change_percentages": len(sc.change_percentages),
            "num_annealings": len(sc.annealings),
            "num_conv_lengths": len(sc.conv_lengths),
            "num_experiments": len(sc.experiments),
            "total_combinations": total,
            "completed": len(all_results),
            "failed": len(failures),
            "skipped": skipped,
            "elapsed_seconds": sweep_elapsed,
            "results": all_results,
            "failures": failures,
        }

        results_path = self.sweep_dir / "sweep_results.json"
        with open(results_path, "w") as f:
            json.dump(sweep_results, f, indent=2)

        # Generate markdown summary
        markdown = self._generate_markdown(all_results, failures)
        md_path = self.sweep_dir / "sweep_summary.md"
        with open(md_path, "w") as f:
            f.write(markdown)

        # Print final summary
        print()
        print("=" * 70)
        print("SWEEP COMPLETE")
        print("=" * 70)
        print(f"Completed: {len(all_results)}/{total}")
        print(f"Failed:    {len(failures)}")
        print(f"Skipped:   {skipped}")
        print(f"Elapsed:   {sweep_elapsed:.0f}s")
        print(f"Results:   {results_path}")
        print(f"Summary:   {md_path}")

        if failures:
            print(f"\nFailed combinations:")
            for f_entry in failures:
                print(f"  - {f_entry['model']} x {f_entry['problem']} x {f_entry['tools']} x {f_entry.get('init', 'random')} x tc{f_entry.get('tool_calls', '20')} x cp{f_entry.get('change_percentage', '2.0')} x {f_entry.get('annealing', 'greedy')} x cl{f_entry.get('conv_length', '1')} x {f_entry['experiment']}: "
                      f"{f_entry.get('error', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a sweep across multiple models, problems, and experiment types."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to sweep configuration YAML file",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Override number of trials per experiment",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override maximum parallel workers per experiment",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum optimization steps per trial",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override base random seed",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override sweep output directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom name for the sweep folder (e.g., --name my_exp → runs/{timestamp}_my_exp)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all combinations without executing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Pass verbose flag to experiments",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass debug flag to experiments",
    )

    args = parser.parse_args()

    cli_overrides = {
        "num_trials": args.num_trials,
        "max_workers": args.max_workers,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "run_dir": args.run_dir,
        "name": args.name,
        "verbose": args.verbose,
        "debug": args.debug,
    }

    try:
        runner = SweepRunner(args.config, cli_overrides)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        runner.dry_run()
    else:
        runner.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
