#!/usr/bin/env python3
"""Main entry point for single-run ToolUsePCG optimization.

For multi-trial experiments, use run_experiment.py instead.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from toolusepcg.config import load_config, config_from_dict, deep_merge
from toolusepcg.optimizer import GreedyOptimizer
from toolusepcg.logging import TraceLogger
from toolusepcg.resume import load_resume_state
from toolusepcg.skill_growth import SkillGrowthManager


def main():
    parser = argparse.ArgumentParser(
        description="Run single LLM agent optimization on a game level. "
                    "For multi-trial experiments, use run_experiment.py instead."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for level generation (overrides config)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum optimization steps (default: 100)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output directory for run (default: runs/{timestamp}_{problem_type})",
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
        "--init",
        type=str,
        default=None,
        help="Level initialization strategy: 'random' (default), 'empty' (all passable tiles), "
             "'filled' (all blocking tiles), 'weighted' (weighted random: common tiles favored, "
             "special tiles rare), or a file path to an ASCII level file",
    )
    parser.add_argument(
        "--initial-level",
        type=str,
        default=None,
        help="(Deprecated: use --init instead) Path to file with initial level (ASCII format)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Enable verbose step output (default: True)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Disable verbose step output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: save full LLM prompts/responses to debug.log and show raw responses in terminal",
    )
    parser.add_argument(
        "--target",
        nargs=2,
        action="append",
        metavar=("METRIC", "VALUE"),
        help="Override target metric value (can be repeated, e.g., --target player_key 15 --target key_door 12)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Optional experiment name for output directory naming",
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
        help="Render simulation GIF + trajectory image for SMB, simulation GIF for Zelda (default: True, use --no-render-sim to disable)",
    )
    parser.add_argument(
        "--edit-tools",
        nargs="*",
        default=None,
        metavar="TOOL",
        help="Edit tools to provide to the agent (e.g., place_single_tile place_line place_patch)",
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
    parser.add_argument(
        "--extra-instruction",
        type=str,
        default=None,
        help="Extra high-level instruction appended to the LLM prompt (e.g., 'make the layout look like the letter A')",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to previous run directory to resume from",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=None,
        help="Step to resume from (default: last step). Requires --resume-from",
    )

    args = parser.parse_args()

    # Validate resume args
    if args.resume_step is not None and args.resume_from is None:
        parser.error("--resume-step requires --resume-from")

    # Build config overrides from CLI arguments
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed

    # LLM overrides (provider, model, base_url, api_key)
    llm_overrides = {}
    if args.provider is not None:
        llm_overrides["provider"] = args.provider
    if args.model is not None:
        llm_overrides["model"] = args.model
    if args.base_url is not None:
        llm_overrides["base_url"] = args.base_url
    if args.api_key is not None:
        llm_overrides["api_key"] = args.api_key
    if args.max_tokens is not None:
        llm_overrides["max_tokens"] = args.max_tokens
    if llm_overrides:
        overrides["llm"] = llm_overrides

    # Handle target overrides from --target arguments
    if args.target:
        targets_overrides = {}
        for metric, value in args.target:
            try:
                parsed_value = int(value) if '.' not in value else float(value)
                targets_overrides[metric] = {"mode": "target", "value": parsed_value}
            except ValueError:
                print(f"Warning: Could not parse target value '{value}' for metric '{metric}'")

        if targets_overrides:
            overrides["targets"] = targets_overrides

    # Handle edit tools override
    if args.edit_tools is not None:
        overrides.setdefault("optimizer", {})["edit_tools"] = args.edit_tools

    # Handle max tool calls override
    if args.max_tool_calls is not None:
        overrides.setdefault("optimizer", {})["max_tool_calls_per_step"] = args.max_tool_calls

    # Handle epsilon accept worse override
    if args.epsilon is not None:
        overrides.setdefault("optimizer", {})["epsilon_accept_worse"] = args.epsilon

    # Handle simulated annealing overrides
    if args.sa_temp is not None:
        overrides.setdefault("optimizer", {})["strategy"] = "annealing"
        overrides.setdefault("optimizer", {})["annealing_initial_temp"] = args.sa_temp
    if args.sa_decay is not None:
        overrides.setdefault("optimizer", {})["annealing_decay_rate"] = args.sa_decay

    # Handle conversation history overrides
    if args.conv_length is not None:
        overrides.setdefault("optimizer", {})["conversation_length"] = args.conv_length
    if args.conv_filter is not None:
        overrides.setdefault("optimizer", {})["conversation_filter"] = args.conv_filter

    # Handle change percentage override
    if args.change_percentage is not None:
        overrides.setdefault("termination", {})["change_budget_multiplier"] = args.change_percentage

    # Handle SMB solver override
    if args.smb_solver is not None:
        overrides.setdefault("env", {})["smb_solver"] = args.smb_solver

    # Handle extra instruction override
    if args.extra_instruction is not None:
        overrides.setdefault("optimizer", {})["extra_instruction"] = args.extra_instruction

    # --- Resume or fresh config loading ---
    resume_state = None
    if args.resume_from:
        try:
            resume_state = load_resume_state(args.resume_from, args.resume_step)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading resume state: {e}")
            return 1
        print(f"Resuming from: {args.resume_from} (step {resume_state.resume_step})")

        # Build config from saved run, then apply CLI overrides on top
        config_dict = resume_state.config_dict
        if overrides:
            config_dict = deep_merge(config_dict, overrides)
        try:
            config = config_from_dict(config_dict)
        except Exception as e:
            print(f"Error loading config from resume state: {e}")
            return 1
    else:
        # Normal config loading
        try:
            config = load_config(args.config, overrides)
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1

    # Use seed from config if not specified on command line
    seed = args.seed if args.seed is not None else config.seed

    # Collect runtime args for saving alongside resolved config
    # Only include non-null, non-default values to keep output clean
    runtime_args = {
        "max_steps": args.max_steps,
    }
    if args.config:
        runtime_args["config_path"] = args.config
    if args.verbose and not args.quiet:
        runtime_args["verbose"] = True
    if args.debug:
        runtime_args["debug"] = True
    if args.render_stats is not True:
        runtime_args["render_stats"] = args.render_stats
    if not args.render_sim:
        runtime_args["render_sim"] = False
    # Resolve init strategy: --init takes precedence over --initial-level
    init_strategy = args.init or (args.initial_level if args.initial_level else "random")
    if init_strategy != "random":
        runtime_args["init"] = init_strategy
    if args.conv_length is not None:
        runtime_args["conv_length"] = args.conv_length
    if args.conv_filter is not None:
        runtime_args["conv_filter"] = args.conv_filter
    if args.extra_instruction:
        runtime_args["extra_instruction"] = args.extra_instruction
    if args.exp_name:
        runtime_args["exp_name"] = args.exp_name
    if resume_state is not None:
        runtime_args["resume_from"] = str(args.resume_from)
        runtime_args["resume_step"] = resume_state.resume_step

    # Initialize logger
    try:
        logger = TraceLogger(config, run_dir=args.run_dir, exp_name=args.exp_name,
                             render_stats=args.render_stats, render_sim=args.render_sim,
                             runtime_args=runtime_args)
    except Exception as e:
        print(f"Error initializing logger: {e}")
        return 1

    print(f"Run directory: {logger.get_run_dir()}")

    # Show target overrides if any were provided
    if args.target:
        print("\nTarget overrides:")
        for metric, value in args.target:
            print(f"  {metric}: {value}")

    # Initialize skill growth manager
    skill_manager = SkillGrowthManager(logger.get_run_dir())

    # Initialize optimizer
    verbose = args.verbose and not args.quiet
    try:
        optimizer = GreedyOptimizer(config, logger=logger, skill_manager=skill_manager, verbose=verbose, debug=args.debug)
    except Exception as e:
        print(f"Error initializing optimizer: {e}")
        return 1

    # Set environment for rendering
    logger.set_env(optimizer.factory.env)

    if resume_state is not None:
        # --- Resume path: use level from trace ---
        initial_level = optimizer.factory.create_from_string(resume_state.level_str)
        print(f"Resuming with level from step {resume_state.resume_step}")

        # Initialize optimizer with the resumed level
        try:
            optimizer.initialize(seed=seed, initial_level=initial_level)
        except Exception as e:
            print(f"Error during initialization: {e}")
            return 1

        # Restore state from resume point
        optimizer.step_count = resume_state.resume_step
        optimizer.termination.set_total_changes(resume_state.total_changes)

        # Re-evaluate to get fresh current_result and current_score
        optimizer.current_result = optimizer.evaluator.evaluate(optimizer.current_level)
        optimizer.current_score = optimizer.scorer.score(optimizer.current_result)

    else:
        # --- Normal path: resolve initial level from init strategy ---
        initial_level = None
        if init_strategy == "empty":
            initial_level = optimizer.factory.create_empty()
            print(f"Using empty initialization (all passable tiles)")
        elif init_strategy == "filled":
            initial_level = optimizer.factory.create_filled()
            print(f"Using filled initialization (all blocking tiles)")
        elif init_strategy == "weighted":
            initial_level = optimizer.factory.create_weighted_random(seed)
            print(f"Using weighted random initialization (common tiles favored)")
        elif init_strategy != "random":
            # Treat as file path
            initial_level_path = Path(init_strategy)
            if initial_level_path.exists():
                try:
                    with open(initial_level_path) as f:
                        initial_level = optimizer.factory.create_from_string(f.read())
                    print(f"Loaded initial level from: {initial_level_path}")
                except Exception as e:
                    print(f"Error loading initial level: {e}")
                    return 1
            else:
                print(f"Error: Initial level file not found: {initial_level_path}")
                return 1

        # Initialize optimizer
        print(f"Initializing with seed={seed}")
        try:
            optimizer.initialize(seed=seed, initial_level=initial_level)
        except Exception as e:
            print(f"Error during initialization: {e}")
            return 1

    # Show initial state
    print("\nInitial Level:")
    print(optimizer.current_level.to_string())
    print(f"\nInitial Score: {optimizer.current_score:.2f}")
    print(f"Initial Metrics: {optimizer.current_result.metrics}")
    print(f"\nChange Budget: {optimizer.termination.change_budget} tiles")
    if resume_state is not None:
        print(f"Resumed Changes: {resume_state.total_changes} tiles (from prior run)")
        print(f"Remaining Budget: {optimizer.termination.remaining_budget} tiles")

    # Run optimization
    remaining = args.max_steps - optimizer.step_count
    print(f"\nStarting optimization (max {args.max_steps} total steps, {remaining} remaining)...")
    print("-" * 50)

    try:
        summary = optimizer.run(max_steps=args.max_steps)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        total_input_tokens = sum(r.input_tokens for r in optimizer.step_records)
        total_output_tokens = sum(r.output_tokens for r in optimizer.step_records)
        summary = {
            "total_steps": optimizer.step_count,
            "final_score": optimizer.current_score,
            "final_metrics": optimizer.current_result.metrics if optimizer.current_result else {},
            "termination_status": optimizer.termination.status(),
            "interrupted": True,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
        logger.log_final_level(optimizer.current_level, optimizer.current_result)
        logger.log_summary(summary)
    except Exception as e:
        print(f"\nError during optimization: {e}")
        return 1

    # Show final state
    print("-" * 50)
    print("\nFinal Level:")
    print(optimizer.current_level.to_string())
    print(f"\nFinal Score: {summary['final_score']:.2f}")
    print(f"Final Metrics: {summary['final_metrics']}")
    print(f"\nTotal Steps: {summary['total_steps']}")
    print(f"Termination: {summary['termination_status']}")
    if summary.get('total_tokens', 0) > 0:
        print(f"Tokens: {summary['total_input_tokens']:,} input / {summary['total_output_tokens']:,} output / {summary['total_tokens']:,} total")

    # Show skill proposals if any
    skill_summary = skill_manager.summarize()
    if skill_summary["total_proposals"] > 0:
        print(f"\nSkills Proposed: {skill_summary['skills_proposed']}")

    print(f"\nResults saved to: {logger.get_run_dir()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
