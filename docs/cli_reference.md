# CLI Reference

## Common Flags (`main.py` and `run_experiment.py`)

| Flag | Description |
|------|-------------|
| `--config` | Path to configuration YAML file |
| `--seed` | Random seed (base seed for experiments) |
| `--max-steps` | Maximum optimization steps per run/trial |
| `--run-dir` | Output directory (default: auto-generated under `runs/`) |
| `--provider` | LLM provider: `portkey` (default), `openai` (for vLLM, TGI, etc.), or `google` |
| `--model` | LLM model override |
| `--base-url` | LLM API base URL override (e.g., `http://localhost:8000/v1` for vLLM) |
| `--api-key` | LLM API key override (defaults to `EMPTY` for vLLM). Also: `OPENAI_API_KEY`, `GOOGLE_API_KEY` env vars |
| `--verbose` / `-v` | Enable verbose step output |
| `--quiet` / `-q` | Disable verbose step output |
| `--debug` | Save full LLM prompts/responses to `debug.log` |
| `--render-stats` / `--no-render-stats` | Overlay metrics on rendered level images (default: on) |
| `--render-sim` / `--no-render-sim` | Render simulation GIF + trajectory (default: on) |
| `--edit-tools TOOL [TOOL ...]` | Override edit tools for the agent |
| `--init STRATEGY` | Level initialization: `random`, `empty`, `filled`, `weighted`, or file path |
| `--max-tool-calls N` | Max tool calls per optimization step |
| `--epsilon F` | Probability of accepting a worse solution (greedy strategy) |
| `--sa-temp T` | Simulated annealing initial temperature (sets `strategy: annealing`) |
| `--sa-decay R` | Simulated annealing decay rate per step (default 0.95) |
| `--conv-length N` | Conversation history length (1 = no history) |
| `--conv-filter {all,accepted}` | Which steps to include in history |
| `--change-percentage F` | Change budget multiplier (budget = multiplier × height × width) |
| `--smb-solver {auto,astar}` | SMB simulation solver |

## `main.py`-only Flags

| Flag | Description |
|------|-------------|
| `--target METRIC VALUE` | Override target metric (repeatable) |
| `--extra-instruction TEXT` | Extra instruction appended to LLM prompt |
| `--init` | Level initialization strategy or file path |
| `--exp-name` | Optional experiment name for output directory |
| `--resume-from PATH` | Resume from a previous run directory |
| `--resume-step N` | Step to resume from (default: last step) |

## `run_experiment.py`-only Flags

| Flag | Description |
|------|-------------|
| `--experiment` | Experiment type: `quality`, `controllability`, `diversity` |
| `--num-trials` | Number of trials (default: 10) |
| `--max-workers` | Maximum parallel workers (default: 4) |
| `--experiment-seed` | Seed for experiment sampling |

## `run_sweep.py`-only Flags

| Flag | Description |
|------|-------------|
| `--config` | Path to sweep configuration YAML (required) |
| `--num-trials` | Override trials per experiment |
| `--max-workers` | Override parallel workers |
| `--max-steps` | Override max steps per trial |
| `--seed` | Override base seed |
| `--run-dir` | Override sweep output directory (also for resuming) |
| `--name` | Custom name for the sweep folder |
| `--dry-run` | Print all combinations without executing |
| `--verbose` / `-v` | Pass verbose flag to experiments |
| `--debug` | Pass debug flag to experiments |
