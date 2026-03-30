# Adding a New Game/Domain

This guide covers all integration points needed to add a new game (referred to as `newgame` below) to ToolUsePCG.

## Prerequisites

Your game must have a problem class in the PCG benchmark at `submodules/pcg_benchmark/pcg_benchmark/probs/newgame/`. The benchmark class must implement the standard interface:

- `info(level_data) -> dict` — evaluate a level and return metrics
- `quality(info) -> float` — quality score from info dict
- `diversity(infos) -> float` — diversity across multiple info dicts
- `controlability(info, control) -> tuple` — controllability score

Only metrics supported by the benchmark's `controlability()` method should be made controllable.

## Integration Checklist

| # | File | What to add |
|---|------|-------------|
| 1 | `toolusepcg/level.py` | `NewgameTiles` enum, 4 mapping dicts, 2 dispatch elifs |
| 2 | `toolusepcg/factory.py` | 3 class dict entries, env registration elif, `get_problem_class` entry |
| 3 | `toolusepcg/config.py` | `_VALID_TARGETS_BY_PROBLEM`, `TargetsConfig` fields, `ScoringConfig` weights, `get_control_ranges`, `get_problem_type` |
| 4 | `toolusepcg/evaluator.py` | `_evaluate_newgame`, `_format_newgame_metrics`, 2 dispatch elifs |
| 5 | `toolusepcg/optimizer/scoring.py` | `_score_newgame`, `format_score` case, dispatch elif |
| 6 | `toolusepcg/optimizer/greedy.py` | `_build_newgame_prompt`, `_FEEDBACK_METRICS` entry, dispatch elif |
| 7 | `toolusepcg/tools/` | Tool registration in `_init_tools` (+ optional custom tools) |
| 8 | `toolusepcg/logging/trace.py` | `_STATS_METRIC_CONFIG` entry (+ optional simulation render) |
| 9 | `run_experiment.py` | `_is_level_solvable` elif |
| 10 | `configs/` | `newgame.yaml` config file |

## Step-by-Step

### Step 1: Tile Definitions — `toolusepcg/level.py`

Every game defines an `IntEnum` for its tiles plus 4 mapping dicts.

```python
class NewgameTiles(IntEnum):
    """Newgame environment tiles."""
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    # ...

NEWGAME_TILE_CHARS = {0: '.', 1: '#', 2: 'P'}
NEWGAME_CHAR_TILES = {'.': 0, '#': 1, 'P': 2}
NEWGAME_TILE_NAMES = {0: 'empty', 1: 'wall', 2: 'player'}
NEWGAME_NAME_TILES = {'empty': 0, 'wall': 1, 'player': 2}
```

Then add elifs to two dispatch functions:

- `get_tile_mappings()` — return your 4 dicts
- `get_problem_type_from_env_name()` — match your env name prefix (e.g., `env_name.startswith('newgame')`)

### Step 2: Factory — `toolusepcg/factory.py`

**A) Import** your `NewgameTiles` enum.

**B) Add entries to all 3 class dicts:**

```python
_EMPTY_TILE = {
    # ...existing entries...
    'newgame': NewgameTiles.EMPTY,
}

_BLOCKING_TILE = {
    # ...existing entries...
    'newgame': NewgameTiles.WALL,
}

_TILE_WEIGHTS = {
    # ...existing entries...
    'newgame': {
        NewgameTiles.EMPTY: 0.45,
        NewgameTiles.WALL: 0.45,
        NewgameTiles.PLAYER: 0.02,
        # ...weights control the `weighted` init strategy
    },
}
```

**C) Add elif in `_create_env_with_config_dims()`** to build `env_name` and `env_params`:

```python
elif self._problem_type == 'newgame':
    env_name = f'newgame-w{width}-h{height}'
    env_params = {"width": width, "height": height}
```

**D) Add entry to `get_problem_class()` class_map:**

```python
'newgame': ('pcg_benchmark.probs.newgame', 'NewgameProblem'),
```

### Step 3: Configuration — `toolusepcg/config.py`

Five places to update:

**A) `_VALID_TARGETS_BY_PROBLEM`** — the set of valid controllable metric names:

```python
'newgame': {'metric1', 'metric2'},
```

**B) `TargetsConfig` dataclass** — add optional `TargetConfig` fields for each controllable metric:

```python
metric1: TargetConfig | None = None
metric2: TargetConfig | None = None
```

**C) `ScoringConfig` dataclass** — add weight fields (one per scoring component):

```python
w_metric1: float = 10.0
w_solvable: float = 50.0
```

**D) `get_control_ranges()`** — return valid ranges for controllability experiments:

```python
elif problem_type == 'newgame':
    return {
        'metric1': MetricRange(min_value=1, max_value=20, step=1),
    }
```

**E) `get_problem_type()`** — add elif matching your env name prefix.

### Step 4: Evaluation — `toolusepcg/evaluator.py`

**A) Add `_evaluate_newgame(self, level)` method:**

```python
def _evaluate_newgame(self, level: Level) -> EvalResult:
    info = self._env.info(level.data)

    metrics = {
        'metric1': info.get('metric1', 0),
        'metric2': info.get('metric2', 0),
    }

    targets = {}
    if self.config.targets.metric1:
        targets['metric1'] = self.config.targets.metric1

    # Solvability check
    solvable = info.get('solvability_signal', 0) > 0
    errors = [] if solvable else ["Level is not solvable"]

    return EvalResult(
        metrics=metrics, targets=targets,
        valid=solvable, errors=errors,
        info=info, problem_type="newgame",
    )
```

**B) Add elif in `evaluate()` dispatch.**

**C) Add `_format_newgame_metrics(self, result)` method** — human-readable metrics with goal annotations for the LLM prompt (e.g., `"metric1: 5 (goal: ~10)"`).

**D) Add elif in `format_metrics_for_prompt()` dispatch.**

### Step 5: Scoring — `toolusepcg/optimizer/scoring.py`

**A) Add `_score_newgame(self, result)` method:**

```python
def _score_newgame(self, result: EvalResult) -> float:
    m = result.metrics
    score = 0.0

    # Controllable metrics via helper
    score += self._metric_score(m.get('metric1', 0), ..., self.config.scoring.w_metric1, default_target)

    # Solvability bonus/penalty
    if m.get('solvable'):
        score += self.config.scoring.w_solvable
    else:
        score -= self.config.scoring.w_solvable

    # Change penalty (copy from existing games)
    change_ratio = m.get('change_ratio', 0.0)
    if change_ratio > 0.6:
        score -= self.config.scoring.w_change_penalty * (change_ratio - 0.6)

    return score
```

**B) Add elif in `score()` dispatch.**

**C) Add case in `format_score()`** for status display.

### Step 6: LLM Prompt — `toolusepcg/optimizer/greedy.py`

**A) Add `_build_newgame_prompt(self)` method.** This is the most game-specific part. Study existing prompts (e.g., `_build_zelda_prompt` or `_build_loderunner_prompt`) as templates. The prompt should include:

1. Current level ASCII (`self.current_level.to_string()`)
2. Game rules and tile meanings
3. Formatted metrics (`self.evaluator.format_metrics_for_prompt(...)`)
4. Dynamic goal text from `config.targets` (target vs maximize mode)
5. Available tools (`self.registry.format_tools_for_prompt()`)
6. Termination status (`self.termination.format_status()`)
7. Score display (`self.scorer.format_score(...)`)
8. JSON response format instructions (STEP/PROPOSE_SKILL/STOP)

**B) Add elif in `build_prompt()` dispatch.**

**C) Add entry to `_FEEDBACK_METRICS` class dict** — controls per-metric deltas shown in conversation history feedback:

```python
"newgame": [
    ("metric1", "metric1", None),       # (metric_key, display_label, format)
    ("metric2", "metric2", None),       # format: None=int, "pct"=percentage, "f3"=3 decimals
],
```

### Step 7: Tools — `toolusepcg/tools/`

Most games just use `PlaceTileTool` which works with any tile enum. In `_init_tools()` auto mode:

- Games with multiple tile types (zelda, sokoban, loderunner, smb) → `PlaceTileTool`
- Binary-like wall games → `PlaceWallSegmentTool`

Add your problem type to the appropriate branch if needed:

```python
if self.problem_type in ('zelda', 'sokoban', 'loderunner', 'smb', 'newgame'):
    self._tile_tool = PlaceTileTool(self.edit_manager)
```

If your game needs custom generator tools (e.g., Zelda has `generate_zelda_wall.py` that wraps binary generators while preserving special tiles), create `toolusepcg/tools/generate_newgame.py` and update:
- `toolusepcg/tools/__init__.py` exports
- `_init_tools()` `valid_tools` set and registration logic

### Step 8: Stats Overlay — `toolusepcg/logging/trace.py`

Add entry to `_STATS_METRIC_CONFIG`:

```python
'newgame': [
    # Controllable metrics first (have config targets)
    ('metric1', 'metric1', None),              # (metric_key, config_target_attr, hardcoded_fallback)
    ('metric2', 'metric2', None),
    # Fixed-target metrics
    ('players', None, 1),                      # hardcoded target=1, no config attr
    # Optional 4th element for display label
    ('info_key', 'config_attr', None, 'display_label'),
],
```

Tuple format: `(metric_key, config_target_attr, hardcoded_fallback [, display_label])`

Optionally add `render_newgame_simulation_gif()` if your game has simulation playback (see SMB/Zelda examples).

### Step 9: Experiment Solvability — `run_experiment.py`

Add elif in `_is_level_solvable()`:

```python
elif problem_type == "newgame":
    return bool(info.get("solvability_signal", 0) > 0)
```

This must match the solvability check in your evaluator. Unsolvable trials get `controllability_score=0`.

### Step 10: Config File — `configs/newgame.yaml`

```yaml
env:
  name: newgame-v0
  height: 16
  width: 16

targets:
  metric1:
    mode: "target"
    value: 5
  metric2:
    mode: "maximize"

scoring:
  w_metric1: 10.0
  w_solvable: 50.0

experiment:
  controllability:
    ranges:
      metric1:
        min_value: 1
        max_value: 20
        step: 1
```

## Optional Steps

### `toolusepcg/tools/eval_stats.py`

Add `_compute_newgame_gaps()` method. This tool is currently disabled but wired for future use.

### Simulation Rendering — `render_simulation.py`

If your game has simulation playback (like SMB/Zelda), add an elif branch in `render_simulation.py` to support re-rendering from saved runs. Also add the simulation GIF generation logic in `toolusepcg/logging/trace.py`.

### Cross-Model Tables — `regenerate_cross_model_tables.py`

If you run sweeps with the new game, update `SWEEP_SOURCES` to include directories containing your game's results.

## Design Notes

- **Single source of truth**: Tile enums are the master definition; all mappings derive from them.
- **Problem-type dispatch**: Most logic uses if/elif chains on `problem_type` string — search for existing games to find all dispatch points.
- **Config-driven targets**: All controllability flows from `config.targets.*` + `_VALID_TARGETS_BY_PROBLEM`.
- **Termination is game-agnostic**: `termination.py` tracks change budget as `height * width * multiplier` — no changes needed.
- **Change penalty is universal**: All `_score_*` methods apply the graduated change penalty when >60% of tiles differ from the last accepted level.
- **Naming consistency**: Control param names, config target names, and metric names should all match (aligned with the PCG benchmark). No translation layer needed.
