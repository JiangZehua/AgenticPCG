# Configuration Reference

## Edit Tools

The `optimizer.edit_tools` field controls which tools are available to the agent.

| Tool | Description |
|------|-------------|
| `null` (default) | Auto: binary/binarydoor → `place_wall_segment`, others → `place_tile` |
| `place_tile` | Generic tile tool (modes: single, line, rect) |
| `place_wall_segment` | Wall segment tool (line-only, wall/empty) |
| `place_single_tile` | Single tile only |
| `place_line` | Line placement only |
| `place_patch` | Rectangular patch only (min 2x2) |
| `generate_random` | Binary/BinaryDoor: random generation with configurable wall probability |
| `generate_maze` | Binary/BinaryDoor: DFS perfect maze |
| `generate_bsp` | Binary/BinaryDoor: BSP room-and-corridor |
| `generate_ca` | Binary/BinaryDoor: cellular automata evolution |
| `generate_connect` | Binary/BinaryDoor: remove small regions and connect |
| `generate_digger` | Binary/BinaryDoor: random-walk cave |
| `generate_wfc` | Binary/BinaryDoor: Wave Function Collapse |
| `generate_zelda_tile_placing` | Zelda: full-level random generation |
| `generate_zelda_random` | Zelda: random wall/empty (preserves special tiles) |
| `generate_zelda_maze` | Zelda: DFS maze walls (preserves special tiles) |
| `generate_zelda_bsp` | Zelda: BSP wall layout (preserves special tiles) |
| `generate_zelda_ca` | Zelda: cellular automata walls (preserves special tiles) |
| `generate_zelda_connect` | Zelda: connect empty regions (preserves special tiles) |
| `generate_zelda_digger` | Zelda: random-walk cave walls (preserves special tiles) |
| `generate_zelda_wfc` | Zelda: WFC wall layout (preserves special tiles) |

Any combination is valid. Binary generators convert between ToolUsePCG tiles (EMPTY=0/WALL=1) and generator tiles (solid=0/empty=1). Zelda generators need no conversion.

```yaml
optimizer:
  edit_tools: ["place_single_tile", "place_line", "place_patch"]
```

```bash
uv run python main.py --config configs/zelda.yaml --edit-tools place_single_tile place_line
uv run python main.py --config configs/zelda.yaml --edit-tools  # no tools (differs from default auto)
```

## Level Initialization

| Strategy | Behavior |
|----------|----------|
| `random` (default) | Uniform random from `content_space.sample()` |
| `empty` | All passable tiles (problem-type aware) |
| `filled` | All blocking tiles (problem-type aware) |
| `weighted` | Weighted random: common tiles ~40-50%, structural ~5-10%, special entities ~1-4% |
| `<file_path>` | Load ASCII level from file |

## Acceptance Strategy

| Strategy | Behavior |
|----------|----------|
| `greedy` (default) | Hill-climbing. Optionally accepts worse with probability `epsilon_accept_worse`. |
| `annealing` | Simulated annealing. `p = exp((new - current) / T)`, `T = T0 * decay^step`. |

```yaml
optimizer:
  strategy: annealing
  annealing_initial_temp: 10.0
  annealing_decay_rate: 0.95
```

## Conversation History

| Field | Default | Description |
|-------|---------|-------------|
| `conversation_length` | `1` | Turns to keep (1 = no history) |
| `conversation_filter` | `"all"` | `"all"` or `"accepted"` |

```yaml
optimizer:
  conversation_length: 3
  conversation_filter: accepted
```

## Control Parameters by Domain

Control parameter names match the PCG benchmark's control space directly.

| Domain | Control Parameters | Range (default) |
|--------|-------------------|-----------------|
| Binary Maze | `path` | W+H to W*H/2 |
| BinaryDoor | `door_path` | W+H to W*H/2 |
| Zelda 16x16 | `player_key`, `key_door` | 10-40 |
| Zelda 50x50 | `player_key`, `key_door` | 50-625 |
| Sokoban | `crates` | 1-9 |
| Lode Runner | `ladder`, `rope` | 0-35 |
| SMB | `enemies`, `jumps`, `coins` | 0-4, 0-6, 0-3 |

## Experiment Types

| Type | Per-Trial | Aggregation |
|------|-----------|-------------|
| `quality` | Vary seeds | `env.quality()` per trial + `env.diversity()` across solvable |
| `controllability` | Sample control params | `env.controlability()` per trial, gated on solvability |
| `diversity` | Same seed (same init) | `env.diversity()` across solvable trials |

## Sweep Config Structure

Sweeps compute a Cartesian product across 9 dimensions: models × problems × tools × inits × tool_calls × change_percentages × annealings × conv_lengths × experiments.

```yaml
defaults:
  provider: portkey
  num_trials: 10
  max_steps: 50
  max_workers: 4

models:
  - name: gemini-pro
    model: "@vertexai/gemini-2.5-pro"
    max_tokens: 65536

problems:
  - name: zelda
    config: configs/zelda.yaml

tools:            # optional, defaults to single "default" entry
  - name: default
  - name: granular
    edit_tools: ["place_single_tile", "place_line", "place_patch"]

inits:            # optional, defaults to "random"
  - name: random
  - name: empty

tool_calls:       # optional, defaults to max_tool_calls: 20
  - name: tc20
    max_tool_calls: 20

change_percentages:  # optional, defaults to change_percentage: 2.0
  - name: cp2
    change_percentage: 2.0

annealings:       # optional, defaults to greedy
  - name: greedy
  - name: sa
    strategy: annealing
    annealing_initial_temp: 10.0
    annealing_decay_rate: 0.95

conv_lengths:     # optional, defaults to conv_length: 1
  - name: cl1
    conv_length: 1

experiments:
  - quality
  - controllability
  - diversity
```

Each entry's `name` must be unique within its dimension (used in directory paths). Resumable via `--run-dir` (checks for `experiment_summary.json` in each combo dir).
