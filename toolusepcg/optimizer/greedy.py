"""Greedy hill-climb optimizer with epsilon exploration."""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from ..config import Config, get_problem_type
from ..level import Level
from ..factory import LevelFactory
from ..evaluator import StatsEvaluator, EvalResult
from ..edit_manager import EditManager
from ..tools import (
    ToolRegistry,
    PlaceWallSegmentTool,
    PlaceTileTool,
    PlaceSingleTileTool,
    PlaceLineTool,
    PlacePatchTool,
    CalculateStatsTool,
    GenerateRandomTool,
    GenerateMazeTool,
    GenerateBSPTool,
    GenerateCATool,
    GenerateConnectTool,
    GenerateDiggerTool,
    GenerateWFCTool,
    GenerateZeldaTilePlacingTool,
    GenerateZeldaRandomTool,
    GenerateZeldaMazeTool,
    GenerateZeldaBSPTool,
    GenerateZeldaCATool,
    GenerateZeldaConnectTool,
    GenerateZeldaDiggerTool,
    GenerateZeldaWFCTool,
)
from ..protocol import (
    MessageType,
    StepMessage,
    ProposeSkillMessage,
    StopMessage,
    parse_message,
    validate_message,
)
from ..protocol.repair import parse_with_repair
from .scoring import Scorer
from .termination import TerminationChecker


# Load environment variables
load_dotenv()


@dataclass
class StepRecord:
    """Record of a single optimization step."""

    step_num: int
    message_type: str
    raw_response: str = ""
    parsed_message: dict = field(default_factory=dict)
    tool_results: list[dict] = field(default_factory=list)
    level_before: str = ""
    level_after: str = ""
    metrics_before: dict = field(default_factory=dict)
    metrics_after: dict = field(default_factory=dict)
    score_before: float = 0.0
    score_after: float = 0.0
    accepted: bool = False
    accept_reason: str = ""
    num_tiles_changed: int = 0
    errors: list[str] = field(default_factory=list)
    truncated: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ConversationEntry:
    """A single turn in the conversation history."""
    prompt: str
    raw_response: str
    accepted: bool
    accept_reason: str
    message_type: str  # "STEP", "PARSE_ERROR", "STOP", "PROPOSE_SKILL"
    step_record: StepRecord | None = None


class GreedyOptimizer:
    """Greedy hill-climb optimizer with LLM agent.

    Uses EditManager for all level modifications, ensuring edits go through
    a central validated interface connected to the PCG benchmark.
    """

    def __init__(self, config: Config, logger: Any = None, skill_manager: Any = None, verbose: bool = True, debug: bool = False):
        """Initialize optimizer.

        Args:
            config: Full configuration.
            logger: Optional TraceLogger instance.
            skill_manager: Optional SkillGrowthManager instance.
            verbose: Whether to print step progress to terminal.
            debug: Whether to save full LLM prompts/responses to debug.log.
        """
        self.config = config
        self.logger = logger
        self.skill_manager = skill_manager
        self.verbose = verbose
        self.debug = debug

        # Initialize components
        self.factory = LevelFactory(config)
        self.evaluator = StatsEvaluator(config, self.factory.env)
        self.scorer = Scorer(config)
        self.termination = TerminationChecker(config)

        # EditManager - central controller for all level modifications
        self.edit_manager: EditManager | None = None

        # Current state (derived from EditManager)
        self.current_result: EvalResult | None = None
        self.current_score: float = float("-inf")

        # Problem type
        self.problem_type = get_problem_type(config.env.name)

        # Tool registry (will be initialized with EditManager)
        self.registry: ToolRegistry | None = None
        self._wall_tool: PlaceWallSegmentTool | None = None
        self._tile_tool: PlaceTileTool | None = None
        self._single_tile_tool: PlaceSingleTileTool | None = None
        self._line_tool: PlaceLineTool | None = None
        self._patch_tool: PlacePatchTool | None = None
        self._stats_tool: CalculateStatsTool | None = None

        # LLM client (Portkey or OpenAI, depending on model prefix)
        self._client: Any = None

        # Step tracking
        self.step_count = 0
        self.step_records: list[StepRecord] = []

        # Epsilon for accepting worse solutions (greedy mode)
        self.epsilon = config.optimizer.epsilon_accept_worse

        # Simulated annealing parameters
        self.sa_initial_temp = config.optimizer.annealing_initial_temp
        self.sa_decay_rate = config.optimizer.annealing_decay_rate

        # Conversation history
        self.conversation_length = config.optimizer.conversation_length
        self.conversation_filter = config.optimizer.conversation_filter
        self._conversation_history: list[ConversationEntry] = []

        # Debug log file handle (lazy init)
        self._debug_file = None

    @property
    def current_level(self) -> Level | None:
        """Get current level from EditManager."""
        if self.edit_manager is None:
            return None
        return self.edit_manager.level

    def _init_client(self) -> None:
        """Initialize LLM client based on config.llm.provider.

        Supported providers:
        - "portkey": Uses the Portkey SDK (default).
        - "openai": Uses the OpenAI SDK (works with vLLM, TGI, etc.).
        """
        if self._client is None:
            provider = self.config.llm.provider
            if provider == "portkey":
                from portkey_ai import Portkey
                self._client = Portkey(
                    base_url=self.config.llm.base_url,
                    strict_open_ai_compliance=False,
                )
            elif provider == "openai":
                from openai import OpenAI
                api_key = self.config.llm.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
                self._client = OpenAI(
                    base_url=self.config.llm.base_url,
                    api_key=api_key,
                )
            elif provider == "google":
                from google import genai
                api_key = self.config.llm.api_key or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("Google provider requires api_key in config or GOOGLE_API_KEY env var.")
                self._client = genai.Client(api_key=api_key)
            else:
                raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'portkey', 'openai', or 'google'.")

    def _init_tools(self) -> None:
        """Initialize tool registry with EditManager.

        All edit tools use EditManager for modifications.
        Tools registered depend on config.optimizer.edit_tools (if set)
        or problem type (auto mode).
        """
        self.registry = ToolRegistry()

        # NOTE: calculate_stats tool disabled — LLM outputs all tool calls in
        # one shot so it cannot actually use mid-step stats feedback.
        # self._stats_tool = CalculateStatsTool(self.evaluator, self.edit_manager)
        # self.registry.register(self._stats_tool)

        edit_tools = self.config.optimizer.edit_tools

        if edit_tools is not None:
            # Explicit tool list from config
            valid_tools = {
                "place_tile", "place_wall_segment", "place_single_tile", "place_line", "place_patch",
                "generate_random", "generate_maze", "generate_bsp", "generate_ca",
                "generate_connect", "generate_digger", "generate_wfc",
                "generate_zelda_tile_placing",
                "generate_zelda_random", "generate_zelda_maze", "generate_zelda_bsp",
                "generate_zelda_ca", "generate_zelda_connect", "generate_zelda_digger",
                "generate_zelda_wfc",
            }
            for tool_name in edit_tools:
                if tool_name not in valid_tools:
                    raise ValueError(
                        f"Unknown edit tool: {tool_name!r}. "
                        f"Valid tools: {sorted(valid_tools)}"
                    )
            if "place_tile" in edit_tools:
                self._tile_tool = PlaceTileTool(self.edit_manager)
                self.registry.register(self._tile_tool)
            if "place_wall_segment" in edit_tools:
                self._wall_tool = PlaceWallSegmentTool(self.edit_manager)
                self.registry.register(self._wall_tool)
            if "place_single_tile" in edit_tools:
                self._single_tile_tool = PlaceSingleTileTool(self.edit_manager)
                self.registry.register(self._single_tile_tool)
            if "place_line" in edit_tools:
                self._line_tool = PlaceLineTool(self.edit_manager)
                self.registry.register(self._line_tool)
            if "place_patch" in edit_tools:
                self._patch_tool = PlacePatchTool(self.edit_manager)
                self.registry.register(self._patch_tool)
            if "generate_random" in edit_tools:
                self.registry.register(GenerateRandomTool(self.edit_manager))
            if "generate_maze" in edit_tools:
                self.registry.register(GenerateMazeTool(self.edit_manager))
            if "generate_bsp" in edit_tools:
                self.registry.register(GenerateBSPTool(self.edit_manager))
            if "generate_ca" in edit_tools:
                self.registry.register(GenerateCATool(self.edit_manager))
            if "generate_connect" in edit_tools:
                self.registry.register(GenerateConnectTool(self.edit_manager))
            if "generate_digger" in edit_tools:
                self.registry.register(GenerateDiggerTool(self.edit_manager))
            if "generate_wfc" in edit_tools:
                self.registry.register(GenerateWFCTool(self.edit_manager))
            if "generate_zelda_tile_placing" in edit_tools:
                self.registry.register(GenerateZeldaTilePlacingTool(self.edit_manager))
            if "generate_zelda_random" in edit_tools:
                self.registry.register(GenerateZeldaRandomTool(self.edit_manager))
            if "generate_zelda_maze" in edit_tools:
                self.registry.register(GenerateZeldaMazeTool(self.edit_manager))
            if "generate_zelda_bsp" in edit_tools:
                self.registry.register(GenerateZeldaBSPTool(self.edit_manager))
            if "generate_zelda_ca" in edit_tools:
                self.registry.register(GenerateZeldaCATool(self.edit_manager))
            if "generate_zelda_connect" in edit_tools:
                self.registry.register(GenerateZeldaConnectTool(self.edit_manager))
            if "generate_zelda_digger" in edit_tools:
                self.registry.register(GenerateZeldaDiggerTool(self.edit_manager))
            if "generate_zelda_wfc" in edit_tools:
                self.registry.register(GenerateZeldaWFCTool(self.edit_manager))
        else:
            # Auto mode: select based on problem type
            if self.problem_type in ('zelda', 'sokoban', 'loderunner', 'smb'):
                self._tile_tool = PlaceTileTool(self.edit_manager)
                self.registry.register(self._tile_tool)
            else:
                self._wall_tool = PlaceWallSegmentTool(self.edit_manager)
                self.registry.register(self._wall_tool)

    def initialize(self, seed: int | None = None, initial_level: Level | None = None) -> None:
        """Initialize optimizer with starting level.

        Args:
            seed: Random seed for level generation.
            initial_level: Optional pre-specified initial level.
        """
        if initial_level is not None:
            level = initial_level.copy()
            # Ensure level has correct problem type
            level.problem_type = self.problem_type
        else:
            level = self.factory.create_random(seed)

        # Create EditManager as central controller
        self.edit_manager = EditManager(self.factory.env, level, problem_type=self.problem_type)

        # Initialize tools with EditManager
        self._init_tools()

        # Evaluate initial level
        self.current_result = self.evaluator.evaluate(self.current_level)
        self.current_score = self.scorer.score(self.current_result)

        # Reset tracking
        self.termination.reset()
        self.step_count = 0
        self.step_records = []
        self._conversation_history = []

        # Log initial state
        if self.logger:
            self.logger.log_initial_level(self.current_level, self.current_result)

    def _write_debug(self, text: str) -> None:
        """Write a line to the debug log file.

        Args:
            text: Text to write.
        """
        if not self.debug:
            return

        # Lazy init: open debug.log in run directory
        if self._debug_file is None and self.logger:
            debug_path = self.logger.get_run_dir() / "debug.log"
            self._debug_file = open(debug_path, "a")

        if self._debug_file:
            self._debug_file.write(text + "\n")
            self._debug_file.flush()

    def build_prompt(self) -> str:
        """Build prompt for LLM.

        Returns:
            Complete prompt string.
        """
        if self.problem_type == 'zelda':
            prompt = self._build_zelda_prompt()
        elif self.problem_type == 'sokoban':
            prompt = self._build_sokoban_prompt()
        elif self.problem_type == 'loderunner':
            prompt = self._build_loderunner_prompt()
        elif self.problem_type == 'smb':
            prompt = self._build_smb_prompt()
        elif self.problem_type == 'binarydoor':
            prompt = self._build_binarydoor_prompt()
        else:
            prompt = self._build_binary_prompt()

        # Inform agent of tool call limit
        max_calls = self.config.optimizer.max_tool_calls_per_step
        prompt += f"\n\nNote: You may use up to {max_calls} tool calls per step. Any beyond this limit will be dropped."

        # Append extra instruction if configured
        extra = self.config.optimizer.extra_instruction
        if extra:
            prompt += f"\n\n## Extra Instruction:\n{extra}"

        return prompt

    def _build_binary_prompt(self) -> str:
        """Build prompt for binary maze optimization."""
        level_str = self.current_level.to_string()
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        # Build dynamic goal based on path target mode
        path_target = self.config.targets.path
        if path_target.mode == "target" and path_target.value is not None:
            path_goal = f"achieve a longest shortest path of ~{path_target.value}"
        else:
            path_goal = "maximize the longest shortest path"

        # Build dynamic goal for regions
        region_target = self.config.targets.num_connected_regions
        if region_target.mode == "target":
            target_regions = region_target.value if region_target.value is not None else 1
            if target_regions == 1:
                region_goal = "while maintaining exactly 1 connected region"
            else:
                region_goal = f"while achieving ~{target_regions} connected regions"
        else:
            region_goal = "while maximizing the number of connected regions"

        prompt = f"""You are an AI agent optimizing a binary maze level. Your goal is to {path_goal} {region_goal}.

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
Legend: '.' = empty (passable), '#' = wall (blocked)

## {metrics_str}

## {score_str}

## Termination Status:
{term_status}

## {tools_str}

## Instructions:
1. Analyze the current level and its metrics
2. Plan edits to improve the score (move metrics toward goals described above)
3. Execute tool calls to modify the level

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_wall_segment", "parameters": {{...}}}}
  ],
  "acceptance_hint": "hint about whether to accept changes"
}}
```

### PROPOSE_SKILL - To propose a new tool:
```json
{{
  "type": "PROPOSE_SKILL",
  "rationale": "why this tool would help",
  "skill_spec": {{
    "name": "tool_name",
    "description": "what it does",
    "parameters": {{}},
    "implementation_hint": "how to implement"
  }}
}}
```

### STOP - To terminate optimization:
```json
{{
  "type": "STOP",
  "rationale": "why stopping now",
  "final_notes": "observations about the result"
}}
```

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _door_adjacent_str(self, door_pos: tuple[int, int]) -> str:
        """Return a human-readable string for the inner cell adjacent to a border door.

        Args:
            door_pos: (row, col) in the augmented/bordered grid.
        """
        r, c = door_pos
        h, w = self.current_level.height, self.current_level.width
        ah, aw = h + 2, w + 2

        if r == 0:
            # Top border: adjacent inner cell is directly below
            return f"row=0, col={c - 1}"
        elif r == ah - 1:
            # Bottom border: adjacent inner cell is directly above
            return f"row={h - 1}, col={c - 1}"
        elif c == 0:
            # Left border: adjacent inner cell is directly right
            return f"row={r - 1}, col=0"
        elif c == aw - 1:
            # Right border: adjacent inner cell is directly left
            return f"row={r - 1}, col={w - 1}"
        else:
            return f"row={r - 1}, col={c - 1}"

    def _build_bordered_level_with_doors(self) -> str:
        """Build a bordered ASCII view of the level with doors marked as 'D'.

        The inner grid is the current level. The border is walls (#) except
        at the two door positions which are marked 'D'.

        Returns:
            Multi-line string of the bordered level.
        """
        level = self.current_level
        h, w = level.height, level.width
        tile_chars = level.tile_chars

        # Get door positions from the env
        door1 = self.factory.env._problem._door1
        door2 = self.factory.env._problem._door2

        # Build augmented grid (H+2 x W+2)
        ah, aw = h + 2, w + 2
        lines = []
        for r in range(ah):
            row_chars = []
            for c in range(aw):
                if (r, c) == door1 or (r, c) == door2:
                    row_chars.append('D')
                elif r == 0 or r == ah - 1 or c == 0 or c == aw - 1:
                    row_chars.append('#')
                else:
                    # Inner cell: (r-1, c-1) in the original level
                    tile_val = int(level.data[r - 1, c - 1])
                    row_chars.append(tile_chars.get(tile_val, '?'))
            lines.append(''.join(row_chars))

        return '\n'.join(lines)

    def _build_binarydoor_prompt(self) -> str:
        """Build prompt for binary door maze optimization."""
        bordered_level_str = self._build_bordered_level_with_doors()
        inner_level_str = self.current_level.to_string()
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        # Get door positions
        door1 = self.factory.env._problem._door1
        door2 = self.factory.env._problem._door2

        # Build dynamic goal based on door_path target mode
        door_path_cfg = self.config.targets.door_path
        if door_path_cfg and door_path_cfg.mode == "target" and door_path_cfg.value is not None:
            path_goal = f"achieve a door-to-door path length of ~{door_path_cfg.value}"
        else:
            path_goal = "maximize the door-to-door path length"

        # Build dynamic goal for regions
        region_target = self.config.targets.num_connected_regions
        if region_target.mode == "target":
            target_regions = region_target.value if region_target.value is not None else 1
            if target_regions == 1:
                region_goal = "while maintaining exactly 1 connected region"
            else:
                region_goal = f"while achieving ~{target_regions} connected regions"
        else:
            region_goal = "while maximizing the number of connected regions"

        prompt = f"""You are an AI agent optimizing a binary maze with two door openings. Your goal is to {path_goal} {region_goal}.

## How This Problem Works:
- The maze has a fixed border of walls with exactly TWO door openings (marked 'D' below).
- You edit only the inner grid (rows 0-{self.current_level.height-1}, columns 0-{self.current_level.width-1}).
- The door_path metric measures the shortest path from door1 to door2 through the maze (including passing through the border doors).
- A longer, more winding path through the maze = higher door_path.
- Doors are at FIXED positions and cannot be moved.

## Bordered Level (showing doors and border):
```
{bordered_level_str}
```
- 'D' = door opening (fixed), '#' = wall (border or inner), '.' = empty (passable)
- Door 1: row={door1[0]}, col={door1[1]} (in bordered grid) → adjacent inner cell: {self._door_adjacent_str(door1)}
- Door 2: row={door2[0]}, col={door2[1]} (in bordered grid) → adjacent inner cell: {self._door_adjacent_str(door2)}

## Editable Inner Level (rows are y, columns are x, 0-indexed):
```
{inner_level_str}
```
Legend: '.' = empty (passable), '#' = wall (blocked)

## {metrics_str}

## {score_str}

## Termination Status:
{term_status}

## {tools_str}

## Key Strategy:
- The inner cells adjacent to each door MUST be empty ('.') for the path to reach the doors.
- Create a single winding path connecting both doors to maximize door_path.
- Avoid creating multiple disconnected regions — keep exactly 1 connected region.
- Place walls strategically to force a longer path between the doors.

## Instructions:
1. Analyze the current level and its metrics
2. Plan edits to improve the score (move metrics toward goals described above)
3. Execute tool calls to modify the level

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_wall_segment", "parameters": {{...}}}}
  ],
  "acceptance_hint": "hint about whether to accept changes"
}}
```

### PROPOSE_SKILL - To propose a new tool:
```json
{{
  "type": "PROPOSE_SKILL",
  "rationale": "why this tool would help",
  "skill_spec": {{
    "name": "tool_name",
    "description": "what it does",
    "parameters": {{}},
    "implementation_hint": "how to implement"
  }}
}}
```

### STOP - To terminate optimization:
```json
{{
  "type": "STOP",
  "rationale": "why stopping now",
  "final_notes": "observations about the result"
}}
```

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _build_zelda_prompt(self) -> str:
        """Build prompt for Zelda level optimization."""
        level_str = self.current_level.to_string()
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        # Build dynamic goal descriptions based on targets
        pk_target = self.config.targets.player_key
        kd_target = self.config.targets.key_door
        sol_target = self.config.targets.solution_length

        # Build sub-goals for player_key and key_door based on mode
        pk_parts = []
        if pk_target and pk_target.mode == "target" and pk_target.value is not None:
            pk_parts.append(f"a player-to-key path length of ~{pk_target.value}")
        elif pk_target and pk_target.mode == "maximize":
            pk_parts.append("a maximized player-to-key path length")
        if kd_target and kd_target.mode == "target" and kd_target.value is not None:
            pk_parts.append(f"a key-to-door path length of ~{kd_target.value}")
        elif kd_target and kd_target.mode == "maximize":
            pk_parts.append("a maximized key-to-door path length")

        if pk_parts:
            path_goal = "3. Achieve " + " and ".join(pk_parts)
        elif sol_target and sol_target.mode == "target" and sol_target.value is not None:
            path_goal = f"3. Achieve a solution length (total path from player -> key -> door) of ~{sol_target.value}"
        else:
            path_goal = "3. Maximize the solution length (total path from player -> key -> door)"

        prompt = f"""You are an AI agent optimizing a Zelda dungeon level. Your goals are:
1. Have exactly 1 player (P), 1 key (K), and 1 door (D)
2. Ensure the level is playable: player can reach key, key can reach door
{path_goal}
4. Maintain exactly 1 connected region
5. Have an appropriate number of enemies (E)

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
Legend: '#' = wall, '.' = empty, 'P' = player, 'K' = key, 'D' = door, 'E' = enemy

## {metrics_str}

## {score_str}

## Termination Status:
{term_status}

## {tools_str}

## Instructions:
1. First ensure the level has exactly 1 player, 1 key, and 1 door
2. Ensure paths exist: player->key and key->door must be reachable
3. Then optimize for longer solution paths by rearranging walls
4. Add enemies for challenge but don't block critical paths

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_tile", "parameters": {{"mode": "single", "tile_type": "player", "y": 1, "x": 1}}}}
  ],
  "acceptance_hint": "hint about whether to accept changes"
}}
```

### PROPOSE_SKILL - To propose a new tool:
```json
{{
  "type": "PROPOSE_SKILL",
  "rationale": "why this tool would help",
  "skill_spec": {{
    "name": "tool_name",
    "description": "what it does",
    "parameters": {{}},
    "implementation_hint": "how to implement"
  }}
}}
```

### STOP - To terminate optimization:
```json
{{
  "type": "STOP",
  "rationale": "why stopping now",
  "final_notes": "observations about the result"
}}
```

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _build_sokoban_prompt(self) -> str:
        """Build prompt for Sokoban level optimization."""
        level_str = self.current_level.to_string()
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        # Build dynamic crate goal based on target mode
        crate_cfg = self.config.targets.crates
        if crate_cfg and crate_cfg.mode == "target" and crate_cfg.value is not None:
            crate_goal = f"2. Have ~{crate_cfg.value} crates ($) that can be pushed onto target locations (.)"
        elif crate_cfg and crate_cfg.mode == "maximize":
            crate_goal = "2. Maximize the number of crates ($) that can be pushed onto target locations (.)"
        else:
            crate_goal = "2. Have crates ($) that can be pushed onto target locations (.)"

        prompt = f"""You are an AI agent optimizing a Sokoban puzzle level. Your goals are:
1. Have exactly 1 player (@)
{crate_goal}
3. Ensure the number of crates equals the number of targets
4. Ensure the level is solvable (player can push all crates onto targets)
5. Maximize the solution length for more interesting puzzles

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
Legend: '#' = solid wall, ' ' = empty floor, '@' = player, '$' = crate, '.' = target

## {metrics_str}

## {score_str}

## Termination Status:
{term_status}

## {tools_str}

## Instructions:
1. First ensure the level has exactly 1 player
2. Ensure crates and targets are balanced (same count)
3. Position crates so they can be pushed onto targets
4. Leave enough empty space for the player to maneuver
5. Avoid creating unsolvable situations (crate in corner, crates blocking each other)

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_tile", "parameters": {{"mode": "single", "tile_type": "player", "y": 1, "x": 1}}}}
  ],
  "acceptance_hint": "hint about whether to accept changes"
}}
```

### PROPOSE_SKILL - To propose a new tool:
```json
{{
  "type": "PROPOSE_SKILL",
  "rationale": "why this tool would help",
  "skill_spec": {{
    "name": "tool_name",
    "description": "what it does",
    "parameters": {{}},
    "implementation_hint": "how to implement"
  }}
}}
```

### STOP - To terminate optimization:
```json
{{
  "type": "STOP",
  "rationale": "why stopping now",
  "final_notes": "observations about the result"
}}
```

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _build_loderunner_prompt(self) -> str:
        """Build prompt for Lode Runner level optimization."""
        level_str = self.current_level.to_string()
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        # Build dynamic ladder/rope goals based on target mode
        ladder_cfg = self.config.targets.ladder
        rope_cfg = self.config.targets.rope
        if ladder_cfg and ladder_cfg.mode == "target" and ladder_cfg.value is not None:
            ladder_goal = f"4. Use ~{ladder_cfg.value} ladders (L) for vertical traversal"
        elif ladder_cfg and ladder_cfg.mode == "maximize":
            ladder_goal = "4. Maximize ladders (L) for vertical traversal"
        else:
            ladder_goal = "4. Use ladders (L) for vertical traversal"
        if rope_cfg and rope_cfg.mode == "target" and rope_cfg.value is not None:
            rope_goal = f"and ~{rope_cfg.value} ropes (R) for horizontal traversal"
        elif rope_cfg and rope_cfg.mode == "maximize":
            rope_goal = "and maximize ropes (R) for horizontal traversal"
        else:
            rope_goal = "and ropes (R) for horizontal traversal"

        prompt = f"""You are an AI agent optimizing a Lode Runner level. Your goals are:
1. Have exactly 1 player (P)
2. Have collectible gold pieces (G) that the player can reach
3. Have enemies (E) for challenge
{ladder_goal} {rope_goal}
5. Ensure the player can collect all gold

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
Legend: 'S' = solid brick, '.' = empty, 'P' = player, 'G' = gold, 'E' = enemy, 'L' = ladder, 'R' = rope

## {metrics_str}

## {score_str}

## Termination Status:
{term_status}

## {tools_str}

## Instructions:
1. First ensure the level has exactly 1 player
2. Add gold pieces that are reachable by the player
3. Create paths using ladders (vertical) and ropes (horizontal)
4. Add enemies for challenge but ensure gold is still collectible

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_tile", "parameters": {{"mode": "single", "tile_type": "player", "y": 1, "x": 1}}}}
  ],
  "acceptance_hint": "hint about whether to accept changes"
}}
```

### PROPOSE_SKILL - To propose a new tool:
```json
{{
  "type": "PROPOSE_SKILL",
  "rationale": "why this tool would help",
  "skill_spec": {{
    "name": "tool_name",
    "description": "what it does",
    "parameters": {{}},
    "implementation_hint": "how to implement"
  }}
}}
```

### STOP - To terminate optimization:
```json
{{
  "type": "STOP",
  "rationale": "why stopping now",
  "final_notes": "observations about the result"
}}
```

Respond with ONLY the JSON object, no additional text."""

        return prompt

    # EventType values from smbtile engine (helper.py)
    _KILL_EVENT_TYPES = {2, 3, 4}  # STOMP_KILL, FIRE_KILL, SHELL_KILL (not FALL_KILL=5)
    _KILL_TYPE_NAMES = {2: "stomped", 3: "fire-killed", 4: "shell-killed"}
    # SpriteType param values → display names
    _SPRITE_TYPE_NAMES = {
        2: "goomba", 3: "winged_goomba",
        4: "red_koopa", 5: "winged_red_koopa",
        6: "green_koopa", 7: "winged_green_koopa",
        8: "spiny", 9: "winged_spiny",
    }
    # Padding added by _convert2str in problem.py
    _ENGINE_PAD = 3

    def _build_smb_trajectory_map(self, level_str: str, info: dict | None) -> str | None:
        """Build an annotated ASCII map showing Mario's trajectory and kill locations.

        Args:
            level_str: Current level as string (from Level.to_string()).
            info: Raw info dict from PCG benchmark evaluation.

        Returns:
            Annotated map string, or None if no simulation data.
        """
        if info is None:
            return None
        locations = info.get("locations", [])
        if not locations:
            return None

        lines = level_str.split('\n')
        grid = [list(line) for line in lines if line]
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        if width == 0:
            return None

        # Collect trajectory tile positions (engine pixel → original content coords)
        trajectory_tiles: set[tuple[int, int]] = set()
        for loc in locations:
            tx = int(loc[0] / 16) - self._ENGINE_PAD
            ty = int(loc[1] / 16)
            tx = max(0, min(width - 1, tx))
            ty = max(0, min(height - 1, ty))
            trajectory_tiles.add((ty, tx))

        # Collect kill positions from game_events
        kill_tiles: set[tuple[int, int]] = set()
        kill_details: dict[tuple[int, int], list[tuple[str, str]]] = {}  # (row,col) → [(kill_method, enemy_name)]
        for ge in info.get("game_events", []):
            if ge["type"] in self._KILL_EVENT_TYPES:
                tx = int(ge["mario_x"] / 16) - self._ENGINE_PAD
                ty = int(ge["mario_y"] / 16)
                tx = max(0, min(width - 1, tx))
                ty = max(0, min(height - 1, ty))
                kill_tiles.add((ty, tx))
                method = self._KILL_TYPE_NAMES.get(ge["type"], "killed")
                enemy = self._SPRITE_TYPE_NAMES.get(ge["param"], f"enemy({ge['param']})")
                kill_details.setdefault((ty, tx), []).append((method, enemy))

        # Build annotated map
        for (row, col) in kill_tiles:
            if 0 <= row < height and 0 <= col < width:
                grid[row][col] = '!'
        for (row, col) in trajectory_tiles:
            if (row, col) not in kill_tiles and 0 <= row < height and 0 <= col < width:
                if grid[row][col] == '.':
                    grid[row][col] = '*'

        return '\n'.join(''.join(row) for row in grid)

    def _build_smb_simulation_summary(self, info: dict | None) -> str | None:
        """Build a text summary of the SMB simulation results.

        Args:
            info: Raw info dict from PCG benchmark evaluation.

        Returns:
            Simulation summary text, or None if info is unavailable.
        """
        if info is None:
            return None
        locations = info.get("locations", [])
        if not locations:
            return ("Simulation was skipped because the level failed structural pre-checks "
                    "(empty ratio, tube structure, noise, or floating enemies). "
                    "Fix structural issues first.")

        complete = info.get("complete", 0.0)
        enemies_killed = info.get("enemies", 0)
        coins_collected = info.get("coins", 0)
        jumps = info.get("jumps", 0)
        width = info.get("width", 32)

        lines = ["## Simulation Results",
                 "The Mario AI agent played through the level. Here is what happened:"]

        # Completion
        if complete >= 1.0:
            lines.append("- Completion: 100% (reached the flag)")
        else:
            last_x = int(locations[-1][0] / 16) - self._ENGINE_PAD
            last_x = max(0, min(width - 1, last_x))
            lines.append(f"- Completion: {complete*100:.0f}% (Mario died or got stuck around column {last_x})")

        # Kill details
        game_events = info.get("game_events", [])
        kill_descriptions = []
        for ge in game_events:
            if ge["type"] in self._KILL_EVENT_TYPES:
                method = self._KILL_TYPE_NAMES.get(ge["type"], "killed")
                enemy = self._SPRITE_TYPE_NAMES.get(ge["param"], f"enemy({ge['param']})")
                tx = max(0, min(width - 1, int(ge["mario_x"] / 16) - self._ENGINE_PAD))
                ty = int(ge["mario_y"] / 16)
                kill_descriptions.append(f"{enemy} {method} at tile [{tx}, {ty}]")

        if kill_descriptions:
            lines.append(f"- Enemies killed: {enemies_killed} ({', '.join(kill_descriptions)})")
        else:
            lines.append(f"- Enemies killed: {enemies_killed}")

        lines.append(f"- Coins collected: {coins_collected}")
        # Count stomp bounces (not counted as jumps)
        stomp_bounces = sum(1 for ge in game_events if ge["type"] == 2)  # STOMP_KILL
        jump_line = f"- Jumps performed: {jumps}"
        if stomp_bounces > 0:
            jump_line += f" (stomp bounces: {stomp_bounces}, not counted as jumps)"
        lines.append(jump_line)

        # Traversal range
        first_x = max(0, int(locations[0][0] / 16) - self._ENGINE_PAD)
        last_x = max(0, int(locations[-1][0] / 16) - self._ENGINE_PAD)
        last_x = min(width - 1, last_x)
        lines.append(f"- Mario traversed from column {first_x} to column {last_x}")

        # Enemy behavior guide
        lines.append("")
        lines.append("Enemy behavior during simulation:")
        lines.append("- Goomba (G): Walks horizontally, reverses on walls. Speed ~1.75 px/frame.")
        lines.append("- Koopa (K): Green koopa walks horizontally, falls off cliffs. Becomes a kickable shell when stomped.")
        lines.append("- Spiny (Y): Walks horizontally like goomba but CANNOT be stomped (hurts Mario). Must be avoided or killed with shell/fireball.")

        return '\n'.join(lines)

    def _build_smb_prompt(self) -> str:
        """Build prompt for Super Mario Bros level optimization."""
        level_str = self.current_level.to_string()
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        # Build dynamic enemy/coin/jump goals based on target mode
        enemies_cfg = self.config.targets.enemies
        coins_cfg = self.config.targets.coins
        jumps_cfg = self.config.targets.jumps
        smb_target_parts = []
        smb_maximize_parts = []
        if enemies_cfg and enemies_cfg.mode == "target" and enemies_cfg.value is not None:
            smb_target_parts.append(f"~{enemies_cfg.value} enemies killed")
        elif enemies_cfg and enemies_cfg.mode == "maximize":
            smb_maximize_parts.append("enemies killed")
        if jumps_cfg and jumps_cfg.mode == "target" and jumps_cfg.value is not None:
            smb_target_parts.append(f"~{jumps_cfg.value} jumps performed")
        elif jumps_cfg and jumps_cfg.mode == "maximize":
            smb_maximize_parts.append("jumps performed")
        if coins_cfg and coins_cfg.mode == "target" and coins_cfg.value is not None:
            smb_target_parts.append(f"~{coins_cfg.value} coins collected")
        elif coins_cfg and coins_cfg.mode == "maximize":
            smb_maximize_parts.append("coins collected")
        smb_goal_parts = []
        if smb_target_parts:
            smb_goal_parts.append(f"achieve {', '.join(smb_target_parts)}")
        if smb_maximize_parts:
            smb_goal_parts.append(f"maximize {', '.join(smb_maximize_parts)}")
        if smb_goal_parts:
            smb_goal = "3. " + " and ".join(smb_goal_parts).capitalize() + " during gameplay simulation"
        else:
            smb_goal = "3. Place enemies on solid ground, not floating"

        # Build simulation trajectory map and summary from current evaluation
        info = self.current_result.info if self.current_result else None
        traj_map = self._build_smb_trajectory_map(level_str, info)
        sim_summary = self._build_smb_simulation_summary(info)

        # Build trajectory map section
        if traj_map is not None:
            traj_section = f"""## Simulation Trajectory Map (last evaluation)
```
{traj_map}
```
Legend: '*' = Mario's path (empty tiles only), '!' = enemy killed here
Note: Enemies move during simulation. '!' marks where Mario was when the kill happened, not the enemy's spawn position. Enemies that were avoided or not reached are still shown at their original tile positions (G/K/Y)."""
        else:
            traj_section = """## Simulation Trajectory Map (last evaluation)
No trajectory available — simulation was skipped or has not run yet."""

        # Build simulation summary section
        sim_section = sim_summary if sim_summary else ""

        prompt = f"""You are an AI agent optimizing a Super Mario Bros level. Your goals are:
1. Create a completable level (Mario can reach the end)
2. Have proper tube/pipe structures (tubes must span 2 tiles properly)
{smb_goal}
4. Minimize horizontal noise for smoother gameplay
5. Add coins and question blocks for rewards

## How Evaluation Works
A Mario AI agent (A* solver) plays your level from left to right. The metrics below reflect the **gameplay outcome**, not the tile layout:
- **enemies_killed**: Number of enemies the player stomped or hit with shells during play. Enemies that fall off cliffs on their own do NOT count. Simply placing enemy tiles does not guarantee kills — enemies must be on the player's path where they cannot be avoided.
- **coins_collected**: Number of coins the player picked up during play. Coins must be on or near the traversal path.
- **jumps**: Number of voluntary jumps the player performed from the ground. Bouncing off enemies (stomp bounces) does NOT count. More gaps and elevated platforms force more jumps.
- **complete**: Whether the player reached the end flag (1.0 = yes).

To increase enemies_killed, place enemies on narrow ground platforms that the player must cross (not on optional platforms). To increase coins_collected, place coins along the main path or on mandatory platforms. To increase jumps, create gaps in the ground and elevated platforms that force the player to jump from the ground — enemy stomps do not count as jumps.

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
Legend: '.' = empty, 'X' = solid floor, 'L' = ladder, 'B' = brick, '?' = question block, 'T' = tube, 'C' = coin, 'G' = goomba, 'K' = koopa, 'Y' = spiny

{traj_section}

## {metrics_str}

{sim_section}

## {score_str}

## Termination Status:
{term_status}

## {tools_str}

## Instructions:
1. Ensure the level is completable (Mario can traverse from start to end)
2. Place solid ground (X) for Mario to walk on
3. Add platforms using bricks (B) and question blocks (?)
4. Place enemies (G, K, Y) on solid ground along the player's path — enemies the player can avoid will not count
5. Place coins (C) along the main traversal path — coins the player never reaches will not count
6. Avoid creating impossible jumps or dead ends

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_tile", "parameters": {{"mode": "single", "tile_type": "solid", "y": 15, "x": 1}}}}
  ],
  "acceptance_hint": "hint about whether to accept changes"
}}
```

### PROPOSE_SKILL - To propose a new tool:
```json
{{
  "type": "PROPOSE_SKILL",
  "rationale": "why this tool would help",
  "skill_spec": {{
    "name": "tool_name",
    "description": "what it does",
    "parameters": {{}},
    "implementation_hint": "how to implement"
  }}
}}
```

### STOP - To terminate optimization:
```json
{{
  "type": "STOP",
  "rationale": "why stopping now",
  "final_notes": "observations about the result"
}}
```

Respond with ONLY the JSON object, no additional text."""

        return prompt

    # Per-problem-type metric keys for feedback.
    # Each entry: (metric_key, display_label, format_func_name_or_None)
    # format_func_name_or_None: "pct" for percentage, None for raw value
    _FEEDBACK_METRICS: dict[str, list[tuple[str, str, str | None]]] = {
        "binary": [
            ("path", "path", None),
            ("num_connected_regions", "regions", None),
        ],
        "binarydoor": [
            ("door_path", "door_path", None),
            ("num_connected_regions", "regions", None),
        ],
        "zelda": [
            ("player_key", "player_key", None),
            ("key_door", "key_door", None),
            ("players", "players", None),
            ("keys", "keys", None),
            ("doors", "doors", None),
            ("enemies", "enemies", None),
            ("solution_length", "solution_length", None),
            ("regions", "regions", None),
        ],
        "sokoban": [
            ("crates", "crates", None),
            ("players", "players", None),
            ("targets", "targets", None),
            ("solution_length", "solution_length", None),
        ],
        "loderunner": [
            ("ladder", "ladder", None),
            ("rope", "rope", None),
            ("player", "player", None),
            ("gold", "gold", None),
            ("enemy", "enemy", None),
            ("collected_gold", "collected_gold", None),
        ],
        "smb": [
            ("enemies", "enemies_killed", None),
            ("coins", "coins_collected", None),
            ("jumps", "jumps", None),
            ("complete", "complete", "pct"),
            ("tube_issues", "tube_issues", None),
            ("noise", "noise", "f3"),
        ],
    }

    def _build_step_feedback(self, entry: ConversationEntry) -> str:
        """Build a rich feedback summary from a conversation entry's step record.

        Includes accept/reject status, per-metric deltas with targets, and
        tool call summary. Falls back to a simple one-liner when no step_record
        is available or the message type is not STEP.

        Args:
            entry: The conversation entry to build feedback from.

        Returns:
            Multi-line feedback string (no trailing newline).
        """
        # Status line (always present)
        if entry.accepted:
            status_line = f"[Previous step result: ACCEPTED - {entry.accept_reason}]"
        else:
            status_line = f"[Previous step result: REJECTED - {entry.accept_reason}]"

        rec = entry.step_record
        # Non-STEP messages or missing record: just the status line
        if rec is None or entry.message_type != "STEP" or not rec.metrics_after:
            return status_line

        lines = [status_line]

        # Metric deltas
        metric_defs = self._FEEDBACK_METRICS.get(self.problem_type, [])
        if metric_defs:
            lines.append("Metrics change from your last edit:")
            for metric_key, display_label, fmt in metric_defs:
                before = rec.metrics_before.get(metric_key)
                after = rec.metrics_after.get(metric_key)
                if before is None and after is None:
                    continue

                # Format values
                if fmt == "pct":
                    before_str = f"{before*100:.1f}%" if isinstance(before, (int, float)) else "?"
                    after_str = f"{after*100:.1f}%" if isinstance(after, (int, float)) else "?"
                elif fmt == "f3":
                    before_str = f"{before:.3f}" if isinstance(before, (int, float)) else "?"
                    after_str = f"{after:.3f}" if isinstance(after, (int, float)) else "?"
                else:
                    before_str = str(before) if before is not None else "?"
                    after_str = str(after) if after is not None else "?"

                # Target info from config
                target_cfg = getattr(self.config.targets, metric_key, None)
                if target_cfg and target_cfg.mode == "target" and target_cfg.value is not None:
                    target_str = f" (target: ~{target_cfg.value})"
                elif target_cfg and target_cfg.mode == "maximize":
                    target_str = " (goal: maximize)"
                else:
                    target_str = ""

                lines.append(f"  {display_label}: {before_str} → {after_str}{target_str}")

        # Tool call summary
        if rec.tool_results:
            # Check for truncation marker
            truncated_entry = next(
                (tr for tr in rec.tool_results if tr.get("tool_name") == "_truncated"), None
            )
            # Count only real tool calls (exclude truncation marker)
            real_results = [tr for tr in rec.tool_results if tr.get("tool_name") != "_truncated"]
            num_calls = len(real_results)
            num_success = sum(1 for tr in real_results if tr.get("result", {}).get("success", False))
            lines.append(f"Tool results: {num_calls} calls, {num_success} succeeded, {rec.num_tiles_changed} tiles changed")
            if truncated_entry:
                lines.append(f"WARNING: {truncated_entry['result']['errors'][0]}")

        return "\n".join(lines)

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        """Build multi-turn messages array from conversation history + current prompt.

        When conversation_length <= 1, returns a single user message (current behavior).
        Otherwise, includes filtered history as alternating user/assistant pairs,
        and prepends feedback about the last step's outcome to the current prompt.

        Args:
            prompt: The current step's prompt.

        Returns:
            List of message dicts with "role" and "content" keys.
        """
        if self.conversation_length <= 1 or not self._conversation_history:
            return [{"role": "user", "content": prompt}]

        # Filter history based on conversation_filter
        if self.conversation_filter == "accepted":
            filtered = [e for e in self._conversation_history if e.accepted]
        else:
            filtered = list(self._conversation_history)

        # Keep last (conversation_length - 1) entries for history pairs
        max_history = self.conversation_length - 1
        history_entries = filtered[-max_history:]

        # Build messages: alternating user/assistant pairs from history
        messages = []
        for entry in history_entries:
            messages.append({"role": "user", "content": entry.prompt})
            messages.append({"role": "assistant", "content": entry.raw_response})

        # Build rich feedback about the most recent step (always from the actual
        # last entry, not the filtered list, so the LLM knows what just happened)
        last_entry = self._conversation_history[-1]
        feedback = self._build_step_feedback(last_entry)

        messages.append({"role": "user", "content": feedback + "\n\n" + prompt})

        return messages

    def call_llm(self, prompt: str, max_retries: int = 3, timeout: float = 300.0) -> tuple[str, bool, int, int]:
        """Call LLM and get response, with retry and timeout.

        Args:
            prompt: Prompt to send.
            max_retries: Max retry attempts on transient failures.
            timeout: Timeout in seconds for each API call.

        Returns:
            Tuple of (raw response text, truncated flag, input_tokens, output_tokens).
            truncated is True if the response was cut off by max_tokens.
            Token counts are 0 if usage info is unavailable.
        """
        self._init_client()

        messages = self._build_messages(prompt)

        # Debug: log conversation context and feedback prefix
        if self.debug and len(messages) > 1:
            self._write_debug(f"--- CONVERSATION: {len(messages)} messages ({(len(messages)-1)//2} history pairs + current) ---")
            # Log the feedback prefix from the last user message (the full prompt
            # is already logged separately in step(), so only capture the new content)
            last_msg = messages[-1]["content"]
            feedback_end = last_msg.find("\n\nYou are an AI agent")
            if feedback_end > 0:
                self._write_debug(f"--- FEEDBACK PREFIX ---\n{last_msg[:feedback_end]}\n--- END FEEDBACK PREFIX ---\n")
            else:
                self._write_debug(f"--- CURRENT MESSAGE (first 500 chars) ---\n{last_msg[:500]}...\n--- END CURRENT MESSAGE ---\n")

        # OpenAI reasoning models (o1, o3, etc.) require max_completion_tokens
        # instead of max_tokens, and don't support the temperature parameter.
        model_lower = self.config.llm.model.lower()
        is_reasoning_model = any(
            tag in model_lower for tag in ("/o1", "/o3", "/o4", "-o1", "-o3", "-o4")
        )

        if is_reasoning_model:
            api_params = dict(
                model=self.config.llm.model,
                messages=messages,
                max_completion_tokens=self.config.llm.max_tokens,
                timeout=timeout,
            )
        else:
            api_params = dict(
                model=self.config.llm.model,
                messages=messages,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=timeout,
            )

        last_error = None
        for attempt in range(max_retries):
            try:
                if self.config.llm.provider == "google":
                    content, truncated, input_tokens, output_tokens = self._call_google(messages, timeout)
                else:
                    response = self._client.chat.completions.create(**api_params)
                    content = response.choices[0].message.content
                    finish_reason = getattr(response.choices[0], "finish_reason", None)
                    truncated = finish_reason == "length"
                    # Extract token usage
                    input_tokens = 0
                    output_tokens = 0
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        output_tokens = getattr(usage, "completion_tokens", 0) or 0
                return content, truncated, input_tokens, output_tokens
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    print(f"  LLM call attempt {attempt + 1} failed ({e}), retrying in {wait}s...")
                    time.sleep(wait)

        raise last_error

    def _call_google(self, messages: list[dict], timeout: float) -> tuple[str, bool, int, int]:
        """Call Google Genai API and return (content, truncated, input_tokens, output_tokens)."""
        from google.genai import types

        # Convert OpenAI-style messages to Google Genai format
        system_instruction = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])]))
            elif msg["role"] == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=msg["content"])]))

        config = types.GenerateContentConfig(
            max_output_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
            system_instruction=system_instruction,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )

        response = self._client.models.generate_content(
            model=self.config.llm.model,
            contents=contents,
            config=config,
        )

        content = response.text
        # Check for truncation (MAX_TOKENS finish reason)
        truncated = False
        if response.candidates and response.candidates[0].finish_reason:
            truncated = response.candidates[0].finish_reason.name == "MAX_TOKENS"

        input_tokens = 0
        output_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return content, truncated, input_tokens, output_tokens

    def parse_response(self, response: str, truncated: bool = False) -> tuple[StepMessage | ProposeSkillMessage | StopMessage | None, list[str]]:
        """Parse LLM response into message.

        Args:
            response: Raw response text.
            truncated: Whether the response was truncated by max_tokens.

        Returns:
            Tuple of (parsed message or None, list of errors).
        """
        # Extract and repair JSON (use aggressive truncation repair if needed)
        parsed, extract_errors = parse_with_repair(response, truncated=truncated)
        if parsed is None:
            return None, extract_errors

        # Validate schema
        validation_errors = validate_message(parsed)
        if validation_errors:
            return None, validation_errors

        # Parse into typed message
        try:
            message = parse_message(parsed)
            return message, []
        except Exception as e:
            return None, [f"Failed to parse message: {e}"]

    def execute_step(self, message: StepMessage) -> tuple[Level, list[dict], int]:
        """Execute a STEP message using EditManager.

        All edits go through EditManager, which is connected to the PCG benchmark.

        Args:
            message: StepMessage with tool calls.

        Returns:
            Tuple of (new level, tool results, total tiles changed).
        """
        # Create checkpoint before edits
        checkpoint = self.edit_manager.checkpoint()

        tool_results = []
        total_changed = 0

        # Edit tools that modify levels
        edit_tools = {
            "place_wall_segment", "place_tile", "place_single_tile", "place_line", "place_patch",
            "generate_random", "generate_maze", "generate_bsp", "generate_ca",
            "generate_connect", "generate_digger", "generate_wfc",
            "generate_zelda_tile_placing",
            "generate_zelda_random", "generate_zelda_maze", "generate_zelda_bsp",
            "generate_zelda_ca", "generate_zelda_connect", "generate_zelda_digger",
            "generate_zelda_wfc",
        }

        max_calls = self.config.optimizer.max_tool_calls_per_step
        total_requested = len(message.tool_calls)
        for i, tc in enumerate(message.tool_calls[:max_calls]):
            result = self.registry.execute(tc.tool_name, **tc.parameters)
            tool_results.append({
                "tool_name": tc.tool_name,
                "parameters": tc.parameters,
                "result": result.to_dict(),
            })

            # Track changes from edit tools
            if tc.tool_name in edit_tools and result.success:
                num_changed = result.data.get("num_tiles_changed", 0)
                total_changed += num_changed

        # Add truncation notice if tool calls were dropped
        if total_requested > max_calls:
            tool_results.append({
                "tool_name": "_truncated",
                "parameters": {},
                "result": {
                    "success": False,
                    "data": {},
                    "errors": [
                        f"Tool calls truncated: {total_requested} requested but limit is {max_calls} per step. "
                        f"Only the first {max_calls} were executed, {total_requested - max_calls} were dropped."
                    ],
                },
            })

        # Get the working level after all edits
        working_level = self.edit_manager.level

        # Rollback to checkpoint (actual acceptance happens in step())
        self.edit_manager.rollback(checkpoint)

        return working_level, tool_results, total_changed

    def decide_accept(self, new_result: EvalResult, num_changes: int) -> tuple[bool, str]:
        """Decide whether to accept new level.

        Args:
            new_result: Evaluation of new level.
            num_changes: Number of tiles changed.

        Returns:
            Tuple of (accept, reason).
        """
        new_score = self.scorer.score(new_result)

        # Never accept no-op steps (all tool calls failed, nothing changed)
        if num_changes == 0:
            return False, f"rejected_no_changes (score: {self.current_score:.2f} -> {new_score:.2f})"

        # Always accept if better
        if new_score > self.current_score:
            return True, f"improved (score: {self.current_score:.2f} -> {new_score:.2f})"

        # Simulated annealing: accept worse moves with decreasing probability
        if self.config.optimizer.strategy == "annealing":
            T = self.sa_initial_temp * (self.sa_decay_rate ** self.step_count)
            if T > 0:
                p = math.exp((new_score - self.current_score) / T)
                if random.random() < p:
                    return True, (f"annealing (T={T:.4f}, p={p:.4f}, "
                                  f"score: {self.current_score:.2f} -> {new_score:.2f})")
            return False, f"rejected (score: {self.current_score:.2f} -> {new_score:.2f})"

        # Greedy: accept worse with probability epsilon
        if random.random() < self.epsilon:
            return True, f"exploration (epsilon={self.epsilon}, score: {self.current_score:.2f} -> {new_score:.2f})"

        return False, f"rejected (score: {self.current_score:.2f} -> {new_score:.2f})"

    def step(self) -> StepRecord:
        """Execute one optimization step.

        Returns:
            StepRecord with step details.
        """
        self.step_count += 1
        record = StepRecord(
            step_num=self.step_count,
            message_type="",
            level_before=self.current_level.to_string(),
            metrics_before=self.current_result.metrics.copy(),
            score_before=self.current_score,
        )

        message = None  # Track parsed message for later processing

        # Build and send prompt
        prompt = self.build_prompt()

        # Debug: log the prompt
        self._write_debug(f"{'='*80}")
        self._write_debug(f"STEP {self.step_count}")
        self._write_debug(f"{'='*80}")
        self._write_debug(f"\n--- PROMPT ---\n{prompt}\n--- END PROMPT ---\n")

        truncated = False
        try:
            raw_response, truncated, input_tokens, output_tokens = self.call_llm(prompt)
            record.raw_response = raw_response
            record.input_tokens = input_tokens
            record.output_tokens = output_tokens
        except Exception as e:
            record.errors.append(f"LLM call failed: {e}")
            record.message_type = "ERROR"
            self._write_debug(f"--- LLM ERROR ---\n{e}\n--- END LLM ERROR ---\n")
            # Fall through to logging

        # Parse response (only if LLM call succeeded)
        if not record.errors:
            # Debug: log the raw response
            truncation_note = " [TRUNCATED by max_tokens]" if truncated else ""
            self._write_debug(f"--- RAW RESPONSE{truncation_note} ---\n{raw_response}\n--- END RAW RESPONSE ---\n")

            record.truncated = truncated
            if truncated:
                # Try to salvage truncated response via aggressive repair
                message, parse_errors = self.parse_response(raw_response, truncated=True)
            else:
                message, parse_errors = self.parse_response(raw_response)
            if message is None:
                if truncated:
                    parse_errors.append("Response was truncated by max_tokens limit")
                record.errors.extend(parse_errors)
                record.message_type = "PARSE_ERROR"
                self._write_debug(f"--- PARSE_ERROR ---\nErrors: {parse_errors}\n--- END PARSE_ERROR ---\n")
                # Fall through to logging
            else:
                record.message_type = message.type.value
                record.parsed_message = message.to_dict()
                self._write_debug(f"--- PARSED OK --- type={record.message_type}\n")

        # Handle message type (only if parsing succeeded)
        if message is not None:
            if isinstance(message, StepMessage):
                # Execute tools (with rollback)
                new_level, tool_results, num_changed = self.execute_step(message)
                record.tool_results = tool_results
                record.num_tiles_changed = num_changed

                # Evaluate new level
                new_result = self.evaluator.evaluate(new_level)

                # Inject change_ratio for scoring penalty (all problems)
                import numpy as np
                total = self.current_level.data.size
                if total > 0:
                    diff = int(np.sum(new_level.data != self.current_level.data))
                    new_result.metrics["change_ratio"] = diff / total

                record.level_after = new_level.to_string()
                record.metrics_after = new_result.metrics.copy()
                record.score_after = self.scorer.score(new_result)

                # Decide acceptance
                accept, reason = self.decide_accept(new_result, num_changed)
                record.accepted = accept
                record.accept_reason = reason

                if accept:
                    # Apply changes via EditManager
                    self.edit_manager.set_level(new_level)
                    self.current_result = new_result
                    self.current_score = record.score_after
                    self.termination.add_changes(num_changed)

            elif isinstance(message, ProposeSkillMessage):
                # Handle skill proposal
                record.accepted = True
                record.accept_reason = "skill_proposed"
                # Save skill spec to disk
                if self.skill_manager:
                    self.skill_manager.propose(message.skill_spec, message.rationale)
                # Also log to trace
                if self.logger:
                    self.logger.log_skill_proposal(message.skill_spec)

            elif isinstance(message, StopMessage):
                # Agent wants to stop
                record.accepted = True
                record.accept_reason = "agent_stop"

        # Always log step (success, error, or parse failure)
        if self.logger:
            self.logger.log_step(record)

        # Print step progress if verbose
        if self.verbose:
            self._print_step(record)

        self.step_records.append(record)

        # Store conversation history entry (skip ERROR steps where LLM call
        # failed entirely since there's no response to include)
        if record.message_type != "ERROR" and record.raw_response:
            self._conversation_history.append(ConversationEntry(
                prompt=prompt,
                raw_response=record.raw_response,
                accepted=record.accepted,
                accept_reason=record.accept_reason,
                message_type=record.message_type,
                step_record=record,
            ))

        return record

    def _print_step(self, record: StepRecord) -> None:
        """Print step progress to terminal.

        Args:
            record: StepRecord with step details.
        """
        # Header
        print(f"\n{'='*60}")
        token_info = f" | Tokens: {record.input_tokens}in/{record.output_tokens}out" if record.input_tokens or record.output_tokens else ""
        print(f"Step {record.step_num} | Type: {record.message_type}{token_info}")
        print(f"{'='*60}")

        # Show rationale from parsed message
        if record.parsed_message:
            rationale = record.parsed_message.get("rationale", "")
            if rationale:
                # Truncate if too long
                if len(rationale) > 200:
                    rationale = rationale[:200] + "..."
                print(f"\nRationale: {rationale}")

        # Show tool calls
        if record.tool_results:
            print(f"\nTool Calls ({len(record.tool_results)}):")
            for i, tr in enumerate(record.tool_results):
                tool_name = tr.get("tool_name", "unknown")
                params = tr.get("parameters", {})
                result = tr.get("result", {})
                success = result.get("success", False)
                status = "OK" if success else "FAIL"

                # Format params concisely
                if tool_name == "place_wall_segment":
                    mode = params.get("mode", "?")
                    tile = params.get("tile_type", "wall")
                    if mode == "start_dir_len":
                        params_str = f"({params.get('start_y')},{params.get('start_x')}) {params.get('direction')} x{params.get('length')} [{tile}]"
                    else:
                        params_str = f"({params.get('start_y')},{params.get('start_x')}) -> ({params.get('end_y')},{params.get('end_x')}) [{tile}]"
                    num_changed = result.get("data", {}).get("num_tiles_changed", 0)
                    print(f"  {i+1}. {tool_name}: {params_str} -> {status} ({num_changed} tiles)")
                elif tool_name == "place_tile":
                    mode = params.get("mode", "?")
                    tile = params.get("tile_type", "?")
                    if mode == "single":
                        params_str = f"({params.get('y')},{params.get('x')}) [{tile}]"
                    elif mode == "line":
                        if params.get("direction"):
                            params_str = f"({params.get('y')},{params.get('x')}) {params.get('direction')} x{params.get('length')} [{tile}]"
                        else:
                            params_str = f"({params.get('y')},{params.get('x')}) -> ({params.get('end_y')},{params.get('end_x')}) [{tile}]"
                    else:
                        params_str = f"({params.get('y')},{params.get('x')}) -> ({params.get('end_y')},{params.get('end_x')}) [{tile}] rect"
                    num_changed = result.get("data", {}).get("num_tiles_changed", 0)
                    print(f"  {i+1}. {tool_name}: {params_str} -> {status} ({num_changed} tiles)")
                elif tool_name == "place_single_tile":
                    tile = params.get("tile_type", "?")
                    params_str = f"({params.get('y')},{params.get('x')}) [{tile}]"
                    num_changed = result.get("data", {}).get("num_tiles_changed", 0)
                    print(f"  {i+1}. {tool_name}: {params_str} -> {status} ({num_changed} tiles)")
                elif tool_name == "place_line":
                    tile = params.get("tile_type", "?")
                    if params.get("direction"):
                        params_str = f"({params.get('y')},{params.get('x')}) {params.get('direction')} x{params.get('length')} [{tile}]"
                    else:
                        params_str = f"({params.get('y')},{params.get('x')}) -> ({params.get('end_y')},{params.get('end_x')}) [{tile}]"
                    num_changed = result.get("data", {}).get("num_tiles_changed", 0)
                    print(f"  {i+1}. {tool_name}: {params_str} -> {status} ({num_changed} tiles)")
                elif tool_name == "place_patch":
                    tile = params.get("tile_type", "?")
                    filled = "filled" if params.get("filled", True) else "border"
                    params_str = f"({params.get('y')},{params.get('x')}) -> ({params.get('end_y')},{params.get('end_x')}) [{tile}] {filled}"
                    num_changed = result.get("data", {}).get("num_tiles_changed", 0)
                    print(f"  {i+1}. {tool_name}: {params_str} -> {status} ({num_changed} tiles)")
                elif tool_name == "calculate_stats":
                    metrics = result.get("data", {}).get("metrics", {})
                    if self.problem_type == 'zelda':
                        players = metrics.get("players", "?")
                        keys = metrics.get("keys", "?")
                        doors = metrics.get("doors", "?")
                        sol_len = metrics.get("solution_length", "?")
                        print(f"  {i+1}. {tool_name}: P={players}, K={keys}, D={doors}, sol_len={sol_len}")
                    elif self.problem_type == 'sokoban':
                        players = metrics.get("players", "?")
                        crates = metrics.get("crates", "?")
                        targets = metrics.get("targets", "?")
                        heuristic = metrics.get("heuristic", "?")
                        sol_len = metrics.get("solution_length", "?")
                        solvable = "Y" if heuristic is not None and heuristic >= 0 else "N"
                        print(f"  {i+1}. {tool_name}: P={players}, crates={crates}, targets={targets}, solvable={solvable}, sol_len={sol_len}")
                    elif self.problem_type == 'loderunner':
                        player = metrics.get("player", "?")
                        gold = metrics.get("gold", "?")
                        collected = metrics.get("collected_gold", "?")
                        print(f"  {i+1}. {tool_name}: P={player}, gold={gold}, collected={collected}")
                    elif self.problem_type == 'smb':
                        complete = metrics.get("complete", 0)
                        enemies = metrics.get("enemies", "?")
                        complete_pct = f"{complete*100:.0f}%" if isinstance(complete, (int, float)) else "?"
                        print(f"  {i+1}. {tool_name}: complete={complete_pct}, enemies={enemies}")
                    elif self.problem_type == 'binarydoor':
                        door_path = metrics.get("door_path", "?")
                        regions = metrics.get("num_connected_regions", "?")
                        print(f"  {i+1}. {tool_name}: door_path={door_path}, regions={regions}")
                    else:
                        path = metrics.get("path", "?")
                        regions = metrics.get("num_connected_regions", "?")
                        print(f"  {i+1}. {tool_name}: path={path}, regions={regions}")
                else:
                    print(f"  {i+1}. {tool_name}: {status}")

        # Show metrics change
        if record.message_type == "STEP":
            print(f"\nMetrics:")

            if self.problem_type == 'zelda':
                # Zelda metrics
                for metric in ["players", "keys", "doors", "enemies"]:
                    before = record.metrics_before.get(metric, "?")
                    after = record.metrics_after.get(metric, "?")
                    print(f"  {metric}: {before} -> {after}")

                sol_before = record.metrics_before.get("solution_length", "?")
                sol_after = record.metrics_after.get("solution_length", "?")
                sol_delta = ""
                if isinstance(sol_before, (int, float)) and isinstance(sol_after, (int, float)):
                    delta = sol_after - sol_before
                    sol_delta = f" ({'+' if delta >= 0 else ''}{delta})"
                print(f"  solution_length: {sol_before} -> {sol_after}{sol_delta}")

                regions_before = record.metrics_before.get("regions", "?")
                regions_after = record.metrics_after.get("regions", "?")
                print(f"  regions: {regions_before} -> {regions_after}")
            elif self.problem_type == 'sokoban':
                # Sokoban metrics
                for metric in ["players", "crates", "targets"]:
                    before = record.metrics_before.get(metric, "?")
                    after = record.metrics_after.get(metric, "?")
                    print(f"  {metric}: {before} -> {after}")

                heur_before = record.metrics_before.get("heuristic", -1)
                heur_after = record.metrics_after.get("heuristic", -1)
                solv_before = "Y" if heur_before is not None and heur_before >= 0 else "N"
                solv_after = "Y" if heur_after is not None and heur_after >= 0 else "N"
                print(f"  solvable: {solv_before} -> {solv_after}")

                sol_before = record.metrics_before.get("solution_length", "?")
                sol_after = record.metrics_after.get("solution_length", "?")
                sol_delta = ""
                if isinstance(sol_before, (int, float)) and isinstance(sol_after, (int, float)):
                    delta = sol_after - sol_before
                    sol_delta = f" ({'+' if delta >= 0 else ''}{delta})"
                print(f"  solution_length: {sol_before} -> {sol_after}{sol_delta}")
            elif self.problem_type == 'loderunner':
                # Lode Runner metrics
                for metric in ["player", "gold", "enemy"]:
                    before = record.metrics_before.get(metric, "?")
                    after = record.metrics_after.get(metric, "?")
                    print(f"  {metric}: {before} -> {after}")

                col_before = record.metrics_before.get("collected_gold", "?")
                col_after = record.metrics_after.get("collected_gold", "?")
                print(f"  collected_gold: {col_before} -> {col_after}")
            elif self.problem_type == 'smb':
                # SMB metrics
                comp_before = record.metrics_before.get("complete", 0)
                comp_after = record.metrics_after.get("complete", 0)
                comp_before_str = f"{comp_before*100:.0f}%" if isinstance(comp_before, (int, float)) else "?"
                comp_after_str = f"{comp_after*100:.0f}%" if isinstance(comp_after, (int, float)) else "?"
                print(f"  complete: {comp_before_str} -> {comp_after_str}")

                for metric in ["enemies", "coins", "jumps"]:
                    before = record.metrics_before.get(metric, "?")
                    after = record.metrics_after.get(metric, "?")
                    print(f"  {metric}: {before} -> {after}")

                tube_before = record.metrics_before.get("tube_issues", "?")
                tube_after = record.metrics_after.get("tube_issues", "?")
                print(f"  tube_issues: {tube_before} -> {tube_after}")
            elif self.problem_type == 'binarydoor':
                # BinaryDoor metrics
                dp_before = record.metrics_before.get("door_path", "?")
                dp_after = record.metrics_after.get("door_path", "?")
                regions_before = record.metrics_before.get("num_connected_regions", "?")
                regions_after = record.metrics_after.get("num_connected_regions", "?")

                dp_delta = ""
                if isinstance(dp_before, (int, float)) and isinstance(dp_after, (int, float)):
                    delta = dp_after - dp_before
                    dp_delta = f" ({'+' if delta >= 0 else ''}{delta})"

                regions_delta = ""
                if isinstance(regions_before, (int, float)) and isinstance(regions_after, (int, float)):
                    delta = regions_after - regions_before
                    regions_delta = f" ({'+' if delta >= 0 else ''}{delta})"

                print(f"  DoorPath: {dp_before} -> {dp_after}{dp_delta}")
                print(f"  Regions:  {regions_before} -> {regions_after}{regions_delta}")
            else:
                # Binary metrics
                path_before = record.metrics_before.get("path", "?")
                path_after = record.metrics_after.get("path", "?")
                regions_before = record.metrics_before.get("num_connected_regions", "?")
                regions_after = record.metrics_after.get("num_connected_regions", "?")

                path_delta = ""
                if isinstance(path_before, (int, float)) and isinstance(path_after, (int, float)):
                    delta = path_after - path_before
                    path_delta = f" ({'+' if delta >= 0 else ''}{delta})"

                regions_delta = ""
                if isinstance(regions_before, (int, float)) and isinstance(regions_after, (int, float)):
                    delta = regions_after - regions_before
                    regions_delta = f" ({'+' if delta >= 0 else ''}{delta})"

                print(f"  Path:    {path_before} -> {path_after}{path_delta}")
                print(f"  Regions: {regions_before} -> {regions_after}{regions_delta}")

            print(f"  Score:   {record.score_before:.2f} -> {record.score_after:.2f}")

        # Show acceptance
        if record.message_type == "STEP":
            status_icon = "ACCEPTED" if record.accepted else "REJECTED"
            print(f"\nDecision: {status_icon} - {record.accept_reason}")

        # Show termination status
        term_status = self.termination.status()
        pct = term_status['budget_percent_used']
        print(f"\nBudget: {term_status['total_changes']}/{term_status['change_budget']} ({pct:.1f}%)")

        # Show errors if any
        if record.errors:
            print(f"\nErrors: {record.errors}")

        # Debug: show raw LLM response in terminal
        if self.debug and record.raw_response:
            print(f"\n--- DEBUG: Raw LLM Response ---")
            print(record.raw_response[:2000])
            if len(record.raw_response) > 2000:
                print(f"... ({len(record.raw_response)} chars total, truncated)")
            print(f"--- END Raw LLM Response ---")

    def run(self, max_steps: int = 100, max_consecutive_errors: int = 5) -> dict:
        """Run optimization loop.

        Args:
            max_steps: Maximum steps to run.
            max_consecutive_errors: Stop after this many consecutive LLM/parse errors.

        Returns:
            Summary dictionary.
        """
        if self.edit_manager is None:
            raise RuntimeError("Optimizer not initialized. Call initialize() first.")

        consecutive_errors = 0

        remaining_steps = max_steps - self.step_count
        for _ in range(remaining_steps):
            # Check termination
            if self.termination.should_terminate():
                break

            # Execute step
            record = self.step()

            # Track consecutive errors for circuit breaker
            # Truncation-caused parse errors don't count - they indicate
            # output length issues, not API failures
            if record.message_type in ("ERROR", "PARSE_ERROR"):
                if record.message_type == "PARSE_ERROR" and record.truncated:
                    # Truncated response - log but don't count toward circuit breaker
                    if self.verbose:
                        print(f"  (truncated response, not counting toward error limit)")
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"\nStopping: {consecutive_errors} consecutive errors.")
                        break
            else:
                consecutive_errors = 0

            # Check for STOP message
            if record.message_type == "STOP":
                break

        # Close debug log file
        if self._debug_file:
            self._debug_file.close()
            self._debug_file = None

        # Build summary
        total_input_tokens = sum(r.input_tokens for r in self.step_records)
        total_output_tokens = sum(r.output_tokens for r in self.step_records)
        summary = {
            "total_steps": self.step_count,
            "final_score": self.current_score,
            "final_metrics": self.current_result.metrics if self.current_result else {},
            "termination_status": self.termination.status(),
            "accepted_steps": sum(1 for r in self.step_records if r.accepted),
            "rejected_steps": sum(1 for r in self.step_records if not r.accepted and r.message_type == "STEP"),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }

        # Log final state
        if self.logger:
            self.logger.log_final_level(self.current_level, self.current_result)
            self.logger.log_summary(summary)

        return summary
