"""Constructive generator tools for Zelda wall layouts.

Wraps binary generators from submodules/ConstructiveGenerators/generators/binary/
to generate wall/empty layouts for Zelda levels. Special tiles (player, key, door,
enemy) are preserved — only wall (0) and empty (1) tiles are affected.

Zelda tiles: WALL=0, EMPTY=1 — matches generator convention (solid=0, empty=1)
directly, so no tile value conversion is needed.
"""

from __future__ import annotations

import sys
import os
from abc import abstractmethod
from typing import Any, TYPE_CHECKING

import numpy as np

from .registry import EditTool, ToolResult

if TYPE_CHECKING:
    from ..edit_manager import EditManager

# Add ConstructiveGenerators to sys.path so we can import generators
_CG_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "submodules", "ConstructiveGenerators")
)
if _CG_ROOT not in sys.path:
    sys.path.insert(0, _CG_ROOT)

from generators.binary.random_generator import RandomGenerator
from generators.binary.maze_generator import MazeGenerator
from generators.binary.bsp_generator import BSPGenerator
from generators.binary.ca_generator import CAGenerator
from generators.binary.connect_generator import ConnectGenerator
from generators.binary.digger_generator import DiggerGenerator
from generators.binary.wfc_generator import WFCGenerator

# Zelda tile values — only WALL and EMPTY are touched by generators
_WALL = 0
_EMPTY = 1


class _ZeldaWallGeneratorTool(EditTool):
    """Base class for Zelda wall-layout generator tools.

    Converts the Zelda level to a binary (wall/empty) grid for the generator,
    then applies only wall/empty changes back, preserving special tiles
    (player, key, door, enemy).
    """

    @abstractmethod
    def _create_generator(self, **kwargs) -> Any:
        """Create and return a generator instance from validated kwargs."""

    def execute(self, **kwargs) -> ToolResult:
        level_data = self._edit_manager.level_data

        # Convert to binary for the generator: special tiles (>=2) become empty (1)
        binary_input = np.where(level_data >= 2, _EMPTY, level_data).astype(int)

        try:
            generator = self._create_generator(**kwargs)
            gen_output = np.asarray(generator.generate(binary_input.copy()), dtype=int)
        except Exception as e:
            return ToolResult(success=False, errors=[f"Generator failed: {e}"])

        if gen_output.shape != level_data.shape:
            return ToolResult(
                success=False,
                errors=[
                    f"Generator output shape {gen_output.shape} != "
                    f"level shape {level_data.shape}"
                ],
            )

        # Only apply changes to tiles that are currently wall or empty.
        # Preserve special tiles (player=2, key=3, door=4, enemy=5).
        is_basic = level_data <= _EMPTY  # True for wall(0) and empty(1)
        changed_mask = is_basic & (gen_output != level_data)

        if not changed_mask.any():
            return ToolResult(
                success=True,
                data={
                    "new_level": self._edit_manager.level.to_string(),
                    "applied": True,
                    "num_tiles_changed": 0,
                    "diff": [],
                },
            )

        # Group changed positions by target tile value
        changes_by_value: dict[int, list[tuple[int, int]]] = {}
        ys, xs = np.where(changed_mask)
        for y, x in zip(ys, xs):
            val = int(gen_output[y, x])
            changes_by_value.setdefault(val, []).append((int(y), int(x)))

        # Apply through EditManager (one set_tiles call per tile value)
        total_changed = 0
        all_diff: list[dict] = []
        all_errors: list[str] = []

        for tile_value, coords in changes_by_value.items():
            result = self._edit_manager.set_tiles(coords, tile_value)
            total_changed += result.num_tiles_changed
            all_diff.extend(result.diff)
            all_errors.extend(result.errors)

        return ToolResult(
            success=True,
            data={
                "new_level": self._edit_manager.level.to_string(),
                "applied": True,
                "num_tiles_changed": total_changed,
                "diff": all_diff,
            },
            errors=all_errors,
        )


# =========================================================================
# Concrete generator tools
# =========================================================================


class GenerateZeldaRandomTool(_ZeldaWallGeneratorTool):
    """Random wall layout generator for Zelda."""

    @property
    def name(self) -> str:
        return "generate_zelda_random"

    @property
    def description(self) -> str:
        return (
            "Generates a random wall/empty layout for the Zelda level. Each "
            "wall/empty tile is independently set to wall with probability "
            "solid_prob or empty with probability 1-solid_prob. Special tiles "
            "(player, key, door, enemy) are preserved. Output is noisy — best "
            "used as a starting point, then follow up with generate_zelda_ca "
            "to smooth and generate_zelda_connect to ensure connectivity."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "solid_prob": {
                    "type": "number",
                    "description": "Probability of each tile being a wall (0.0-1.0, default 0.5)",
                    "default": 0.5,
                },
            },
            "required": [],
        }

    def _create_generator(self, **kwargs) -> Any:
        return RandomGenerator(solid_prob=kwargs.get("solid_prob", 0.5))


class GenerateZeldaMazeTool(_ZeldaWallGeneratorTool):
    """DFS maze generator for Zelda wall layout."""

    @property
    def name(self) -> str:
        return "generate_zelda_maze"

    @property
    def description(self) -> str:
        return (
            "Carves a perfect maze into the wall tiles of the Zelda level "
            "using depth-first search. Only converts wall tiles to empty — "
            "does not add walls. Special tiles (player, key, door, enemy) are "
            "preserved. IMPORTANT: needs wall tiles to carve into; on an "
            "all-empty level it has no effect. Use generate_zelda_random with "
            "high solid_prob first, or place_tile to fill with walls."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def _create_generator(self, **kwargs) -> Any:
        return MazeGenerator()


class GenerateZeldaBSPTool(_ZeldaWallGeneratorTool):
    """BSP room-and-corridor generator for Zelda wall layout."""

    @property
    def name(self) -> str:
        return "generate_zelda_bsp"

    @property
    def description(self) -> str:
        return (
            "Generates a room-and-corridor wall layout for the Zelda level "
            "using Binary Space Partitioning. Recursively splits the space, "
            "carves rooms, and connects them with corridors. Special tiles "
            "(player, key, door, enemy) are preserved. Increase splits for "
            "more, smaller rooms; decrease for fewer, larger rooms."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "splits": {
                    "type": "integer",
                    "description": "Number of recursive splits (default 3)",
                    "default": 3,
                },
                "min_width": {
                    "type": "integer",
                    "description": "Minimum partition width (default 5)",
                    "default": 5,
                },
                "min_height": {
                    "type": "integer",
                    "description": "Minimum partition height (default 5)",
                    "default": 5,
                },
            },
            "required": [],
        }

    def _create_generator(self, **kwargs) -> Any:
        return BSPGenerator(
            splits=kwargs.get("splits", 3),
            min_wdith=kwargs.get("min_width", 5),
            min_height=kwargs.get("min_height", 5),
        )


class GenerateZeldaCATool(_ZeldaWallGeneratorTool):
    """Cellular automata smoother for Zelda wall layout."""

    @property
    def name(self) -> str:
        return "generate_zelda_ca"

    @property
    def description(self) -> str:
        return (
            "REFINEMENT tool: smooths the wall/empty layout of the Zelda "
            "level in-place using cellular automata. Rounds rough edges into "
            "organic cave-like shapes. Special tiles (player, key, door, "
            "enemy) are preserved. Has no effect on all-empty or all-wall "
            "levels. Best used after generate_zelda_random."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "iterations": {
                    "type": "integer",
                    "description": "Number of CA iterations (default 10)",
                    "default": 10,
                },
                "solid_count": {
                    "type": "integer",
                    "description": "Neighbor threshold for becoming/staying wall (default 2)",
                    "default": 2,
                },
                "empty_count": {
                    "type": "integer",
                    "description": "Neighbor threshold for becoming empty (default 6)",
                    "default": 6,
                },
            },
            "required": [],
        }

    def _create_generator(self, **kwargs) -> Any:
        return CAGenerator(
            iterations=kwargs.get("iterations", 10),
            solid_count=kwargs.get("solid_count", 2),
            empty_count=kwargs.get("empty_count", 6),
        )


class GenerateZeldaConnectTool(_ZeldaWallGeneratorTool):
    """Region connector for Zelda wall layout."""

    @property
    def name(self) -> str:
        return "generate_zelda_connect"

    @property
    def description(self) -> str:
        return (
            "REFINEMENT tool: fixes connectivity of the Zelda wall layout "
            "in-place. Removes small empty regions (fills with wall) and "
            "connects remaining regions with corridors. Special tiles "
            "(player, key, door, enemy) are preserved. Best used as a final "
            "pass after other generators to ensure the level is fully "
            "connected."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smallest_region_size": {
                    "type": "integer",
                    "description": "Minimum region size to keep; smaller regions become wall (default 5)",
                    "default": 5,
                },
            },
            "required": [],
        }

    def _create_generator(self, **kwargs) -> Any:
        return ConnectGenerator(
            smallest_region_size=kwargs.get("smallest_region_size", 5),
        )


class GenerateZeldaDiggerTool(_ZeldaWallGeneratorTool):
    """Random-walk digger cave generator for Zelda wall layout."""

    @property
    def name(self) -> str:
        return "generate_zelda_digger"

    @property
    def description(self) -> str:
        return (
            "Generates a cave wall layout for the Zelda level using a "
            "random-walk digger. Starts from an all-solid grid and carves "
            "empty tiles by walking randomly, occasionally carving rooms. "
            "Special tiles (player, key, door, enemy) are preserved. Output "
            "is a naturally-connected cave. Higher stop_size = more open "
            "space; lower = tighter tunnels."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "change_prob": {
                    "type": "number",
                    "description": "Probability of changing walk direction (default 0.15)",
                    "default": 0.15,
                },
                "room_prob": {
                    "type": "number",
                    "description": "Probability of carving a room instead of a single tile (default 0.01)",
                    "default": 0.01,
                },
                "room_size": {
                    "type": "integer",
                    "description": "Half-size of carved rooms (actual size is 2*room_size+1, default 3)",
                    "default": 3,
                },
                "stop_size": {
                    "type": "number",
                    "description": "Stop when empty tile fraction reaches this value (0.0-1.0, default 0.3)",
                    "default": 0.3,
                },
            },
            "required": [],
        }

    def _create_generator(self, **kwargs) -> Any:
        return DiggerGenerator(
            change_prob=kwargs.get("change_prob", 0.15),
            room_prob=kwargs.get("room_prob", 0.01),
            room_size=kwargs.get("room_size", 3),
            stop_size=kwargs.get("stop_size", 0.3),
        )


class GenerateZeldaWFCTool(_ZeldaWallGeneratorTool):
    """Wave Function Collapse generator for Zelda wall layout."""

    @property
    def name(self) -> str:
        return "generate_zelda_wfc"

    @property
    def description(self) -> str:
        return (
            "Generates a wall layout for the Zelda level using Wave Function "
            "Collapse from a built-in reference maze image. Produces regular, "
            "maze-like corridors and walls. Special tiles (player, key, door, "
            "enemy) are preserved. Can fail if constraints become "
            "unsatisfiable — on failure the level is left unchanged."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern_size": {
                    "type": "integer",
                    "description": "Size of overlapping patterns to extract from reference (default 3)",
                    "default": 3,
                },
            },
            "required": [],
        }

    def _create_generator(self, **kwargs) -> Any:
        return WFCGenerator(
            pattern_size=kwargs.get("pattern_size", 3),
        )
