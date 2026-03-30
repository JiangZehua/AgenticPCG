"""Constructive generator tools for binary maze levels.

Each tool wraps a generator from submodules/ConstructiveGenerators/generators/binary/.
Generators use solid=0/empty=1; ToolUsePCG uses EMPTY=0/WALL=1. Conversion: 1 - value.
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


class _BinaryGeneratorTool(EditTool):
    """Base class for binary constructive generator tools.

    Subclasses define generator-specific name, description, parameters,
    and instantiation. This base handles tile-value conversion, diffing,
    and applying changes through EditManager.
    """

    @abstractmethod
    def _create_generator(self, **kwargs) -> Any:
        """Create and return a generator instance from validated kwargs."""

    def execute(self, **kwargs) -> ToolResult:
        # Get current level in generator coordinate system (solid=0, empty=1)
        level_data = self._edit_manager.level_data  # EMPTY=0, WALL=1
        gen_input = 1 - level_data  # -> solid=0, empty=1

        try:
            generator = self._create_generator(**kwargs)
            gen_output = np.asarray(generator.generate(gen_input.copy()), dtype=int)
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

        # Convert back: solid=0->WALL=1, empty=1->EMPTY=0
        new_data = 1 - gen_output

        # Find changed positions and group by tile value
        changed_mask = new_data != level_data
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
            val = int(new_data[y, x])
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


class GenerateRandomTool(_BinaryGeneratorTool):
    """Random binary level generator."""

    @property
    def name(self) -> str:
        return "generate_random"

    @property
    def description(self) -> str:
        return (
            "Ignores the current level and generates a completely new random "
            "binary layout. Each tile is independently set to wall with "
            "probability solid_prob or empty with probability 1-solid_prob. "
            "Works regardless of the current level state. Output is noisy with "
            "no structural guarantees — likely many disconnected regions and no "
            "clear path. Best used as a starting point, then follow up with "
            "generate_ca to smooth and generate_connect to ensure connectivity. "
            "Calling this again discards all previous progress."
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


class GenerateMazeTool(_BinaryGeneratorTool):
    """Depth-first search maze generator."""

    @property
    def name(self) -> str:
        return "generate_maze"

    @property
    def description(self) -> str:
        return (
            "Carves a perfect maze into the WALL tiles of the current level "
            "using depth-first search. Only converts wall tiles to empty to "
            "create corridors — does not add walls. IMPORTANT: the current "
            "level must contain wall tiles for this tool to work; on an "
            "all-empty level it has no effect. Output has long, winding "
            "single-tile-wide corridors with no loops — every pair of cells is "
            "connected by exactly one path. Typically produces a single "
            "connected region with high path lengths. Use place_tile to fill "
            "the level with walls first if the level is currently empty, or "
            "use generate_random with high solid_prob. Calling this again "
            "re-carves the current level's walls into a different random maze."
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


class GenerateBSPTool(_BinaryGeneratorTool):
    """Binary Space Partitioning room-and-corridor generator."""

    @property
    def name(self) -> str:
        return "generate_bsp"

    @property
    def description(self) -> str:
        return (
            "Ignores the current level and generates a new room-and-corridor "
            "layout using Binary Space Partitioning. Recursively splits the "
            "space into rectangles, carves a room inside each partition, and "
            "connects adjacent rooms with corridors. Works regardless of the "
            "current level state. Output has open rectangular rooms linked by "
            "narrow passages — typically 1 connected region with moderate path "
            "lengths. Increase splits for more, smaller rooms (longer paths); "
            "decrease for fewer, larger rooms (shorter paths). Calling this "
            "again discards all previous progress."
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
            # Note: upstream parameter has typo "min_wdith"
            min_wdith=kwargs.get("min_width", 5),
            min_height=kwargs.get("min_height", 5),
        )


class GenerateCATool(_BinaryGeneratorTool):
    """Cellular automata cave generator."""

    @property
    def name(self) -> str:
        return "generate_ca"

    @property
    def description(self) -> str:
        return (
            "REFINEMENT tool: Refines the CURRENT level in-place using cellular automata — does "
            "NOT discard previous work. Smooths noisy layouts into organic "
            "cave-like shapes: isolated wall tiles in mostly-empty areas get "
            "removed, isolated empty tiles in mostly-wall areas get filled, and "
            "dense clusters are preserved. The net effect is rounding rough "
            "edges into smooth blobs. IMPORTANT: has NO effect on all-empty or "
            "all-wall levels (both are fixed points) — needs a mix of wall and "
            "empty tiles to work. Best used after generate_random to smooth "
            "noisy output into caves. Fewer iterations = subtle smoothing; more "
            "iterations = stronger smoothing into larger blobs."
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


class GenerateConnectTool(_BinaryGeneratorTool):
    """Region connector that removes small regions and links the rest."""

    @property
    def name(self) -> str:
        return "generate_connect"

    @property
    def description(self) -> str:
        return (
            "REFINEMENT tool: post-processes the CURRENT level in-place to fix "
            "connectivity — does NOT discard previous work. Finds all connected "
            "empty regions, removes regions smaller than smallest_region_size "
            "(fills them with wall), then connects remaining regions with "
            "straight corridors. Effect: reduces num_connected_regions toward 1 "
            "and increases path length by linking previously isolated areas. "
            "IMPORTANT: has no effect on a blank (all-empty) level since it is "
            "already one region. Best used as a final pass after any generator "
            "(especially generate_random, generate_ca, or generate_digger) to "
            "ensure the level is fully connected. Can also be called after "
            "manual place_tile edits that may have split the level."
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


class GenerateDiggerTool(_BinaryGeneratorTool):
    """Random-walk digger cave generator."""

    @property
    def name(self) -> str:
        return "generate_digger"

    @property
    def description(self) -> str:
        return (
            "Ignores the current level and generates a new cave layout using a "
            "random-walk digger. Starts from an all-solid grid at a random "
            "position and carves empty tiles by walking in random directions, "
            "occasionally carving rooms. Stops when the fraction of empty tiles "
            "reaches stop_size. Works regardless of the current level state. "
            "Output is a single naturally-connected cave with organic, irregular "
            "shape — guaranteed 1 connected region. Higher stop_size = more open "
            "space (shorter paths); lower stop_size = tighter tunnels (longer "
            "paths). Follow up with generate_ca to smooth rough edges. Calling "
            "this again discards all previous progress."
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


class GenerateWFCTool(_BinaryGeneratorTool):
    """Wave Function Collapse generator from a reference maze image."""

    @property
    def name(self) -> str:
        return "generate_wfc"

    @property
    def description(self) -> str:
        return (
            "Overwrites the current level using Wave Function Collapse from a built-in "
            "reference maze image. Learns local patterns from the reference and "
            "produces a level where every local neighborhood is consistent with "
            "those patterns. Output has regular, maze-like corridors and walls. "
            "On success, replaces all tiles (discards previous progress). "
            "Can fail if constraints become unsatisfiable — on failure the "
            "level is left UNCHANGED. Larger pattern_size = more faithful to "
            "reference but more likely to fail; smaller = more variety but less "
            "coherent."
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
