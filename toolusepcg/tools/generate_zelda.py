"""Constructive generator tools for Zelda levels.

Each tool wraps a generator from submodules/ConstructiveGenerators/generators/zelda/.
Zelda generator tiles (solid=0, empty=1, player=2, key=3, door=4, enemy=5)
match ToolUsePCG's ZeldaTiles enum — no tile conversion needed.
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

from generators.zelda.tileplacing_generator import TilePlacingGenerator


class _ZeldaGeneratorTool(EditTool):
    """Base class for Zelda constructive generator tools.

    Subclasses define generator-specific name, description, parameters,
    and instantiation. This base handles diffing and applying changes
    through EditManager. No tile conversion is needed since generator
    tile values match ZeldaTiles.
    """

    @abstractmethod
    def _create_generator(self, **kwargs) -> Any:
        """Create and return a generator instance from validated kwargs."""

    def execute(self, **kwargs) -> ToolResult:
        level_data = self._edit_manager.level_data

        try:
            generator = self._create_generator(**kwargs)
            gen_output = np.asarray(generator.generate(level_data.copy()), dtype=int)
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

        # Find changed positions and group by tile value
        changed_mask = gen_output != level_data
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


class GenerateZeldaTilePlacingTool(_ZeldaGeneratorTool):
    """Tile-placing Zelda level generator."""

    @property
    def name(self) -> str:
        return "generate_zelda_tile_placing"

    @property
    def description(self) -> str:
        return (
            "Ignores the current level and generates a completely new random "
            "Zelda layout using tile placing. Places wall segments and enemies "
            "randomly across the grid, then positions the player, key, and door "
            "in remaining empty spaces. Wall probability is ~20% with random-length "
            "segments, enemy probability is ~5%. Output has all required Zelda "
            "entities (player, key, door) and a mix of walls and enemies. Best "
            "used as a starting point for further refinement with place_tile. "
            "Calling this again discards all previous progress."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def _create_generator(self, **kwargs) -> Any:
        return TilePlacingGenerator()
