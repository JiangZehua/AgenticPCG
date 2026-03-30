"""Place a single tile editing tool using EditManager - works for all problem types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import EditTool, ToolResult

if TYPE_CHECKING:
    from ..edit_manager import EditManager


class PlaceSingleTileTool(EditTool):
    """Tool for placing exactly one tile on the level.

    Works with all problem types (Binary, Zelda, Sokoban, LodeRunner, SMB).
    Uses EditManager for all modifications.
    """

    @property
    def name(self) -> str:
        return "place_single_tile"

    @property
    def description(self) -> str:
        if self._edit_manager:
            tile_names = list(self._edit_manager.name_tiles.keys())
            tiles_str = ", ".join(tile_names)
        else:
            tiles_str = "wall, empty, player, key, door, enemy"

        return (
            f"Places exactly one tile on the level. Supported tile types: {tiles_str}. "
            "Specify the tile type and (y, x) coordinate. "
            "Returns the modified level and diff information."
        )

    @property
    def parameters_schema(self) -> dict:
        if self._edit_manager:
            tile_types = list(self._edit_manager.name_tiles.keys())
        else:
            tile_types = ["wall", "empty", "player", "key", "door", "enemy"]

        return {
            "type": "object",
            "properties": {
                "tile_type": {
                    "type": "string",
                    "enum": tile_types,
                    "description": f"Type of tile to place: {', '.join(tile_types)}",
                },
                "y": {
                    "type": "integer",
                    "description": "Row coordinate (0-indexed)",
                },
                "x": {
                    "type": "integer",
                    "description": "Column coordinate (0-indexed)",
                },
            },
            "required": ["tile_type", "y", "x"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute single tile placement via EditManager.

        Args:
            tile_type: Name of tile type (e.g., 'wall', 'empty', 'player')
            y: Row coordinate
            x: Column coordinate

        Returns:
            ToolResult with new_level, applied, num_tiles_changed, diff, errors.
        """
        tile_type = kwargs.get("tile_type")
        y = kwargs.get("y")
        x = kwargs.get("x")

        if tile_type is None:
            return ToolResult(success=False, errors=["Missing required parameter: tile_type"])
        if y is None:
            return ToolResult(success=False, errors=["Missing required parameter: y"])
        if x is None:
            return ToolResult(success=False, errors=["Missing required parameter: x"])

        if not self._edit_manager.validate_position(y, x):
            return ToolResult(
                success=False,
                errors=[f"Position ({y}, {x}) is out of bounds"],
            )

        name_tiles = self._edit_manager.name_tiles
        if tile_type not in name_tiles:
            valid_types = list(name_tiles.keys())
            return ToolResult(
                success=False,
                errors=[f"Invalid tile type '{tile_type}'. Valid types: {valid_types}"],
            )
        tile_value = name_tiles[tile_type]

        edit_result = self._edit_manager.set_tile(y, x, tile_value)

        if not edit_result.success:
            return ToolResult(success=False, errors=edit_result.errors)

        return ToolResult(
            success=True,
            data={
                "new_level": edit_result.new_level.to_string() if edit_result.new_level else "",
                "applied": True,
                "num_tiles_changed": edit_result.num_tiles_changed,
                "diff": edit_result.diff,
            },
        )
