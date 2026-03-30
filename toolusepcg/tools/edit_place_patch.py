"""Place a rectangular patch editing tool using EditManager - works for all problem types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import EditTool, ToolResult

if TYPE_CHECKING:
    from ..edit_manager import EditManager


class PlacePatchTool(EditTool):
    """Tool for placing a rectangular patch (minimum 2x2) on the level.

    Works with all problem types (Binary, Zelda, Sokoban, LodeRunner, SMB).
    Uses EditManager for all modifications.
    """

    @property
    def name(self) -> str:
        return "place_patch"

    @property
    def description(self) -> str:
        if self._edit_manager:
            tile_names = list(self._edit_manager.name_tiles.keys())
            tiles_str = ", ".join(tile_names)
        else:
            tiles_str = "wall, empty, player, key, door, enemy"

        return (
            f"Places a rectangular patch of tiles (minimum 2x2) on the level. "
            f"Supported tile types: {tiles_str}. "
            "Specify the tile type, top-left corner (y, x), and bottom-right corner (end_y, end_x). "
            "Can fill the entire rectangle or just the border. "
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
                    "description": "Top-left row coordinate (0-indexed)",
                },
                "x": {
                    "type": "integer",
                    "description": "Top-left column coordinate (0-indexed)",
                },
                "end_y": {
                    "type": "integer",
                    "description": "Bottom-right row coordinate (0-indexed, inclusive)",
                },
                "end_x": {
                    "type": "integer",
                    "description": "Bottom-right column coordinate (0-indexed, inclusive)",
                },
                "filled": {
                    "type": "boolean",
                    "description": "If true, fill entire rectangle; if false, only draw border",
                    "default": True,
                },
            },
            "required": ["tile_type", "y", "x", "end_y", "end_x"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute rectangular patch placement via EditManager.

        Enforces minimum 2x2 patch size.

        Args:
            tile_type: Name of tile type (e.g., 'wall', 'empty', 'player')
            y: Top-left row coordinate
            x: Top-left column coordinate
            end_y: Bottom-right row coordinate (inclusive)
            end_x: Bottom-right column coordinate (inclusive)
            filled: Fill rectangle or just border (default: True)

        Returns:
            ToolResult with new_level, applied, num_tiles_changed, diff, errors.
        """
        tile_type = kwargs.get("tile_type")
        y = kwargs.get("y")
        x = kwargs.get("x")
        end_y = kwargs.get("end_y")
        end_x = kwargs.get("end_x")
        filled = kwargs.get("filled", True)

        if tile_type is None:
            return ToolResult(success=False, errors=["Missing required parameter: tile_type"])
        if y is None:
            return ToolResult(success=False, errors=["Missing required parameter: y"])
        if x is None:
            return ToolResult(success=False, errors=["Missing required parameter: x"])
        if end_y is None:
            return ToolResult(success=False, errors=["Missing required parameter: end_y"])
        if end_x is None:
            return ToolResult(success=False, errors=["Missing required parameter: end_x"])

        # Normalize so top-left <= bottom-right
        top_y, bot_y = min(y, end_y), max(y, end_y)
        left_x, right_x = min(x, end_x), max(x, end_x)

        # Enforce minimum 2x2 patch size
        height = bot_y - top_y + 1
        width = right_x - left_x + 1
        if height < 2 or width < 2:
            return ToolResult(
                success=False,
                errors=[
                    f"Patch must be at least 2x2, got {height}x{width}. "
                    "Use place_single_tile or place_tile for smaller edits."
                ],
            )

        if not self._edit_manager.validate_position(top_y, left_x):
            return ToolResult(
                success=False,
                errors=[f"Top-left position ({top_y}, {left_x}) is out of bounds"],
            )
        if not self._edit_manager.validate_position(bot_y, right_x):
            return ToolResult(
                success=False,
                errors=[f"Bottom-right position ({bot_y}, {right_x}) is out of bounds"],
            )

        name_tiles = self._edit_manager.name_tiles
        if tile_type not in name_tiles:
            valid_types = list(name_tiles.keys())
            return ToolResult(
                success=False,
                errors=[f"Invalid tile type '{tile_type}'. Valid types: {valid_types}"],
            )
        tile_value = name_tiles[tile_type]

        edit_result = self._edit_manager.set_rect(
            top_y, left_x, bot_y, right_x, tile_value, filled=filled
        )

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
