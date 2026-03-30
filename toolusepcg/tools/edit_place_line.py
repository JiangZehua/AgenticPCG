"""Place a line of tiles editing tool using EditManager - works for all problem types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import EditTool, ToolResult

if TYPE_CHECKING:
    from ..edit_manager import EditManager


# Direction mappings (shared with PlaceTileTool)
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
}


class PlaceLineTool(EditTool):
    """Tool for placing a straight line of tiles (horizontal or vertical) on the level.

    Works with all problem types (Binary, Zelda, Sokoban, LodeRunner, SMB).
    Uses EditManager for all modifications.
    """

    @property
    def name(self) -> str:
        return "place_line"

    @property
    def description(self) -> str:
        if self._edit_manager:
            tile_names = list(self._edit_manager.name_tiles.keys())
            tiles_str = ", ".join(tile_names)
        else:
            tiles_str = "wall, empty, player, key, door, enemy"

        return (
            f"Places a straight line of tiles (horizontal or vertical) on the level. "
            f"Supported tile types: {tiles_str}. "
            "Specify the tile type, start position (y, x), and either "
            "(direction + length) or (end_y + end_x) to define the line. "
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
                    "description": "Start row coordinate (0-indexed)",
                },
                "x": {
                    "type": "integer",
                    "description": "Start column coordinate (0-indexed)",
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right", "north", "south", "west", "east"],
                    "description": "Direction to extend the line (alternative to end_y/end_x)",
                },
                "length": {
                    "type": "integer",
                    "description": "Number of tiles to place (used with direction)",
                },
                "end_y": {
                    "type": "integer",
                    "description": "End row coordinate (0-indexed, alternative to direction+length)",
                },
                "end_x": {
                    "type": "integer",
                    "description": "End column coordinate (0-indexed, alternative to direction+length)",
                },
            },
            "required": ["tile_type", "y", "x"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute line placement via EditManager.

        Args:
            tile_type: Name of tile type (e.g., 'wall', 'empty', 'player')
            y: Start row coordinate
            x: Start column coordinate
            direction: Direction string (alternative to end_y/end_x)
            length: Number of tiles (used with direction)
            end_y: End row coordinate (alternative to direction+length)
            end_x: End column coordinate (alternative to direction+length)

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

        # Check for direction+length mode
        direction = kwargs.get("direction")
        length = kwargs.get("length")

        if direction is not None and length is not None:
            direction_lower = direction.lower()
            if direction_lower not in DIRECTIONS:
                return ToolResult(
                    success=False,
                    errors=[f"Invalid direction: {direction}. Valid: {list(DIRECTIONS.keys())}"],
                )
            if length < 1:
                return ToolResult(success=False, errors=["Length must be at least 1"])

            dy, dx = DIRECTIONS[direction_lower]
            coords = []
            for i in range(length):
                cy = y + i * dy
                cx = x + i * dx
                coords.append((cy, cx))

            edit_result = self._edit_manager.set_tiles(coords, tile_value)

        else:
            # End coordinates mode
            end_y = kwargs.get("end_y")
            end_x = kwargs.get("end_x")

            if end_y is None or end_x is None:
                return ToolResult(
                    success=False,
                    errors=["Provide either (direction + length) or (end_y + end_x) to define the line"],
                )

            edit_result = self._edit_manager.set_line(y, x, end_y, end_x, tile_value)

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
