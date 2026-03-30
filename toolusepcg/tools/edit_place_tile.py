"""Place tile editing tool using EditManager - works for all problem types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import EditTool, ToolResult

if TYPE_CHECKING:
    from ..edit_manager import EditManager


# Direction mappings
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


class PlaceTileTool(EditTool):
    """Tool for placing any tile type on the level.

    Works with all problem types (Binary, Zelda, etc.).
    Uses EditManager for all modifications.
    """

    @property
    def name(self) -> str:
        return "place_tile"

    @property
    def description(self) -> str:
        # Build tile types description from EditManager
        if self._edit_manager:
            tile_names = list(self._edit_manager.name_tiles.keys())
            tiles_str = ", ".join(tile_names)
        else:
            tiles_str = "wall, empty, player, key, door, enemy"

        return (
            f"Places a tile on the level. Supported tile types: {tiles_str}. "
            "Supports three modes:\n"
            "- 'single': place one tile at (y, x).\n"
            "- 'line': place a straight horizontal or vertical segment. "
            "Specify the endpoint using ONE of: "
            "(a) direction ('up'/'down'/'left'/'right') + length, "
            "(b) end_y + end_x for an explicit endpoint, "
            "(c) end_x only (end_y defaults to y, drawing a horizontal line), "
            "or (d) end_y only (end_x defaults to x, drawing a vertical line). "
            "Example horizontal line: {mode:'line', y:5, x:0, end_x:10} "
            "Example vertical line: {mode:'line', y:0, x:3, end_y:8}\n"
            "- 'rect': fill a rectangle from (y, x) to (end_y, end_x). "
            "Requires end_y and end_x."
        )

    @property
    def parameters_schema(self) -> dict:
        # Get available tile types from EditManager
        if self._edit_manager:
            tile_types = list(self._edit_manager.name_tiles.keys())
        else:
            # Fallback to all known types
            tile_types = ["wall", "empty", "player", "key", "door", "enemy"]

        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["single", "line", "rect"],
                    "description": "Mode of operation: 'single' for one tile, 'line' for a segment, 'rect' for rectangle",
                },
                "tile_type": {
                    "type": "string",
                    "enum": tile_types,
                    "description": f"Type of tile to place: {', '.join(tile_types)}",
                },
                "y": {
                    "type": "integer",
                    "description": "Row coordinate (0-indexed). For line/rect, this is start_y.",
                },
                "x": {
                    "type": "integer",
                    "description": "Column coordinate (0-indexed). For line/rect, this is start_x.",
                },
                "end_y": {
                    "type": "integer",
                    "description": (
                        "Ending row coordinate. "
                        "Required for rect mode. "
                        "For line mode: if only end_y is given (no end_x), draws a vertical line from y to end_y at column x. "
                        "If both end_y and end_x are given, draws a line from (y,x) to (end_y,end_x)."
                    ),
                },
                "end_x": {
                    "type": "integer",
                    "description": (
                        "Ending column coordinate. "
                        "Required for rect mode. "
                        "For line mode: if only end_x is given (no end_y), draws a horizontal line from x to end_x at row y. "
                        "If both end_y and end_x are given, draws a line from (y,x) to (end_y,end_x)."
                    ),
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right", "north", "south", "west", "east"],
                    "description": "Direction to extend from (y,x) — alternative to end_y/end_x for line mode. Must be paired with 'length'.",
                },
                "length": {
                    "type": "integer",
                    "description": "Number of tiles to place in 'direction' (for line mode with direction). Must be paired with 'direction'.",
                },
                "filled": {
                    "type": "boolean",
                    "description": "For rect mode: if true, fill entire rectangle; if false, only draw border",
                    "default": True,
                },
            },
            "required": ["mode", "tile_type", "y", "x"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute tile placement via EditManager.

        Args:
            mode: 'single', 'line', or 'rect'
            tile_type: Name of tile type (e.g., 'wall', 'empty', 'player')
            y: Row coordinate (start_y for line/rect)
            x: Column coordinate (start_x for line/rect)
            end_y: Ending row (for line/rect with explicit coordinates)
            end_x: Ending column (for line/rect with explicit coordinates)
            direction: Direction string (for line mode)
            length: Number of tiles (for line mode with direction)
            filled: Fill rectangle or just border (for rect mode)

        Returns:
            ToolResult with new_level, applied, num_tiles_changed, diff, errors.
        """
        mode = kwargs.get("mode")
        tile_type = kwargs.get("tile_type")
        y = kwargs.get("y")
        x = kwargs.get("x")

        # Validate required parameters
        if mode is None:
            return ToolResult(success=False, errors=["Missing required parameter: mode"])
        if tile_type is None:
            return ToolResult(success=False, errors=["Missing required parameter: tile_type"])
        if y is None:
            return ToolResult(success=False, errors=["Missing required parameter: y"])
        if x is None:
            return ToolResult(success=False, errors=["Missing required parameter: x"])

        # Validate position
        if not self._edit_manager.validate_position(y, x):
            return ToolResult(
                success=False,
                errors=[f"Position ({y}, {x}) is out of bounds"],
            )

        # Convert tile name to value
        name_tiles = self._edit_manager.name_tiles
        if tile_type not in name_tiles:
            valid_types = list(name_tiles.keys())
            return ToolResult(
                success=False,
                errors=[f"Invalid tile type '{tile_type}'. Valid types: {valid_types}"],
            )
        tile_value = name_tiles[tile_type]

        # Execute based on mode
        if mode == "single":
            return self._execute_single(y, x, tile_value)
        elif mode == "line":
            return self._execute_line(y, x, tile_value, kwargs)
        elif mode == "rect":
            return self._execute_rect(y, x, tile_value, kwargs)
        else:
            return ToolResult(success=False, errors=[f"Invalid mode: {mode}"])

    def _execute_single(self, y: int, x: int, tile_value: int) -> ToolResult:
        """Execute single tile placement."""
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

    def _execute_line(self, start_y: int, start_x: int, tile_value: int, kwargs: dict) -> ToolResult:
        """Execute line segment placement."""
        # Check for direction+length mode
        direction = kwargs.get("direction")
        length = kwargs.get("length")

        if direction is not None and length is not None:
            # Direction + length mode
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
                y = start_y + i * dy
                x = start_x + i * dx
                coords.append((y, x))

            edit_result = self._edit_manager.set_tiles(coords, tile_value)

        else:
            # End coordinates mode
            end_y = kwargs.get("end_y")
            end_x = kwargs.get("end_x")

            # Infer missing coordinate: only end_x → horizontal line (end_y = start_y)
            # only end_y → vertical line (end_x = start_x)
            if end_y is None and end_x is None:
                return ToolResult(
                    success=False,
                    errors=[
                        "For line mode, specify the endpoint using one of: "
                        "(a) direction + length, "
                        "(b) end_y + end_x, "
                        "(c) end_x alone for a horizontal line (end_y defaults to y), "
                        "or (d) end_y alone for a vertical line (end_x defaults to x)."
                    ],
                )
            if end_y is None:
                end_y = start_y  # horizontal line: same row
            if end_x is None:
                end_x = start_x  # vertical line: same column

            edit_result = self._edit_manager.set_line(start_y, start_x, end_y, end_x, tile_value)

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

    def _execute_rect(self, top_y: int, left_x: int, tile_value: int, kwargs: dict) -> ToolResult:
        """Execute rectangle placement."""
        end_y = kwargs.get("end_y")
        end_x = kwargs.get("end_x")
        filled = kwargs.get("filled", True)

        if end_y is None or end_x is None:
            return ToolResult(
                success=False,
                errors=["For rect mode, end_y and end_x are required"],
            )

        edit_result = self._edit_manager.set_rect(top_y, left_x, end_y, end_x, tile_value, filled=filled)

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
