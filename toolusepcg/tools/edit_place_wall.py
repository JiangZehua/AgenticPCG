"""Place wall segment editing tool using EditManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..level import WALL, EMPTY
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


class PlaceWallSegmentTool(EditTool):
    """Tool for placing straight wall/empty segments on the level.

    Uses EditManager for all modifications.
    """

    @property
    def name(self) -> str:
        return "place_wall_segment"

    @property
    def description(self) -> str:
        return (
            "Places a straight line of tiles (wall or empty) on the level. "
            "Supports two modes: 'start_end' (specify start and end coordinates) "
            "or 'start_dir_len' (specify start, direction, and length). "
            "Returns the modified level and diff information."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["start_end", "start_dir_len"],
                    "description": "Mode of operation: 'start_end' or 'start_dir_len'",
                },
                "start_y": {
                    "type": "integer",
                    "description": "Starting row coordinate (0-indexed)",
                },
                "start_x": {
                    "type": "integer",
                    "description": "Starting column coordinate (0-indexed)",
                },
                "end_y": {
                    "type": "integer",
                    "description": "Ending row coordinate (for start_end mode)",
                },
                "end_x": {
                    "type": "integer",
                    "description": "Ending column coordinate (for start_end mode)",
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right", "north", "south", "west", "east"],
                    "description": "Direction to extend (for start_dir_len mode)",
                },
                "length": {
                    "type": "integer",
                    "description": "Number of tiles to place (for start_dir_len mode)",
                },
                "tile_type": {
                    "type": "string",
                    "enum": ["wall", "empty"],
                    "description": "Type of tile to place (default: wall)",
                    "default": "wall",
                },
            },
            "required": ["mode", "start_y", "start_x"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute wall segment placement via EditManager.

        Args:
            mode: 'start_end' or 'start_dir_len'
            start_y: Starting row
            start_x: Starting column
            end_y: Ending row (for start_end)
            end_x: Ending column (for start_end)
            direction: Direction string (for start_dir_len)
            length: Number of tiles (for start_dir_len)
            tile_type: 'wall' or 'empty' (default: 'wall')

        Returns:
            ToolResult with new_level, applied, num_tiles_changed, diff, errors.
        """
        mode = kwargs.get("mode")
        start_y = kwargs.get("start_y")
        start_x = kwargs.get("start_x")
        tile_type = kwargs.get("tile_type", "wall")

        # Validate required parameters
        if mode is None:
            return ToolResult(success=False, errors=["Missing required parameter: mode"])
        if start_y is None:
            return ToolResult(success=False, errors=["Missing required parameter: start_y"])
        if start_x is None:
            return ToolResult(success=False, errors=["Missing required parameter: start_x"])

        # Validate start position
        if not self._edit_manager.validate_position(start_y, start_x):
            return ToolResult(
                success=False,
                errors=[f"Start position ({start_y}, {start_x}) is out of bounds"],
            )

        # Determine tile value
        tile_value = WALL if tile_type == "wall" else EMPTY

        # Get coordinates based on mode
        if mode == "start_end":
            result = self._execute_start_end(
                start_y, start_x,
                kwargs.get("end_y"), kwargs.get("end_x"),
                tile_value,
            )
        elif mode == "start_dir_len":
            result = self._execute_start_dir_len(
                start_y, start_x,
                kwargs.get("direction"), kwargs.get("length"),
                tile_value,
            )
        else:
            return ToolResult(success=False, errors=[f"Invalid mode: {mode}"])

        return result

    def _execute_start_end(
        self,
        start_y: int,
        start_x: int,
        end_y: int | None,
        end_x: int | None,
        tile_value: int,
    ) -> ToolResult:
        """Execute start_end mode using EditManager.set_line()."""
        if end_y is None:
            return ToolResult(success=False, errors=["Missing required parameter for start_end mode: end_y"])
        if end_x is None:
            return ToolResult(success=False, errors=["Missing required parameter for start_end mode: end_x"])

        if not self._edit_manager.validate_position(end_y, end_x):
            return ToolResult(
                success=False,
                errors=[f"End position ({end_y}, {end_x}) is out of bounds"],
            )

        # Use EditManager's set_line
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

    def _execute_start_dir_len(
        self,
        start_y: int,
        start_x: int,
        direction: str | None,
        length: int | None,
        tile_value: int,
    ) -> ToolResult:
        """Execute start_dir_len mode using EditManager.set_tiles()."""
        if direction is None:
            return ToolResult(success=False, errors=["Missing required parameter for start_dir_len mode: direction"])
        if length is None:
            return ToolResult(success=False, errors=["Missing required parameter for start_dir_len mode: length"])

        direction_lower = direction.lower()
        if direction_lower not in DIRECTIONS:
            return ToolResult(
                success=False,
                errors=[f"Invalid direction: {direction}. Valid: {list(DIRECTIONS.keys())}"],
            )

        if length < 1:
            return ToolResult(success=False, errors=["Length must be at least 1"])

        # Generate coordinates
        dy, dx = DIRECTIONS[direction_lower]
        coords = []
        for i in range(length):
            y = start_y + i * dy
            x = start_x + i * dx
            coords.append((y, x))

        # Use EditManager's set_tiles
        edit_result = self._edit_manager.set_tiles(coords, tile_value)

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
