"""EditManager - Central controller for all level modifications.

All edit tools/skills must use EditManager to modify levels.
This ensures:
1. All edits go through a single validated interface
2. Edits are validated against PCG benchmark content_space
3. Changes are tracked centrally
4. The level state is always consistent with PCG benchmark
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .level import Level, WALL, EMPTY, get_tile_mappings, get_problem_type_from_env_name

if TYPE_CHECKING:
    pass


@dataclass
class EditResult:
    """Result of an edit operation."""

    success: bool
    num_tiles_changed: int = 0
    diff: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    new_level: Level | None = None


class EditManager:
    """Central manager for all level modifications.

    All edit tools must use this manager to modify levels.
    Direct Level.set_tile() calls should be avoided.
    """

    def __init__(self, env, level: Level, problem_type: str | None = None):
        """Initialize EditManager.

        Args:
            env: PCG benchmark environment for validation.
            level: Initial level state.
            problem_type: Problem type ('binary' or 'zelda'). Auto-detected if None.
        """
        self._env = env
        self._level = level.copy()
        self._edit_history: list[dict] = []
        self._total_tiles_changed = 0

        # Determine problem type
        if problem_type:
            self._problem_type = problem_type
        else:
            # Try to infer from level or env
            self._problem_type = level.problem_type

        # Update level's problem_type if needed
        self._level.problem_type = self._problem_type

        # Get tile mappings for this problem type
        self._tile_chars, self._char_tiles, self._tile_names, self._name_tiles = get_tile_mappings(self._problem_type)

        # Get valid tile values from content_space
        self._valid_tiles = self._get_valid_tiles()

    @property
    def problem_type(self) -> str:
        """Get problem type."""
        return self._problem_type

    @property
    def tile_names(self) -> dict[int, str]:
        """Get tile-to-name mapping."""
        return self._tile_names

    @property
    def name_tiles(self) -> dict[str, int]:
        """Get name-to-tile mapping."""
        return self._name_tiles

    def _get_valid_tiles(self) -> set[int]:
        """Extract valid tile values from PCG benchmark content_space."""
        # content_space is ArraySpace with IntegerSpace elements
        # The IntegerSpace has _min_value and _max_value attributes
        try:
            # Try the old structure first (content_space._space._n)
            inner_space = self._env.content_space._space
            return set(range(inner_space._n))
        except AttributeError:
            pass

        try:
            # Try the new structure (ArraySpace with _value containing IntegerSpace)
            inner_space = self._env.content_space._value[0][0]
            return set(range(inner_space._min_value, inner_space._max_value))
        except (AttributeError, IndexError):
            pass

        # Fallback based on problem type
        if self._problem_type == 'zelda':
            return {0, 1, 2, 3, 4, 5}
        elif self._problem_type == 'sokoban':
            return {0, 1, 2, 3, 4}
        elif self._problem_type == 'loderunner':
            return {0, 1, 2, 3, 4, 5, 6}
        elif self._problem_type == 'smb':
            return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        return {EMPTY, WALL}

    @property
    def level(self) -> Level:
        """Get current level (read-only copy)."""
        return self._level.copy()

    @property
    def level_data(self) -> np.ndarray:
        """Get current level data (read-only copy)."""
        return self._level.data.copy()

    @property
    def height(self) -> int:
        """Level height."""
        return self._level.height

    @property
    def width(self) -> int:
        """Level width."""
        return self._level.width

    @property
    def total_tiles_changed(self) -> int:
        """Total number of tile changes made."""
        return self._total_tiles_changed

    @property
    def edit_history(self) -> list[dict]:
        """Get edit history."""
        return self._edit_history.copy()

    def set_level(self, level: Level) -> None:
        """Replace current level (used when optimizer rejects changes).

        Args:
            level: New level state.
        """
        self._level = level.copy()
        self._level.problem_type = self._problem_type

    def reset_tracking(self) -> None:
        """Reset edit tracking (history and tile count)."""
        self._edit_history = []
        self._total_tiles_changed = 0

    def validate_tile(self, tile_value: int) -> bool:
        """Check if tile value is valid for this environment.

        Args:
            tile_value: Tile value to check.

        Returns:
            True if valid.
        """
        return tile_value in self._valid_tiles

    def validate_position(self, y: int, x: int) -> bool:
        """Check if position is within bounds.

        Args:
            y: Row coordinate.
            x: Column coordinate.

        Returns:
            True if valid.
        """
        return 0 <= y < self.height and 0 <= x < self.width

    def get_tile(self, y: int, x: int) -> int | None:
        """Get tile value at position.

        Args:
            y: Row coordinate.
            x: Column coordinate.

        Returns:
            Tile value or None if out of bounds.
        """
        if not self.validate_position(y, x):
            return None
        return int(self._level.data[y, x])

    # =========================================================================
    # Primitive Edit Operations - All tools should use these
    # =========================================================================

    def set_tile(self, y: int, x: int, tile_value: int) -> EditResult:
        """Set a single tile.

        Args:
            y: Row coordinate.
            x: Column coordinate.
            tile_value: New tile value.

        Returns:
            EditResult with success status and diff.
        """
        errors = []

        # Validate position
        if not self.validate_position(y, x):
            return EditResult(
                success=False,
                errors=[f"Position ({y}, {x}) is out of bounds (level is {self.height}x{self.width})"],
            )

        # Validate tile value
        if not self.validate_tile(tile_value):
            return EditResult(
                success=False,
                errors=[f"Invalid tile value {tile_value}. Valid values: {self._valid_tiles}"],
            )

        # Check if actually changing
        old_value = int(self._level.data[y, x])
        if old_value == tile_value:
            return EditResult(
                success=True,
                num_tiles_changed=0,
                diff=[],
                new_level=self._level.copy(),
            )

        # Apply change
        self._level.data[y, x] = tile_value

        diff = [{
            "y": y,
            "x": x,
            "old": self._tile_names.get(old_value, str(old_value)),
            "new": self._tile_names.get(tile_value, str(tile_value)),
        }]

        self._total_tiles_changed += 1
        self._edit_history.append({
            "operation": "set_tile",
            "y": y,
            "x": x,
            "old_value": old_value,
            "new_value": tile_value,
        })

        return EditResult(
            success=True,
            num_tiles_changed=1,
            diff=diff,
            new_level=self._level.copy(),
        )

    def set_tiles(self, coords: list[tuple[int, int]], tile_value: int) -> EditResult:
        """Set multiple tiles to the same value.

        Args:
            coords: List of (y, x) coordinates.
            tile_value: New tile value.

        Returns:
            EditResult with combined diff.
        """
        # Validate tile value first
        if not self.validate_tile(tile_value):
            return EditResult(
                success=False,
                errors=[f"Invalid tile value {tile_value}. Valid values: {self._valid_tiles}"],
            )

        diff = []
        num_changed = 0
        errors = []

        for y, x in coords:
            if not self.validate_position(y, x):
                errors.append(f"Position ({y}, {x}) is out of bounds")
                continue

            old_value = int(self._level.data[y, x])
            if old_value != tile_value:
                self._level.data[y, x] = tile_value
                diff.append({
                    "y": y,
                    "x": x,
                    "old": self._tile_names.get(old_value, str(old_value)),
                    "new": self._tile_names.get(tile_value, str(tile_value)),
                })
                num_changed += 1

        self._total_tiles_changed += num_changed

        if num_changed > 0:
            self._edit_history.append({
                "operation": "set_tiles",
                "num_coords": len(coords),
                "tile_value": tile_value,
                "num_changed": num_changed,
            })

        return EditResult(
            success=True,
            num_tiles_changed=num_changed,
            diff=diff,
            errors=errors,
            new_level=self._level.copy(),
        )

    def set_line(
        self,
        start_y: int,
        start_x: int,
        end_y: int,
        end_x: int,
        tile_value: int,
    ) -> EditResult:
        """Set a horizontal or vertical line of tiles.

        Args:
            start_y: Starting row.
            start_x: Starting column.
            end_y: Ending row.
            end_x: Ending column.
            tile_value: Tile value to set.

        Returns:
            EditResult.
        """
        # Validate that line is horizontal or vertical
        if start_y != end_y and start_x != end_x:
            return EditResult(
                success=False,
                errors=["Line must be horizontal or vertical (start and end must share y or x)"],
            )

        # Generate coordinates
        coords = []
        if start_y == end_y:
            # Horizontal line
            x_min, x_max = min(start_x, end_x), max(start_x, end_x)
            for x in range(x_min, x_max + 1):
                coords.append((start_y, x))
        else:
            # Vertical line
            y_min, y_max = min(start_y, end_y), max(start_y, end_y)
            for y in range(y_min, y_max + 1):
                coords.append((y, start_x))

        return self.set_tiles(coords, tile_value)

    def set_rect(
        self,
        top_y: int,
        left_x: int,
        bottom_y: int,
        right_x: int,
        tile_value: int,
        filled: bool = True,
    ) -> EditResult:
        """Set a rectangular region of tiles.

        Args:
            top_y: Top row (inclusive).
            left_x: Left column (inclusive).
            bottom_y: Bottom row (inclusive).
            right_x: Right column (inclusive).
            tile_value: Tile value to set.
            filled: If True, fill entire rectangle. If False, only draw border.

        Returns:
            EditResult.
        """
        # Normalize coordinates
        y_min, y_max = min(top_y, bottom_y), max(top_y, bottom_y)
        x_min, x_max = min(left_x, right_x), max(left_x, right_x)

        coords = []
        if filled:
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    coords.append((y, x))
        else:
            # Border only
            for x in range(x_min, x_max + 1):
                coords.append((y_min, x))  # Top
                coords.append((y_max, x))  # Bottom
            for y in range(y_min + 1, y_max):
                coords.append((y, x_min))  # Left
                coords.append((y, x_max))  # Right

        return self.set_tiles(coords, tile_value)

    def flood_fill(
        self,
        start_y: int,
        start_x: int,
        tile_value: int,
        max_tiles: int = 100,
    ) -> EditResult:
        """Flood fill from a starting position.

        Args:
            start_y: Starting row.
            start_x: Starting column.
            tile_value: Tile value to fill with.
            max_tiles: Maximum tiles to fill (safety limit).

        Returns:
            EditResult.
        """
        if not self.validate_position(start_y, start_x):
            return EditResult(
                success=False,
                errors=[f"Start position ({start_y}, {start_x}) is out of bounds"],
            )

        if not self.validate_tile(tile_value):
            return EditResult(
                success=False,
                errors=[f"Invalid tile value {tile_value}"],
            )

        original_value = int(self._level.data[start_y, start_x])
        if original_value == tile_value:
            return EditResult(success=True, num_tiles_changed=0, diff=[], new_level=self._level.copy())

        # BFS flood fill
        visited = set()
        queue = [(start_y, start_x)]
        coords_to_fill = []

        while queue and len(coords_to_fill) < max_tiles:
            y, x = queue.pop(0)
            if (y, x) in visited:
                continue
            if not self.validate_position(y, x):
                continue
            if int(self._level.data[y, x]) != original_value:
                continue

            visited.add((y, x))
            coords_to_fill.append((y, x))

            # Add 4-neighbors
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (ny, nx) not in visited:
                    queue.append((ny, nx))

        return self.set_tiles(coords_to_fill, tile_value)

    # =========================================================================
    # Checkpoint / Rollback Support
    # =========================================================================

    def checkpoint(self) -> Level:
        """Create a checkpoint of current level state.

        Returns:
            Copy of current level.
        """
        return self._level.copy()

    def rollback(self, checkpoint: Level) -> None:
        """Rollback to a previous checkpoint.

        Args:
            checkpoint: Level state to restore.
        """
        self._level = checkpoint.copy()

    # =========================================================================
    # Evaluation Integration
    # =========================================================================

    def evaluate(self) -> dict:
        """Evaluate current level using PCG benchmark.

        Returns:
            Info dict from env.info().
        """
        return self._env.info(self._level.data)

    def render(self, info: dict | None = None):
        """Render current level.

        Args:
            info: Optional pre-computed info dict.

        Returns:
            PIL Image.
        """
        if info is None:
            info = self.evaluate()
        return self._env.render(self._level.data, info)
