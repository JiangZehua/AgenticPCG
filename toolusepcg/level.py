"""Level representation for PCG problems (Binary, Zelda, etc.).

Tile definitions align with /home/zehua/ReasonPCG/get_reason_traces/env_definitions.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


# =============================================================================
# TILE DEFINITIONS - Aligned with env_definitions.py
# =============================================================================

class BinaryTiles(IntEnum):
    """Binary maze tiles - only walls and empty spaces."""
    EMPTY = 0
    WALL = 1


class ZeldaTiles(IntEnum):
    """Zelda environment tiles."""
    WALL = 0
    EMPTY = 1
    PLAYER = 2
    KEY = 3
    DOOR = 4
    ENEMY = 5


class SokobanTiles(IntEnum):
    """Sokoban environment tiles."""
    SOLID = 0
    EMPTY = 1
    PLAYER = 2
    CRATE = 3
    TARGET = 4


class LodeRunnerTiles(IntEnum):
    """Lode Runner environment tiles."""
    SOLID = 0
    EMPTY = 1
    PLAYER = 2
    GOLD = 3
    ENEMY = 4
    LADDER = 5
    ROPE = 6


class SMBTiles(IntEnum):
    """Super Mario Bros environment tiles."""
    EMPTY = 0
    SOLID = 1
    LADDER = 2
    BRICK = 3
    QUESTION = 4
    TUBE = 5
    COIN = 6
    GOOMBA = 7
    KOOPA = 8
    SPINY = 9


# Legacy constants for backward compatibility
EMPTY = 0
WALL = 1


# =============================================================================
# TILE MAPPINGS
# =============================================================================

# Binary: EMPTY=0 -> ".", WALL=1 -> "#"
BINARY_TILE_CHARS = {0: '.', 1: '#'}
BINARY_CHAR_TILES = {'.': 0, '#': 1}
BINARY_TILE_NAMES = {0: 'empty', 1: 'wall'}
BINARY_NAME_TILES = {'empty': 0, 'wall': 1}

# Zelda: WALL=0 -> "#", EMPTY=1 -> ".", PLAYER=2 -> "P", KEY=3 -> "K", DOOR=4 -> "D", ENEMY=5 -> "E"
ZELDA_TILE_CHARS = {0: '#', 1: '.', 2: 'P', 3: 'K', 4: 'D', 5: 'E'}
ZELDA_CHAR_TILES = {'#': 0, '.': 1, 'P': 2, 'K': 3, 'D': 4, 'E': 5}
ZELDA_TILE_NAMES = {0: 'wall', 1: 'empty', 2: 'player', 3: 'key', 4: 'door', 5: 'enemy'}
ZELDA_NAME_TILES = {'wall': 0, 'empty': 1, 'player': 2, 'key': 3, 'door': 4, 'enemy': 5}

# Sokoban: SOLID=0 -> "#", EMPTY=1 -> " ", PLAYER=2 -> "@", CRATE=3 -> "$", TARGET=4 -> "."
SOKOBAN_TILE_CHARS = {0: '#', 1: ' ', 2: '@', 3: '$', 4: '.'}
SOKOBAN_CHAR_TILES = {'#': 0, ' ': 1, '@': 2, '$': 3, '.': 4}
SOKOBAN_TILE_NAMES = {0: 'solid', 1: 'empty', 2: 'player', 3: 'crate', 4: 'target'}
SOKOBAN_NAME_TILES = {'solid': 0, 'empty': 1, 'player': 2, 'crate': 3, 'target': 4}

# Lode Runner: SOLID=0, EMPTY=1, PLAYER=2, GOLD=3, ENEMY=4, LADDER=5, ROPE=6
LODERUNNER_TILE_CHARS = {0: 'S', 1: '.', 2: 'P', 3: 'G', 4: 'E', 5: 'L', 6: 'R'}
LODERUNNER_CHAR_TILES = {'S': 0, '.': 1, 'P': 2, 'G': 3, 'E': 4, 'L': 5, 'R': 6}
LODERUNNER_TILE_NAMES = {0: 'solid', 1: 'empty', 2: 'player', 3: 'gold', 4: 'enemy', 5: 'ladder', 6: 'rope'}
LODERUNNER_NAME_TILES = {'solid': 0, 'empty': 1, 'player': 2, 'gold': 3, 'enemy': 4, 'ladder': 5, 'rope': 6}

# SMB: EMPTY=0, SOLID=1, LADDER=2, BRICK=3, QUESTION=4, TUBE=5, COIN=6, GOOMBA=7, KOOPA=8, SPINY=9
SMB_TILE_CHARS = {0: '.', 1: 'X', 2: 'L', 3: 'B', 4: '?', 5: 'T', 6: 'C', 7: 'G', 8: 'K', 9: 'Y'}
SMB_CHAR_TILES = {'.': 0, 'X': 1, 'L': 2, 'B': 3, '?': 4, 'T': 5, 'C': 6, 'G': 7, 'K': 8, 'Y': 9}
SMB_TILE_NAMES = {0: 'empty', 1: 'solid', 2: 'ladder', 3: 'brick', 4: 'question', 5: 'tube', 6: 'coin', 7: 'goomba', 8: 'koopa', 9: 'spiny'}
SMB_NAME_TILES = {'empty': 0, 'solid': 1, 'ladder': 2, 'brick': 3, 'question': 4, 'tube': 5, 'coin': 6, 'goomba': 7, 'koopa': 8, 'spiny': 9}


def get_tile_mappings(problem_type: str) -> tuple[dict, dict, dict, dict]:
    """Get tile mappings for a problem type.

    Args:
        problem_type: 'binary', 'zelda', 'sokoban', 'loderunner', or 'smb'

    Returns:
        Tuple of (tile_chars, char_tiles, tile_names, name_tiles)
    """
    if problem_type == 'zelda':
        return ZELDA_TILE_CHARS, ZELDA_CHAR_TILES, ZELDA_TILE_NAMES, ZELDA_NAME_TILES
    elif problem_type == 'sokoban':
        return SOKOBAN_TILE_CHARS, SOKOBAN_CHAR_TILES, SOKOBAN_TILE_NAMES, SOKOBAN_NAME_TILES
    elif problem_type == 'loderunner':
        return LODERUNNER_TILE_CHARS, LODERUNNER_CHAR_TILES, LODERUNNER_TILE_NAMES, LODERUNNER_NAME_TILES
    elif problem_type == 'smb':
        return SMB_TILE_CHARS, SMB_CHAR_TILES, SMB_TILE_NAMES, SMB_NAME_TILES
    elif problem_type == 'binarydoor':
        return BINARY_TILE_CHARS, BINARY_CHAR_TILES, BINARY_TILE_NAMES, BINARY_NAME_TILES
    else:
        return BINARY_TILE_CHARS, BINARY_CHAR_TILES, BINARY_TILE_NAMES, BINARY_NAME_TILES


def get_problem_type_from_env_name(env_name: str) -> str:
    """Infer problem type from environment name.

    Args:
        env_name: Environment name like 'binary-v0', 'zelda-v0', 'sokoban-v0', etc.

    Returns:
        Problem type: 'binary', 'zelda', 'sokoban', 'loderunner', or 'smb'
    """
    if env_name.startswith('zelda'):
        return 'zelda'
    elif env_name.startswith('sokoban'):
        return 'sokoban'
    elif env_name.startswith('loderunner'):
        return 'loderunner'
    elif env_name.startswith('smb'):
        return 'smb'
    elif env_name.startswith('binarydoor'):
        return 'binarydoor'
    else:
        return 'binary'


@dataclass
class Level:
    """Represents a game level (works for Binary, Zelda, etc.)."""

    data: np.ndarray  # 2D array of tile values
    problem_type: str = "binary"  # 'binary' or 'zelda'

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def tile_chars(self) -> dict[int, str]:
        """Get tile-to-char mapping for this problem type."""
        return get_tile_mappings(self.problem_type)[0]

    @property
    def char_tiles(self) -> dict[str, int]:
        """Get char-to-tile mapping for this problem type."""
        return get_tile_mappings(self.problem_type)[1]

    @property
    def tile_names(self) -> dict[int, str]:
        """Get tile-to-name mapping for this problem type."""
        return get_tile_mappings(self.problem_type)[2]

    @property
    def name_tiles(self) -> dict[str, int]:
        """Get name-to-tile mapping for this problem type."""
        return get_tile_mappings(self.problem_type)[3]

    def to_string(self) -> str:
        """Convert level to ASCII string representation.

        Returns:
            String with appropriate chars for each tile type.
        """
        tile_chars = self.tile_chars
        lines = []
        for row in self.data:
            line = "".join(tile_chars.get(int(cell), '?') for cell in row)
            lines.append(line)
        return "\n".join(lines)

    @classmethod
    def from_string(cls, s: str, problem_type: str = "binary") -> Level:
        """Parse level from ASCII string representation.

        Args:
            s: String with appropriate chars for each tile type.
            problem_type: 'binary' or 'zelda'

        Returns:
            Level object.
        """
        _, char_tiles, _, _ = get_tile_mappings(problem_type)
        # Default tile: WALL for zelda (0), EMPTY for binary (0)
        default_tile = 0

        lines = [line for line in s.strip("\n").split("\n") if line]
        height = len(lines)
        width = len(lines[0]) if lines else 0

        data = np.zeros((height, width), dtype=np.int32)
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if x < width:
                    data[y, x] = char_tiles.get(char, default_tile)

        return cls(data=data, problem_type=problem_type)

    def copy(self) -> Level:
        """Create a deep copy of the level."""
        return Level(data=self.data.copy(), problem_type=self.problem_type)

    def get_tile(self, y: int, x: int) -> int:
        """Get tile value at position."""
        return int(self.data[y, x])

    def set_tile(self, y: int, x: int, value: int) -> None:
        """Set tile value at position."""
        self.data[y, x] = value

    def is_valid_position(self, y: int, x: int) -> bool:
        """Check if position is within bounds."""
        return 0 <= y < self.height and 0 <= x < self.width

    def count_tiles(self, tile_type: int) -> int:
        """Count tiles of a specific type."""
        return int(np.sum(self.data == tile_type))

    def diff(self, other: Level) -> list[tuple[int, int, int, int]]:
        """Compute difference with another level.

        Returns:
            List of (y, x, old_value, new_value) tuples for changed tiles.
        """
        changes = []
        diff_mask = self.data != other.data
        ys, xs = np.where(diff_mask)
        for y, x in zip(ys, xs):
            changes.append((int(y), int(x), int(self.data[y, x]), int(other.data[y, x])))
        return changes

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Level):
            return False
        return np.array_equal(self.data, other.data)
