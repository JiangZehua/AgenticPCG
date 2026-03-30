"""LevelFactory for creating and initializing levels."""

from __future__ import annotations

import importlib

import pcg_benchmark
import numpy as np

from .level import (
    Level, get_problem_type_from_env_name,
    BinaryTiles, ZeldaTiles, SokobanTiles, LodeRunnerTiles, SMBTiles,
)
from .config import Config


def get_problem_class(problem_type: str):
    """Dynamically import and return the problem class for a given problem type."""
    class_map = {
        'binary': ('pcg_benchmark.probs.binary', 'BinaryProblem'),
        'zelda': ('pcg_benchmark.probs.zelda', 'ZeldaProblem'),
        'sokoban': ('pcg_benchmark.probs.sokoban', 'SokobanProblem'),
        'loderunner': ('pcg_benchmark.probs.loderunnertile', 'LodeRunnerProblem'),
        'smb': ('pcg_benchmark.probs.smbtile', 'MarioProblem'),
        'binarydoor': ('pcg_benchmark.probs.binarydoor', 'BinaryDoorProblem'),
    }

    if problem_type not in class_map:
        raise ValueError(f"Unknown problem type: {problem_type}")

    module_name, class_name = class_map[problem_type]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class LevelFactory:
    """Factory for creating game levels (Binary, Zelda, etc.)."""

    def __init__(self, config: Config):
        """Initialize factory with configuration.

        Args:
            config: Configuration object specifying env parameters.
        """
        self.config = config
        self._problem_type = get_problem_type_from_env_name(config.env.name)

        # Register and create environment with config dimensions
        self._env = self._create_env_with_config_dims()

    def _create_env_with_config_dims(self):
        """Create environment with dimensions from config.

        Registers a custom environment if needed to match config dimensions.
        """
        width = self.config.env.width
        height = self.config.env.height

        # Build environment params based on problem type
        if self._problem_type == 'binary':
            # Binary needs: width, height, path (target path length)
            target_path = width + height  # Default target
            env_name = f'binary-w{width}-h{height}-p{target_path}'
            env_params = {"width": width, "height": height, "path": target_path}

        elif self._problem_type == 'zelda':
            # Zelda needs: width, height, enemies, sol_length
            n_enemies = int(0.8 * width)  # Default enemy count
            sol_length = width + height  # Default solution length
            env_name = f'zelda-w{width}-h{height}-e{n_enemies}-s{sol_length}'
            env_params = {"width": width, "height": height, "enemies": n_enemies, "sol_length": sol_length}

        elif self._problem_type == 'sokoban':
            # Sokoban needs: width, height, difficulty (default 1)
            difficulty = 1  # Default difficulty
            env_name = f'sokoban-w{width}-h{height}-d{difficulty}'
            env_params = {"width": width, "height": height, "difficulty": difficulty}

        elif self._problem_type == 'loderunner':
            # Loderunner needs: width, height, gold, enemies
            gold = max(1, width // 4)  # Default gold count
            enemies = max(1, width // 6)  # Default enemy count
            env_name = f'loderunner-w{width}-h{height}-g{gold}-e{enemies}'
            env_params = {"width": width, "height": height, "gold": gold, "enemies": enemies}

        elif self._problem_type == 'smb':
            # SMB needs: width, height, agent (solver type)
            agent = self.config.env.smb_solver
            env_name = f'smb-w{width}-h{height}'
            if agent != "auto":
                env_name += f'-{agent}'
            env_params = {"width": width, "height": height, "agent": agent}

        elif self._problem_type == 'binarydoor':
            # BinaryDoor needs: width, height, door_path (target), door_seed
            target_path = width + height  # Default target
            door_seed = self.config.seed
            env_name = f'binarydoor-w{width}-h{height}-dp{target_path}-ds{door_seed}'
            env_params = {"width": width, "height": height, "door_path": target_path, "door_seed": door_seed}

        else:
            # Fallback: use the original env name without custom dimensions
            return pcg_benchmark.make(self.config.env.name)

        # Register environment if not already registered
        if env_name not in pcg_benchmark.PROBLEMS:
            problem_class = get_problem_class(self._problem_type)
            pcg_benchmark.register(env_name, problem_class, env_params)

        return pcg_benchmark.make(env_name)

    @property
    def env(self):
        """Access the underlying PCG benchmark environment."""
        return self._env

    @property
    def problem_type(self) -> str:
        """Get the problem type (binary or zelda)."""
        return self._problem_type

    def create_random(self, seed: int | None = None) -> Level:
        """Create a random level with optional seed.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Randomly initialized Level.
        """
        if seed is not None:
            self._env.seed(seed)

        content = self._env.content_space.sample()
        return Level(data=np.array(content, dtype=np.int32), problem_type=self._problem_type)

    def create_from_string(self, level_str: str) -> Level:
        """Create level from ASCII string representation.

        Args:
            level_str: ASCII representation with appropriate chars for tile types.

        Returns:
            Level parsed from string.
        """
        return Level.from_string(level_str, problem_type=self._problem_type)

    # Per-problem-type tile lookups derived from tile enums (single source of truth).
    _EMPTY_TILE = {
        'binary': BinaryTiles.EMPTY,
        'binarydoor': BinaryTiles.EMPTY,
        'zelda': ZeldaTiles.EMPTY,
        'sokoban': SokobanTiles.EMPTY,
        'loderunner': LodeRunnerTiles.EMPTY,
        'smb': SMBTiles.EMPTY,
    }

    _BLOCKING_TILE = {
        'binary': BinaryTiles.WALL,
        'binarydoor': BinaryTiles.WALL,
        'zelda': ZeldaTiles.WALL,
        'sokoban': SokobanTiles.SOLID,
        'loderunner': LodeRunnerTiles.SOLID,
        'smb': SMBTiles.SOLID,
    }

    # Per-problem-type tile weights for weighted random initialization.
    # Common terrain tiles get high weight, structural tiles medium, special entities low.
    # Weights are unnormalized (normalized at sampling time).
    _TILE_WEIGHTS = {
        'binary': {
            BinaryTiles.EMPTY: 1.0,
            BinaryTiles.WALL: 1.0,
        },
        'binarydoor': {
            BinaryTiles.EMPTY: 1.0,
            BinaryTiles.WALL: 1.0,
        },
        'zelda': {
            ZeldaTiles.WALL: 0.40,
            ZeldaTiles.EMPTY: 0.40,
            ZeldaTiles.PLAYER: 0.02,
            ZeldaTiles.KEY: 0.02,
            ZeldaTiles.DOOR: 0.02,
            ZeldaTiles.ENEMY: 0.04,
        },
        'sokoban': {
            SokobanTiles.SOLID: 0.40,
            SokobanTiles.EMPTY: 0.40,
            SokobanTiles.PLAYER: 0.02,
            SokobanTiles.CRATE: 0.04,
            SokobanTiles.TARGET: 0.04,
        },
        'loderunner': {
            LodeRunnerTiles.SOLID: 0.30,
            LodeRunnerTiles.EMPTY: 0.30,
            LodeRunnerTiles.PLAYER: 0.02,
            LodeRunnerTiles.GOLD: 0.04,
            LodeRunnerTiles.ENEMY: 0.04,
            LodeRunnerTiles.LADDER: 0.10,
            LodeRunnerTiles.ROPE: 0.10,
        },
        'smb': {
            SMBTiles.EMPTY: 0.35,
            SMBTiles.SOLID: 0.35,
            SMBTiles.LADDER: 0.05,
            SMBTiles.BRICK: 0.05,
            SMBTiles.QUESTION: 0.02,
            SMBTiles.TUBE: 0.02,
            SMBTiles.COIN: 0.02,
            SMBTiles.GOOMBA: 0.01,
            SMBTiles.KOOPA: 0.01,
            SMBTiles.SPINY: 0.01,
        },
    }

    def create_empty(self) -> Level:
        """Create an empty level (all passable tiles)."""
        tile = int(self._EMPTY_TILE[self._problem_type])
        data = np.full((self.config.env.height, self.config.env.width), tile, dtype=np.int32)
        return Level(data=data, problem_type=self._problem_type)

    def create_filled(self) -> Level:
        """Create a filled level (all blocking tiles)."""
        tile = int(self._BLOCKING_TILE[self._problem_type])
        data = np.full((self.config.env.height, self.config.env.width), tile, dtype=np.int32)
        return Level(data=data, problem_type=self._problem_type)

    def create_weighted_random(self, seed: int | None = None) -> Level:
        """Create a level by sampling tiles from a weighted probability distribution.

        Common tiles (empty, wall/solid) are heavily favored while special
        entity tiles (player, key, door, enemy, coin, etc.) are rare.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Weighted-random initialized Level.
        """
        rng = np.random.RandomState(seed)
        h, w = self.config.env.height, self.config.env.width

        weights_dict = self._TILE_WEIGHTS[self._problem_type]
        # Sort by integer tile value to ensure tiles and weights stay aligned
        sorted_items = sorted(weights_dict.items(), key=lambda kv: int(kv[0]))
        tiles = [int(k) for k, _ in sorted_items]
        weights = np.array([v for _, v in sorted_items], dtype=np.float64)
        weights /= weights.sum()

        data = rng.choice(tiles, size=(h, w), p=weights).astype(np.int32)
        return Level(data=data, problem_type=self._problem_type)
