"""StatsEvaluator for computing level metrics (Binary, Zelda, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .level import Level
from .config import Config, get_problem_type


@dataclass
class EvalResult:
    """Result of level evaluation."""

    metrics: dict[str, int | float] = field(default_factory=dict)
    # For binary: {path, num_connected_regions}
    # For zelda: {regions, players, keys, doors, enemies, player_key, key_door}

    targets: dict[str, dict] = field(default_factory=dict)
    # Target specifications from config

    valid: bool = True
    # Whether the level is structurally valid

    errors: list[str] = field(default_factory=list)
    # Any errors encountered during evaluation

    info: dict | None = None
    # Raw info dict from PCG benchmark (includes d_map for rendering, paths for zelda)

    problem_type: str = "binary"
    # Problem type for this result


class StatsEvaluator:
    """Evaluates levels using PCG benchmark (supports Binary and Zelda)."""

    def __init__(self, config: Config, env):
        """Initialize evaluator.

        Args:
            config: Configuration with target specifications.
            env: PCG benchmark environment instance.
        """
        self.config = config
        self._env = env
        self.problem_type = get_problem_type(config.env.name)

    def evaluate(self, level: Level) -> EvalResult:
        """Evaluate a level and compute metrics.

        Args:
            level: Level to evaluate.

        Returns:
            EvalResult with metrics, targets, and validity.
        """
        if self.problem_type == 'zelda':
            return self._evaluate_zelda(level)
        elif self.problem_type == 'sokoban':
            return self._evaluate_sokoban(level)
        elif self.problem_type == 'loderunner':
            return self._evaluate_loderunner(level)
        elif self.problem_type == 'smb':
            return self._evaluate_smb(level)
        elif self.problem_type == 'binarydoor':
            return self._evaluate_binarydoor(level)
        else:
            return self._evaluate_binary(level)

    def _evaluate_binary(self, level: Level) -> EvalResult:
        """Evaluate a binary maze level."""
        errors = []

        # Get info from PCG benchmark
        try:
            info = self._env.info(level.data)
        except Exception as e:
            return EvalResult(
                metrics={"path": 0, "num_connected_regions": 0},
                valid=False,
                errors=[f"Failed to evaluate level: {e}"],
                problem_type="binary",
            )

        # Extract metrics
        path = info.get("path", 0)
        num_connected_regions = info.get("regions", 0)

        metrics = {
            "path": path,
            "num_connected_regions": num_connected_regions,
        }

        # Build targets dict
        targets = {
            "path": {
                "mode": self.config.targets.path.mode,
                "value": self.config.targets.path.value,
                "min_threshold": self.config.targets.path.min_threshold,
            },
            "num_connected_regions": {
                "mode": self.config.targets.num_connected_regions.mode,
                "value": self.config.targets.num_connected_regions.value,
            },
        }

        # Check validity
        valid = True
        if num_connected_regions == 0:
            valid = False
            errors.append("Level has no connected regions (all walls)")
        if path == 0:
            valid = False
            errors.append("No path exists (maze is untraversable)")

        return EvalResult(
            metrics=metrics,
            targets=targets,
            valid=valid,
            errors=errors,
            info=info,
            problem_type="binary",
        )

    def _evaluate_binarydoor(self, level: Level) -> EvalResult:
        """Evaluate a binary door maze level."""
        errors = []

        # Get info from PCG benchmark
        try:
            info = self._env.info(level.data)
        except Exception as e:
            return EvalResult(
                metrics={"door_path": 0, "num_connected_regions": 0},
                valid=False,
                errors=[f"Failed to evaluate level: {e}"],
                problem_type="binarydoor",
            )

        # Extract metrics
        door_path = info.get("door_path", 0)
        num_connected_regions = info.get("regions", 0)
        door1 = info.get("door1", (0, 0))
        door2 = info.get("door2", (0, 0))

        metrics = {
            "door_path": door_path,
            "num_connected_regions": num_connected_regions,
        }

        # Build targets dict
        targets = {
            "num_connected_regions": {
                "mode": self.config.targets.num_connected_regions.mode,
                "value": self.config.targets.num_connected_regions.value,
            },
        }

        # Add door_path target if configured
        door_path_cfg = self.config.targets.door_path
        if door_path_cfg:
            targets["door_path"] = {
                "mode": door_path_cfg.mode,
                "value": door_path_cfg.value,
                "min_threshold": door_path_cfg.min_threshold,
            }

        # Check validity
        valid = True
        if num_connected_regions == 0:
            valid = False
            errors.append("Level has no connected regions (all walls)")
        if door_path == 0:
            valid = False
            errors.append("No path exists between doors")

        return EvalResult(
            metrics=metrics,
            targets=targets,
            valid=valid,
            errors=errors,
            info=info,
            problem_type="binarydoor",
        )

    def _evaluate_zelda(self, level: Level) -> EvalResult:
        """Evaluate a Zelda level."""
        errors = []

        # Get info from PCG benchmark
        try:
            info = self._env.info(level.data)
        except Exception as e:
            return EvalResult(
                metrics={
                    "regions": 0, "players": 0, "keys": 0, "doors": 0,
                    "enemies": 0, "player_key": 0, "key_door": 0, "solution_length": 0,
                },
                valid=False,
                errors=[f"Failed to evaluate level: {e}"],
                problem_type="zelda",
            )

        # Extract metrics from Zelda info
        # Info contains: regions, players, keys, doors, enemies, player_key, key_door, pk_path, kd_path
        metrics = {
            "regions": info.get("regions", 0),
            "players": info.get("players", 0),
            "keys": info.get("keys", 0),
            "doors": info.get("doors", 0),
            "enemies": info.get("enemies", 0),
            "player_key": info.get("player_key", 0),  # Distance from player to key
            "key_door": info.get("key_door", 0),      # Distance from key to door
        }

        # Compute solution_length (total path)
        solution_length = metrics["player_key"] + metrics["key_door"]
        metrics["solution_length"] = solution_length

        # Build targets dict
        targets = {
            "regions": {"mode": "target", "value": 1},
            "players": {"mode": "target", "value": 1},
            "keys": {"mode": "target", "value": 1},
            "doors": {"mode": "target", "value": 1},
        }

        # Add configured targets
        if self.config.targets.enemies:
            targets["enemies"] = {
                "mode": self.config.targets.enemies.mode,
                "value": self.config.targets.enemies.value,
            }
        if self.config.targets.player_key:
            targets["player_key"] = {
                "mode": self.config.targets.player_key.mode,
                "value": self.config.targets.player_key.value,
            }
        if self.config.targets.key_door:
            targets["key_door"] = {
                "mode": self.config.targets.key_door.mode,
                "value": self.config.targets.key_door.value,
            }
        if self.config.targets.solution_length:
            targets["solution_length"] = {
                "mode": self.config.targets.solution_length.mode,
                "value": self.config.targets.solution_length.value,
            }

        # Check validity
        valid = True

        # Must have exactly 1 player, 1 key, 1 door
        if metrics["players"] != 1:
            valid = False
            errors.append(f"Level has {metrics['players']} players (need exactly 1)")
        if metrics["keys"] != 1:
            valid = False
            errors.append(f"Level has {metrics['keys']} keys (need exactly 1)")
        if metrics["doors"] != 1:
            valid = False
            errors.append(f"Level has {metrics['doors']} doors (need exactly 1)")

        # Must be solvable (paths exist)
        if metrics["player_key"] == 0 and metrics["players"] == 1 and metrics["keys"] == 1:
            valid = False
            errors.append("Player cannot reach key (no path)")
        if metrics["key_door"] == 0 and metrics["keys"] == 1 and metrics["doors"] == 1:
            valid = False
            errors.append("Key cannot reach door (no path)")

        # Should have 1 connected region
        if metrics["regions"] != 1:
            errors.append(f"Level has {metrics['regions']} regions (should be 1)")

        return EvalResult(
            metrics=metrics,
            targets=targets,
            valid=valid,
            errors=errors,
            info=info,
            problem_type="zelda",
        )

    def _evaluate_sokoban(self, level: Level) -> EvalResult:
        """Evaluate a Sokoban level."""
        errors = []

        # Get info from PCG benchmark
        try:
            info = self._env.info(level.data)
        except Exception as e:
            return EvalResult(
                metrics={
                    "players": 0, "crates": 0, "targets": 0,
                    "heuristic": -1, "solution_length": 0,
                },
                valid=False,
                errors=[f"Failed to evaluate level: {e}"],
                problem_type="sokoban",
            )

        # Extract metrics from Sokoban info
        # Info contains: players, crates, targets, content, heuristic, solution
        players = info.get("players", 0)
        crates = info.get("crates", 0)
        targets = info.get("targets", 0)
        heuristic = info.get("heuristic", -1)
        solution = info.get("solution", [])
        solution_length = len(solution)

        metrics = {
            "players": players,
            "crates": crates,
            "targets": targets,
            "heuristic": heuristic,
            "solution_length": solution_length,
        }

        # Build targets dict
        target_crates = 3  # Default
        if self.config.targets.crates and self.config.targets.crates.value is not None:
            target_crates = self.config.targets.crates.value

        targets_dict = {
            "players": {"mode": "target", "value": 1},
            "crates": {"mode": "target", "value": target_crates},
            "targets": {"mode": "target", "value": target_crates},
            "heuristic": {"mode": "target", "value": 0},  # 0 means solvable
            "solution_length": {"mode": "maximize"},
        }

        # Check validity
        valid = True

        # Must have exactly 1 player
        if players != 1:
            valid = False
            errors.append(f"Level has {players} players (need exactly 1)")

        # Must have at least 1 crate
        if crates == 0:
            valid = False
            errors.append("Level has no crates")

        # Crates must equal targets
        if crates != targets:
            valid = False
            errors.append(f"Crates ({crates}) != targets ({targets}) - must be equal")

        # Must be solvable (solution_length > 0 means solver found a solution;
        # heuristic > 0 means solver ran but failed, heuristic < 0 means structural issues)
        if solution_length == 0:
            valid = False
            errors.append("Level is not solvable")

        return EvalResult(
            metrics=metrics,
            targets=targets_dict,
            valid=valid,
            errors=errors,
            info=info,
            problem_type="sokoban",
        )

    def _evaluate_loderunner(self, level: Level) -> EvalResult:
        """Evaluate a Lode Runner level."""
        errors = []

        # Get info from PCG benchmark
        try:
            info = self._env.info(level.data)
        except Exception as e:
            return EvalResult(
                metrics={
                    "player": 0, "gold": 0, "enemy": 0,
                    "ladder": 0, "rope": 0, "collected_gold": 0,
                    "used_tiles": 0, "tiles": 0,
                },
                valid=False,
                errors=[f"Failed to evaluate level: {e}"],
                problem_type="loderunner",
            )

        # Extract metrics
        player = info.get("player", 0)
        gold = info.get("gold", 0)
        enemy = info.get("enemy", 0)
        ladder = info.get("ladder", 0)
        rope = info.get("rope", 0)
        collected_gold = info.get("collected_gold", 0)
        used_tiles = info.get("used_tiles", 0)
        tiles = info.get("tiles", 0)

        metrics = {
            "player": player,
            "gold": gold,
            "enemy": enemy,
            "ladder": ladder,
            "rope": rope,
            "collected_gold": collected_gold,
            "used_tiles": used_tiles,
            "tiles": tiles,
        }

        # Build targets dict
        target_gold = 3
        target_enemy = 2
        if self.config.targets.gold and self.config.targets.gold.value is not None:
            target_gold = self.config.targets.gold.value
        if self.config.targets.enemies and self.config.targets.enemies.value is not None:
            target_enemy = self.config.targets.enemies.value

        targets_dict = {
            "player": {"mode": "target", "value": 1},
            "gold": {"mode": "target", "value": target_gold},
            "enemy": {"mode": "target", "value": target_enemy},
            "collected_gold": {"mode": "maximize"},
        }

        # Add ladder/rope targets if configured
        ladder_cfg = self.config.targets.ladder
        if ladder_cfg:
            targets_dict["ladder"] = {"mode": ladder_cfg.mode, "value": ladder_cfg.value}
        rope_cfg = self.config.targets.rope
        if rope_cfg:
            targets_dict["rope"] = {"mode": rope_cfg.mode, "value": rope_cfg.value}

        # Check validity
        valid = True

        if player != 1:
            valid = False
            errors.append(f"Level has {player} players (need exactly 1)")

        if gold == 0:
            valid = False
            errors.append("Level has no gold")

        # Check if all gold is collectible
        if player == 1 and gold > 0 and collected_gold < gold:
            valid = False
            errors.append(f"Not all gold is collectible ({collected_gold}/{gold} reachable)")

        return EvalResult(
            metrics=metrics,
            targets=targets_dict,
            valid=valid,
            errors=errors,
            info=info,
            problem_type="loderunner",
        )

    def _evaluate_smb(self, level: Level) -> EvalResult:
        """Evaluate a Super Mario Bros level."""
        errors = []

        # Get info from PCG benchmark
        try:
            info = self._env.info(level.data)
        except Exception as e:
            return EvalResult(
                metrics={
                    "complete": 0.0, "enemies": 0, "coins": 0,
                    "jumps": 0, "tube_issues": 0, "empty_ratio": 0.0, "noise": 0.0,
                },
                valid=False,
                errors=[f"Failed to evaluate level: {e}"],
                problem_type="smb",
            )

        # Extract metrics
        complete = info.get("complete", 0.0)
        enemies = info.get("enemies", 0)
        coins = info.get("coins", 0)
        jumps = info.get("jumps", 0)
        tube = info.get("tube", 0)
        empty = info.get("empty", 0.0)
        noise = info.get("noise", [])
        noise_sum = float(np.sum(noise)) if len(noise) > 0 else 0.0

        metrics = {
            "complete": complete,
            "enemies": enemies,
            "coins": coins,
            "jumps": jumps,
            "tube_issues": tube,
            "empty_ratio": empty,
            "noise": noise_sum,
        }

        # Build targets dict
        targets_dict = {
            "complete": {"mode": "target", "value": 1.0},
            "tube_issues": {"mode": "target", "value": 0},
            "noise": {"mode": "minimize"},
        }

        # Add controllable metric targets if configured
        enemies_cfg = self.config.targets.enemies
        if enemies_cfg:
            targets_dict["enemies"] = {"mode": enemies_cfg.mode, "value": enemies_cfg.value}
        coins_cfg = self.config.targets.coins
        if coins_cfg:
            targets_dict["coins"] = {"mode": coins_cfg.mode, "value": coins_cfg.value}
        jumps_cfg = self.config.targets.jumps
        if jumps_cfg:
            targets_dict["jumps"] = {"mode": jumps_cfg.mode, "value": jumps_cfg.value}

        # Check validity
        valid = True

        if complete < 1.0:
            valid = False
            errors.append(f"Level not completable (completion: {complete*100:.1f}%)")

        if tube > 0:
            errors.append(f"Level has {tube} tube issues")

        return EvalResult(
            metrics=metrics,
            targets=targets_dict,
            valid=valid,
            errors=errors,
            info=info,
            problem_type="smb",
        )

    def format_metrics_for_prompt(self, result: EvalResult) -> str:
        """Format evaluation result for LLM prompt.

        Args:
            result: EvalResult to format.

        Returns:
            Human-readable string describing metrics and targets.
        """
        if result.problem_type == 'zelda':
            return self._format_zelda_metrics(result)
        elif result.problem_type == 'sokoban':
            return self._format_sokoban_metrics(result)
        elif result.problem_type == 'loderunner':
            return self._format_loderunner_metrics(result)
        elif result.problem_type == 'smb':
            return self._format_smb_metrics(result)
        elif result.problem_type == 'binarydoor':
            return self._format_binarydoor_metrics(result)
        else:
            return self._format_binary_metrics(result)

    def _format_binary_metrics(self, result: EvalResult) -> str:
        """Format binary maze metrics."""
        lines = ["Current Level Metrics:"]

        # Path metric
        path_val = result.metrics.get("path", 0)
        path_target = self.config.targets.path
        if path_target.mode == "target" and path_target.value is not None:
            lines.append(f"  - path: {path_val} (goal: ~{path_target.value})")
        else:
            lines.append(f"  - path: {path_val} (goal: maximize, min_threshold: {path_target.min_threshold})")

        # Region metric
        regions_val = result.metrics.get("num_connected_regions", 0)
        region_cfg = self.config.targets.num_connected_regions
        if region_cfg.mode == "target":
            regions_target = region_cfg.value if region_cfg.value is not None else 1
            lines.append(f"  - num_connected_regions: {regions_val} (goal: ~{regions_target})")
        else:
            lines.append(f"  - num_connected_regions: {regions_val} (goal: maximize)")

        # Traversability
        traversable = path_val > 0
        lines.append(f"  - traversable: {'yes' if traversable else 'NO - no path exists'}")

        if not result.valid:
            lines.append(f"\nWarning: Level is invalid - {', '.join(result.errors)}")

        return "\n".join(lines)

    def _format_binarydoor_metrics(self, result: EvalResult) -> str:
        """Format binary door maze metrics."""
        lines = ["Current Level Metrics:"]

        # Door path metric
        door_path_val = result.metrics.get("door_path", 0)
        door_path_cfg = self.config.targets.door_path
        if door_path_cfg and door_path_cfg.mode == "target" and door_path_cfg.value is not None:
            lines.append(f"  - door_path: {door_path_val} (goal: ~{door_path_cfg.value})")
        elif door_path_cfg and door_path_cfg.mode == "maximize":
            threshold = door_path_cfg.min_threshold or 0
            lines.append(f"  - door_path: {door_path_val} (goal: maximize, min_threshold: {threshold})")
        else:
            lines.append(f"  - door_path: {door_path_val}")

        # Region metric
        regions_val = result.metrics.get("num_connected_regions", 0)
        region_cfg = self.config.targets.num_connected_regions
        if region_cfg.mode == "target":
            regions_target = region_cfg.value if region_cfg.value is not None else 1
            lines.append(f"  - num_connected_regions: {regions_val} (goal: ~{regions_target})")
        else:
            lines.append(f"  - num_connected_regions: {regions_val} (goal: maximize)")

        # Door positions from info
        if result.info:
            door1 = result.info.get("door1", None)
            door2 = result.info.get("door2", None)
            if door1 and door2:
                lines.append(f"  - door1 position: row={door1[0]}, col={door1[1]} (in bordered grid)")
                lines.append(f"  - door2 position: row={door2[0]}, col={door2[1]} (in bordered grid)")

        # Connectivity
        connected = door_path_val > 0
        lines.append(f"  - doors connected: {'yes' if connected else 'NO - no path between doors'}")

        if not result.valid:
            lines.append(f"\nWarning: Level is invalid - {', '.join(result.errors)}")

        return "\n".join(lines)

    def _format_zelda_metrics(self, result: EvalResult) -> str:
        """Format Zelda metrics."""
        lines = ["Current Level Metrics:"]

        # Entity counts
        lines.append(f"  - players: {result.metrics.get('players', 0)} (goal: exactly 1)")
        lines.append(f"  - keys: {result.metrics.get('keys', 0)} (goal: exactly 1)")
        lines.append(f"  - doors: {result.metrics.get('doors', 0)} (goal: exactly 1)")

        # Enemies
        enemies = result.metrics.get('enemies', 0)
        if self.config.targets.enemies and self.config.targets.enemies.value:
            lines.append(f"  - enemies: {enemies} (goal: ~{self.config.targets.enemies.value})")
        else:
            lines.append(f"  - enemies: {enemies}")

        # Paths
        player_key = result.metrics.get('player_key', 0)
        key_door = result.metrics.get('key_door', 0)
        solution_length = result.metrics.get('solution_length', 0)

        pk_cfg = self.config.targets.player_key
        if pk_cfg and pk_cfg.mode == "target" and pk_cfg.value is not None:
            lines.append(f"  - player_key path: {player_key} (goal: ~{pk_cfg.value})")
        elif pk_cfg and pk_cfg.mode == "maximize":
            lines.append(f"  - player_key path: {player_key} (goal: maximize)")
        else:
            lines.append(f"  - player_key path: {player_key} (player to key distance)")
        kd_cfg = self.config.targets.key_door
        if kd_cfg and kd_cfg.mode == "target" and kd_cfg.value is not None:
            lines.append(f"  - key_door path: {key_door} (goal: ~{kd_cfg.value})")
        elif kd_cfg and kd_cfg.mode == "maximize":
            lines.append(f"  - key_door path: {key_door} (goal: maximize)")
        else:
            lines.append(f"  - key_door path: {key_door} (key to door distance)")
        sol_cfg = self.config.targets.solution_length
        if sol_cfg and sol_cfg.mode == "target" and sol_cfg.value is not None:
            lines.append(f"  - solution_length: {solution_length} (total path, goal: ~{sol_cfg.value})")
        else:
            lines.append(f"  - solution_length: {solution_length} (total path, goal: maximize)")

        # Regions
        regions = result.metrics.get('regions', 0)
        lines.append(f"  - regions: {regions} (goal: exactly 1)")

        # Playability
        playable = player_key > 0 and key_door > 0
        lines.append(f"  - playable: {'yes' if playable else 'NO'}")

        if not result.valid:
            lines.append(f"\nWarning: Level is invalid - {', '.join(result.errors)}")

        return "\n".join(lines)

    def _format_sokoban_metrics(self, result: EvalResult) -> str:
        """Format Sokoban metrics."""
        lines = ["Current Level Metrics:"]

        # Entity counts
        players = result.metrics.get('players', 0)
        crates = result.metrics.get('crates', 0)
        targets = result.metrics.get('targets', 0)
        lines.append(f"  - players: {players} (goal: exactly 1)")

        # Target crates
        crate_cfg = self.config.targets.crates
        if crate_cfg and crate_cfg.mode == "target" and crate_cfg.value is not None:
            lines.append(f"  - crates: {crates} (goal: ~{crate_cfg.value})")
        elif crate_cfg and crate_cfg.mode == "maximize":
            lines.append(f"  - crates: {crates} (goal: maximize)")
        else:
            lines.append(f"  - crates: {crates}")
        lines.append(f"  - targets: {targets} (goal: = crates)")

        # Balance check
        balanced = crates == targets
        lines.append(f"  - balanced: {'yes' if balanced else 'NO'} (crates must equal targets)")

        # Solvability (solution_length > 0 means solver found a real solution)
        solution_length = result.metrics.get('solution_length', 0)
        if solution_length > 0:
            lines.append(f"  - solvable: yes (solution length: {solution_length})")
        else:
            heuristic = result.metrics.get('heuristic', -1)
            if heuristic < 0:
                lines.append(f"  - solvable: NO (structural issues prevent solving)")
            else:
                lines.append(f"  - solvable: NO (solver ran but could not find solution)")

        lines.append(f"  - solution_length: {solution_length} (goal: maximize)")

        if not result.valid:
            lines.append(f"\nWarning: Level is invalid - {', '.join(result.errors)}")

        return "\n".join(lines)

    def _format_loderunner_metrics(self, result: EvalResult) -> str:
        """Format Lode Runner metrics."""
        lines = ["Current Level Metrics:"]

        # Entity counts
        player = result.metrics.get('player', 0)
        gold = result.metrics.get('gold', 0)
        enemy = result.metrics.get('enemy', 0)
        ladder = result.metrics.get('ladder', 0)
        rope = result.metrics.get('rope', 0)

        lines.append(f"  - player: {player} (goal: exactly 1)")

        target_gold = 3
        if self.config.targets.gold and self.config.targets.gold.value is not None:
            target_gold = self.config.targets.gold.value
        lines.append(f"  - gold: {gold} (goal: ~{target_gold})")

        target_enemy = 2
        if self.config.targets.enemies and self.config.targets.enemies.value is not None:
            target_enemy = self.config.targets.enemies.value
        lines.append(f"  - enemy: {enemy} (goal: ~{target_enemy})")

        ladder_cfg = self.config.targets.ladder
        if ladder_cfg and ladder_cfg.mode == "target" and ladder_cfg.value is not None:
            lines.append(f"  - ladder: {ladder} (goal: ~{ladder_cfg.value})")
        elif ladder_cfg and ladder_cfg.mode == "maximize":
            lines.append(f"  - ladder: {ladder} (goal: maximize)")
        else:
            lines.append(f"  - ladder: {ladder}")
        rope_cfg = self.config.targets.rope
        if rope_cfg and rope_cfg.mode == "target" and rope_cfg.value is not None:
            lines.append(f"  - rope: {rope} (goal: ~{rope_cfg.value})")
        elif rope_cfg and rope_cfg.mode == "maximize":
            lines.append(f"  - rope: {rope} (goal: maximize)")
        else:
            lines.append(f"  - rope: {rope}")

        # Collectibility
        collected_gold = result.metrics.get('collected_gold', 0)
        used_tiles = result.metrics.get('used_tiles', 0)
        tiles = result.metrics.get('tiles', 0)

        if gold > 0:
            collect_pct = collected_gold / gold * 100
            lines.append(f"  - collected_gold: {collected_gold}/{gold} ({collect_pct:.0f}%)")
        else:
            lines.append(f"  - collected_gold: 0 (no gold)")

        if tiles > 0:
            used_pct = used_tiles / tiles * 100
            lines.append(f"  - accessibility: {used_tiles}/{tiles} tiles reachable ({used_pct:.0f}%)")

        playable = player == 1 and gold > 0 and collected_gold >= gold
        lines.append(f"  - playable: {'yes' if playable else 'NO'}")

        if not result.valid:
            lines.append(f"\nWarning: Level is invalid - {', '.join(result.errors)}")

        return "\n".join(lines)

    def _format_smb_metrics(self, result: EvalResult) -> str:
        """Format Super Mario Bros metrics."""
        lines = ["Current Level Metrics:"]

        complete = result.metrics.get('complete', 0.0)
        enemies = result.metrics.get('enemies', 0)
        coins = result.metrics.get('coins', 0)
        jumps = result.metrics.get('jumps', 0)
        tube_issues = result.metrics.get('tube_issues', 0)
        empty_ratio = result.metrics.get('empty_ratio', 0.0)
        noise = result.metrics.get('noise', 0.0)

        lines.append(f"  - complete: {complete*100:.1f}% (goal: 100%)")
        enemies_cfg = self.config.targets.enemies
        if enemies_cfg and enemies_cfg.mode == "target" and enemies_cfg.value is not None:
            lines.append(f"  - enemies_killed: {enemies} (goal: ~{enemies_cfg.value})")
        elif enemies_cfg and enemies_cfg.mode == "maximize":
            lines.append(f"  - enemies_killed: {enemies} (goal: maximize)")
        else:
            lines.append(f"  - enemies_killed: {enemies}")
        coins_cfg = self.config.targets.coins
        if coins_cfg and coins_cfg.mode == "target" and coins_cfg.value is not None:
            lines.append(f"  - coins_collected: {coins} (goal: ~{coins_cfg.value})")
        elif coins_cfg and coins_cfg.mode == "maximize":
            lines.append(f"  - coins_collected: {coins} (goal: maximize)")
        else:
            lines.append(f"  - coins_collected: {coins}")
        # Count stomp bounces from game events (not counted as jumps)
        stomp_bounces = 0
        if result.info:
            for ge in result.info.get("game_events", []):
                if ge.get("type") == 2:  # STOMP_KILL
                    stomp_bounces += 1

        bounce_note = f" [stomp bounces: {stomp_bounces}, not counted]" if stomp_bounces > 0 else ""
        jumps_cfg = self.config.targets.jumps
        if jumps_cfg and jumps_cfg.mode == "target" and jumps_cfg.value is not None:
            lines.append(f"  - jumps: {jumps} (goal: ~{jumps_cfg.value}){bounce_note}")
        elif jumps_cfg and jumps_cfg.mode == "maximize":
            lines.append(f"  - jumps: {jumps} (goal: maximize){bounce_note}")
        else:
            lines.append(f"  - jumps: {jumps}{bounce_note}")
        lines.append(f"  - tube_issues: {tube_issues} (goal: 0)")
        lines.append(f"  - empty_ratio: {empty_ratio*100:.1f}%")
        lines.append(f"  - noise: {noise:.3f} (goal: minimize)")

        # Simulation status
        if result.info:
            locations = result.info.get("locations", [])
            if locations:
                lines.append(f"  - simulation: ran")
            else:
                lines.append(f"  - simulation: skipped (structural pre-check failed)")
        else:
            lines.append(f"  - simulation: no data")

        playable = complete >= 1.0
        lines.append(f"  - playable: {'yes' if playable else 'NO'}")

        if not result.valid:
            lines.append(f"\nWarning: Level is invalid - {', '.join(result.errors)}")

        return "\n".join(lines)
