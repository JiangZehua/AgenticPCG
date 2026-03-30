"""Calculate stats evaluation tool (supports Binary and Zelda)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..level import Level
from ..evaluator import StatsEvaluator, EvalResult
from ..config import get_problem_type
from .registry import Tool, ToolResult

if TYPE_CHECKING:
    from ..edit_manager import EditManager


class CalculateStatsTool(Tool):
    """Tool for calculating level statistics.

    Uses EditManager to get current level state.
    Works for both Binary and Zelda problems.
    """

    def __init__(self, evaluator: StatsEvaluator, edit_manager: "EditManager"):
        """Initialize with evaluator and edit manager.

        Args:
            evaluator: StatsEvaluator instance.
            edit_manager: EditManager for accessing current level.
        """
        self._evaluator = evaluator
        self._edit_manager = edit_manager
        self._last_result: EvalResult | None = None

    def set_edit_manager(self, edit_manager: "EditManager") -> None:
        """Update the edit manager reference.

        Args:
            edit_manager: New edit manager.
        """
        self._edit_manager = edit_manager

    @property
    def name(self) -> str:
        return "calculate_stats"

    @property
    def description(self) -> str:
        problem_type = self._evaluator.problem_type
        if problem_type == 'zelda':
            return (
                "Calculates statistics for the current Zelda level including: "
                "players, keys, doors, enemies (entity counts), "
                "player_key (distance from player to key), key_door (distance from key to door), "
                "solution_length (total path), regions (connected regions). "
                "Returns metrics, targets, and gaps to help guide optimization."
            )
        elif problem_type == 'sokoban':
            return (
                "Calculates statistics for the current Sokoban level including: "
                "players (should be exactly 1), crates (number of pushable boxes), "
                "targets (number of goal locations), heuristic (solvability: -1=unsolvable, 0=optimal, >0=heuristic), "
                "solution_length (number of moves in solution). "
                "Returns metrics, targets, and gaps to help guide optimization."
            )
        elif problem_type == 'loderunner':
            return (
                "Calculates statistics for the current Lode Runner level including: "
                "player (should be exactly 1), gold (collectible items), enemy (enemies), "
                "ladder, rope (traversal aids), collected_gold (gold the player can reach). "
                "Returns metrics, targets, and gaps to help guide optimization."
            )
        elif problem_type == 'smb':
            return (
                "Runs a gameplay simulation of the current Super Mario Bros level. "
                "A Mario AI agent plays the level from left to right. Returns gameplay outcome metrics: "
                "complete (did the agent reach the flag, 0-1), "
                "enemies_killed (enemies the agent stomped/hit during play - NOT tile count), "
                "coins_collected (coins the agent picked up during play - NOT tile count), "
                "jumps (jumps the agent performed during play), "
                "tube_issues (malformed tube count), empty_ratio (open space ratio). "
                "Returns metrics, targets, and gaps to help guide optimization."
            )
        elif problem_type == 'binarydoor':
            return (
                "Calculates statistics for the current binary door maze level including door_path "
                "(the shortest path between the two door openings in the border) and num_connected_regions "
                "(number of separate connected regions of empty tiles). "
                "Returns metrics, targets, and gaps to help guide optimization."
            )
        else:
            return (
                "Calculates statistics for the current level including path "
                "(the longest shortest path between any two empty tiles) and num_connected_regions "
                "(number of separate connected regions of empty tiles). "
                "Returns metrics, targets, and gaps to help guide optimization."
            )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "level_string": {
                    "type": "string",
                    "description": "Optional: ASCII level representation to evaluate. If not provided, evaluates the current level.",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute stats calculation.

        Args:
            level_string: Optional level string to evaluate instead of current level.

        Returns:
            ToolResult with metrics, targets, gaps, and formatted summary.
        """
        level_string = kwargs.get("level_string")
        problem_type = self._edit_manager.problem_type

        if level_string:
            # Parse provided level string
            try:
                level = Level.from_string(level_string, problem_type=problem_type)
            except Exception as e:
                return ToolResult(
                    success=False,
                    errors=[f"Failed to parse level string: {e}"],
                )
        else:
            # Get current level from EditManager
            level = self._edit_manager.level

        # Evaluate level
        result = self._evaluator.evaluate(level)
        self._last_result = result

        if not result.valid:
            return ToolResult(
                success=False,
                data={
                    "metrics": result.metrics,
                    "valid": False,
                    "errors": result.errors,
                },
                errors=result.errors,
            )

        # Compute gaps to targets based on problem type
        if problem_type == 'zelda':
            gaps, summary_lines = self._compute_zelda_gaps(result)
        elif problem_type == 'sokoban':
            gaps, summary_lines = self._compute_sokoban_gaps(result)
        elif problem_type == 'loderunner':
            gaps, summary_lines = self._compute_loderunner_gaps(result)
        elif problem_type == 'smb':
            gaps, summary_lines = self._compute_smb_gaps(result)
        elif problem_type == 'binarydoor':
            gaps, summary_lines = self._compute_binarydoor_gaps(result)
        else:
            gaps, summary_lines = self._compute_binary_gaps(result)

        return ToolResult(
            success=True,
            data={
                "metrics": result.metrics,
                "targets": result.targets,
                "gaps": gaps,
                "valid": result.valid,
                "summary": "\n".join(summary_lines),
            },
        )

    def _compute_binary_gaps(self, result: EvalResult) -> tuple[dict, list[str]]:
        """Compute gaps for binary maze."""
        gaps = {}

        # Path gap
        path_val = result.metrics.get("path", 0)
        path_target = self._evaluator.config.targets.path
        if path_target.mode == "target" and path_target.value is not None:
            gaps["path"] = {
                "current": path_val,
                "target": path_target.value,
                "gap": abs(path_val - path_target.value),
                "status": "ok" if path_val == path_target.value else "off_target",
            }
        else:
            # maximize mode
            threshold = path_target.min_threshold or 0
            gaps["path"] = {
                "current": path_val,
                "target": f">= {threshold} (maximize)",
                "gap": max(0, threshold - path_val),
                "status": "ok" if path_val >= threshold else "below_threshold",
            }

        # Region gap
        regions_val = result.metrics.get("num_connected_regions", 0)
        region_cfg = self._evaluator.config.targets.num_connected_regions
        if region_cfg.mode == "target":
            regions_target = region_cfg.value if region_cfg.value is not None else 1
            gaps["num_connected_regions"] = {
                "current": regions_val,
                "target": regions_target,
                "gap": abs(regions_val - regions_target),
                "status": "ok" if regions_val == regions_target else "off_target",
            }
        else:
            # maximize mode
            gaps["num_connected_regions"] = {
                "current": regions_val,
                "target": "maximize",
                "gap": 0,
                "status": "ok",
            }

        # Format summary
        summary_lines = [
            "Level Statistics:",
            f"  path: {path_val}",
            f"  num_connected_regions: {regions_val}",
            "",
            "Target Gaps:",
        ]
        for metric, gap_info in gaps.items():
            summary_lines.append(f"  {metric}: {gap_info['status']} (gap: {gap_info['gap']})")

        return gaps, summary_lines

    def _compute_binarydoor_gaps(self, result: EvalResult) -> tuple[dict, list[str]]:
        """Compute gaps for binary door maze."""
        gaps = {}

        # Door path gap
        door_path_val = result.metrics.get("door_path", 0)
        door_path_cfg = self._evaluator.config.targets.door_path
        if door_path_cfg and door_path_cfg.mode == "target" and door_path_cfg.value is not None:
            gaps["door_path"] = {
                "current": door_path_val,
                "target": door_path_cfg.value,
                "gap": abs(door_path_val - door_path_cfg.value),
                "status": "ok" if door_path_val == door_path_cfg.value else "off_target",
            }
        elif door_path_cfg and door_path_cfg.mode == "maximize":
            threshold = door_path_cfg.min_threshold or 0
            gaps["door_path"] = {
                "current": door_path_val,
                "target": f">= {threshold} (maximize)",
                "gap": max(0, threshold - door_path_val),
                "status": "ok" if door_path_val >= threshold else "below_threshold",
            }
        else:
            gaps["door_path"] = {
                "current": door_path_val,
                "target": "> 0 (connected)",
                "gap": 0 if door_path_val > 0 else 1,
                "status": "ok" if door_path_val > 0 else "disconnected",
            }

        # Region gap
        regions_val = result.metrics.get("num_connected_regions", 0)
        region_cfg = self._evaluator.config.targets.num_connected_regions
        if region_cfg.mode == "target":
            regions_target = region_cfg.value if region_cfg.value is not None else 1
            gaps["num_connected_regions"] = {
                "current": regions_val,
                "target": regions_target,
                "gap": abs(regions_val - regions_target),
                "status": "ok" if regions_val == regions_target else "off_target",
            }
        else:
            gaps["num_connected_regions"] = {
                "current": regions_val,
                "target": "maximize",
                "gap": 0,
                "status": "ok",
            }

        # Format summary
        summary_lines = [
            "Level Statistics:",
            f"  door_path: {door_path_val}",
            f"  num_connected_regions: {regions_val}",
            f"  doors connected: {'YES' if door_path_val > 0 else 'NO'}",
            "",
            "Target Gaps:",
        ]
        for metric, gap_info in gaps.items():
            summary_lines.append(f"  {metric}: {gap_info['status']} (gap: {gap_info['gap']})")

        return gaps, summary_lines

    def _compute_zelda_gaps(self, result: EvalResult) -> tuple[dict, list[str]]:
        """Compute gaps for Zelda level."""
        gaps = {}

        # Entity counts - all should be exactly 1
        for entity in ["players", "keys", "doors"]:
            val = result.metrics.get(entity, 0)
            gaps[entity] = {
                "current": val,
                "target": 1,
                "gap": abs(val - 1),
                "status": "ok" if val == 1 else "off_target",
            }

        # Enemies - target depends on config
        enemies_val = result.metrics.get("enemies", 0)
        target_enemies = 3  # Default for zelda-v0
        if self._evaluator.config.targets.enemies and self._evaluator.config.targets.enemies.value:
            target_enemies = self._evaluator.config.targets.enemies.value
        gaps["enemies"] = {
            "current": enemies_val,
            "target": target_enemies,
            "gap": abs(enemies_val - target_enemies),
            "status": "ok" if enemies_val == target_enemies else "off_target",
        }

        # Paths - mode-aware gaps
        player_key = result.metrics.get("player_key", 0)
        key_door = result.metrics.get("key_door", 0)
        solution_length = result.metrics.get("solution_length", 0)

        pk_cfg = self._evaluator.config.targets.player_key
        if pk_cfg and pk_cfg.mode == "target" and pk_cfg.value is not None:
            gaps["player_key"] = {
                "current": player_key,
                "target": pk_cfg.value,
                "gap": abs(player_key - pk_cfg.value),
                "status": "ok" if player_key == pk_cfg.value else "off_target",
            }
        elif pk_cfg and pk_cfg.mode == "maximize":
            gaps["player_key"] = {
                "current": player_key,
                "target": "maximize",
                "gap": 0 if player_key > 0 else 1,
                "status": "ok" if player_key > 0 else "unreachable",
            }
        else:
            gaps["player_key"] = {
                "current": player_key,
                "target": "> 0 (reachable)",
                "gap": 0 if player_key > 0 else 1,
                "status": "ok" if player_key > 0 else "unreachable",
            }

        kd_cfg = self._evaluator.config.targets.key_door
        if kd_cfg and kd_cfg.mode == "target" and kd_cfg.value is not None:
            gaps["key_door"] = {
                "current": key_door,
                "target": kd_cfg.value,
                "gap": abs(key_door - kd_cfg.value),
                "status": "ok" if key_door == kd_cfg.value else "off_target",
            }
        elif kd_cfg and kd_cfg.mode == "maximize":
            gaps["key_door"] = {
                "current": key_door,
                "target": "maximize",
                "gap": 0 if key_door > 0 else 1,
                "status": "ok" if key_door > 0 else "unreachable",
            }
        else:
            gaps["key_door"] = {
                "current": key_door,
                "target": "> 0 (reachable)",
                "gap": 0 if key_door > 0 else 1,
                "status": "ok" if key_door > 0 else "unreachable",
            }

        sol_cfg = self._evaluator.config.targets.solution_length
        if sol_cfg and sol_cfg.mode == "target" and sol_cfg.value is not None:
            gaps["solution_length"] = {
                "current": solution_length,
                "target": sol_cfg.value,
                "gap": abs(solution_length - sol_cfg.value),
                "status": "ok" if solution_length == sol_cfg.value else "off_target",
            }
        else:
            gaps["solution_length"] = {
                "current": solution_length,
                "target": "maximize",
                "gap": 0,
                "status": "ok" if solution_length > 0 else "no_path",
            }

        # Regions
        regions_val = result.metrics.get("regions", 0)
        gaps["regions"] = {
            "current": regions_val,
            "target": 1,
            "gap": abs(regions_val - 1),
            "status": "ok" if regions_val == 1 else "off_target",
        }

        # Format summary
        playable = player_key > 0 and key_door > 0
        summary_lines = [
            "Level Statistics:",
            f"  players: {result.metrics.get('players', 0)}",
            f"  keys: {result.metrics.get('keys', 0)}",
            f"  doors: {result.metrics.get('doors', 0)}",
            f"  enemies: {enemies_val}",
            f"  player_key path: {player_key}",
            f"  key_door path: {key_door}",
            f"  solution_length: {solution_length}",
            f"  regions: {regions_val}",
            f"  playable: {'YES' if playable else 'NO'}",
            "",
            "Target Gaps:",
        ]

        for metric in ["players", "keys", "doors", "enemies", "player_key", "key_door", "regions"]:
            gap_info = gaps[metric]
            summary_lines.append(f"  {metric}: {gap_info['status']} (gap: {gap_info['gap']})")

        return gaps, summary_lines

    def _compute_sokoban_gaps(self, result: EvalResult) -> tuple[dict, list[str]]:
        """Compute gaps for Sokoban level."""
        gaps = {}

        # Player count - should be exactly 1
        players_val = result.metrics.get("players", 0)
        gaps["players"] = {
            "current": players_val,
            "target": 1,
            "gap": abs(players_val - 1),
            "status": "ok" if players_val == 1 else "off_target",
        }

        # Crates - target depends on config
        crates_val = result.metrics.get("crates", 0)
        target_crates = 3  # Default
        if self._evaluator.config.targets.crates and self._evaluator.config.targets.crates.value:
            target_crates = self._evaluator.config.targets.crates.value
        gaps["crates"] = {
            "current": crates_val,
            "target": target_crates,
            "gap": abs(crates_val - target_crates),
            "status": "ok" if crates_val == target_crates else "off_target",
        }

        # Targets - should equal crates
        targets_val = result.metrics.get("targets", 0)
        gaps["targets"] = {
            "current": targets_val,
            "target": f"= crates ({crates_val})",
            "gap": abs(targets_val - crates_val),
            "status": "ok" if targets_val == crates_val else "unbalanced",
        }

        # Balance check
        balanced = crates_val == targets_val
        gaps["balance"] = {
            "current": f"crates={crates_val}, targets={targets_val}",
            "target": "crates == targets",
            "gap": abs(crates_val - targets_val),
            "status": "ok" if balanced else "unbalanced",
        }

        # Solvability - heuristic should be >= 0
        heuristic = result.metrics.get("heuristic", -1)
        solution_length = result.metrics.get("solution_length", 0)
        gaps["solvability"] = {
            "current": heuristic,
            "target": ">= 0 (solvable)",
            "gap": 0 if heuristic >= 0 else 1,
            "status": "ok" if heuristic >= 0 else "unsolvable",
        }

        gaps["solution_length"] = {
            "current": solution_length,
            "target": "maximize",
            "gap": 0,
            "status": "ok" if solution_length > 0 else "no_solution",
        }

        # Format summary
        solvable = heuristic >= 0
        summary_lines = [
            "Level Statistics:",
            f"  players: {players_val}",
            f"  crates: {crates_val}",
            f"  targets: {targets_val}",
            f"  balanced: {'YES' if balanced else 'NO'} (crates == targets)",
            f"  heuristic: {heuristic}",
            f"  solution_length: {solution_length}",
            f"  solvable: {'YES' if solvable else 'NO'}",
            "",
            "Target Gaps:",
        ]

        for metric in ["players", "crates", "targets", "balance", "solvability"]:
            gap_info = gaps[metric]
            summary_lines.append(f"  {metric}: {gap_info['status']} (gap: {gap_info['gap']})")

        return gaps, summary_lines

    def _compute_loderunner_gaps(self, result: EvalResult) -> tuple[dict, list[str]]:
        """Compute gaps for Lode Runner level."""
        gaps = {}

        # Player count - should be exactly 1
        player_val = result.metrics.get("player", 0)
        gaps["player"] = {
            "current": player_val,
            "target": 1,
            "gap": abs(player_val - 1),
            "status": "ok" if player_val == 1 else "off_target",
        }

        # Gold - target depends on config
        gold_val = result.metrics.get("gold", 0)
        target_gold = 3
        if self._evaluator.config.targets.gold and self._evaluator.config.targets.gold.value:
            target_gold = self._evaluator.config.targets.gold.value
        gaps["gold"] = {
            "current": gold_val,
            "target": target_gold,
            "gap": abs(gold_val - target_gold),
            "status": "ok" if gold_val == target_gold else "off_target",
        }

        # Enemy
        enemy_val = result.metrics.get("enemy", 0)
        target_enemy = 2
        if self._evaluator.config.targets.enemies and self._evaluator.config.targets.enemies.value:
            target_enemy = self._evaluator.config.targets.enemies.value
        gaps["enemy"] = {
            "current": enemy_val,
            "target": target_enemy,
            "gap": abs(enemy_val - target_enemy),
            "status": "ok" if enemy_val == target_enemy else "off_target",
        }

        # Collected gold - should equal total gold
        collected_gold = result.metrics.get("collected_gold", 0)
        gaps["collected_gold"] = {
            "current": collected_gold,
            "target": f"= gold ({gold_val})",
            "gap": gold_val - collected_gold if gold_val > 0 else 0,
            "status": "ok" if gold_val > 0 and collected_gold == gold_val else "not_all_collectible",
        }

        # Ladder gap (mode-aware)
        ladder_val = result.metrics.get("ladder", 0)
        ladder_cfg = self._evaluator.config.targets.ladder
        if ladder_cfg and ladder_cfg.mode == "target" and ladder_cfg.value is not None:
            gaps["ladder"] = {
                "current": ladder_val,
                "target": ladder_cfg.value,
                "gap": abs(ladder_val - ladder_cfg.value),
                "status": "ok" if ladder_val == ladder_cfg.value else "off_target",
            }
        elif ladder_cfg and ladder_cfg.mode == "maximize":
            gaps["ladder"] = {
                "current": ladder_val,
                "target": "maximize",
                "gap": 0,
                "status": "ok",
            }

        # Rope gap (mode-aware)
        rope_val = result.metrics.get("rope", 0)
        rope_cfg = self._evaluator.config.targets.rope
        if rope_cfg and rope_cfg.mode == "target" and rope_cfg.value is not None:
            gaps["rope"] = {
                "current": rope_val,
                "target": rope_cfg.value,
                "gap": abs(rope_val - rope_cfg.value),
                "status": "ok" if rope_val == rope_cfg.value else "off_target",
            }
        elif rope_cfg and rope_cfg.mode == "maximize":
            gaps["rope"] = {
                "current": rope_val,
                "target": "maximize",
                "gap": 0,
                "status": "ok",
            }

        # Format summary
        all_collectible = gold_val > 0 and collected_gold == gold_val
        summary_lines = [
            "Level Statistics:",
            f"  player: {player_val}",
            f"  gold: {gold_val}",
            f"  enemy: {enemy_val}",
            f"  ladder: {ladder_val}",
            f"  rope: {rope_val}",
            f"  collected_gold: {collected_gold}/{gold_val}",
            f"  all_collectible: {'YES' if all_collectible else 'NO'}",
            "",
            "Target Gaps:",
        ]

        gap_metrics = ["player", "gold", "enemy", "collected_gold"]
        if "ladder" in gaps:
            gap_metrics.append("ladder")
        if "rope" in gaps:
            gap_metrics.append("rope")
        for metric in gap_metrics:
            gap_info = gaps[metric]
            summary_lines.append(f"  {metric}: {gap_info['status']} (gap: {gap_info['gap']})")

        return gaps, summary_lines

    def _compute_smb_gaps(self, result: EvalResult) -> tuple[dict, list[str]]:
        """Compute gaps for Super Mario Bros level."""
        gaps = {}

        # Completion - should be 100%
        complete = result.metrics.get("complete", 0.0)
        gaps["complete"] = {
            "current": f"{complete*100:.1f}%",
            "target": "100%",
            "gap": 1.0 - complete,
            "status": "ok" if complete >= 1.0 else "incomplete",
        }

        # Tube issues - should be 0
        tube_issues = result.metrics.get("tube_issues", 0)
        gaps["tube_issues"] = {
            "current": tube_issues,
            "target": 0,
            "gap": tube_issues,
            "status": "ok" if tube_issues == 0 else "has_issues",
        }

        # Noise
        noise = result.metrics.get("noise", 0.0)
        gaps["noise"] = {
            "current": f"{noise:.3f}",
            "target": "minimize",
            "gap": noise,
            "status": "ok" if noise == 0 else "has_noise",
        }

        # Controllable metric gaps (mode-aware)
        enemies = result.metrics.get("enemies", 0)
        coins = result.metrics.get("coins", 0)
        jumps = result.metrics.get("jumps", 0)

        enemies_cfg = self._evaluator.config.targets.enemies
        if enemies_cfg and enemies_cfg.mode == "target" and enemies_cfg.value is not None:
            gaps["enemies_killed"] = {
                "current": enemies,
                "target": enemies_cfg.value,
                "gap": abs(enemies - enemies_cfg.value),
                "status": "ok" if enemies == enemies_cfg.value else "off_target",
            }
        elif enemies_cfg and enemies_cfg.mode == "maximize":
            gaps["enemies_killed"] = {
                "current": enemies,
                "target": "maximize",
                "gap": 0,
                "status": "ok",
            }

        coins_cfg = self._evaluator.config.targets.coins
        if coins_cfg and coins_cfg.mode == "target" and coins_cfg.value is not None:
            gaps["coins_collected"] = {
                "current": coins,
                "target": coins_cfg.value,
                "gap": abs(coins - coins_cfg.value),
                "status": "ok" if coins == coins_cfg.value else "off_target",
            }
        elif coins_cfg and coins_cfg.mode == "maximize":
            gaps["coins_collected"] = {
                "current": coins,
                "target": "maximize",
                "gap": 0,
                "status": "ok",
            }

        jumps_cfg = self._evaluator.config.targets.jumps
        if jumps_cfg and jumps_cfg.mode == "target" and jumps_cfg.value is not None:
            gaps["jumps_made"] = {
                "current": jumps,
                "target": jumps_cfg.value,
                "gap": abs(jumps - jumps_cfg.value),
                "status": "ok" if jumps == jumps_cfg.value else "off_target",
            }
        elif jumps_cfg and jumps_cfg.mode == "maximize":
            gaps["jumps_made"] = {
                "current": jumps,
                "target": "maximize",
                "gap": 0,
                "status": "ok",
            }

        # Format summary
        completable = complete >= 1.0

        # Count stomp bounces (not counted as jumps)
        stomp_bounces = 0
        if result.info:
            for ge in result.info.get("game_events", []):
                if ge.get("type") == 2:  # STOMP_KILL
                    stomp_bounces += 1
        jump_line = f"  jumps: {jumps}"
        if stomp_bounces > 0:
            jump_line += f" (stomp bounces: {stomp_bounces}, not counted)"

        summary_lines = [
            "Level Statistics:",
            f"  complete: {complete*100:.1f}%",
            f"  tube_issues: {tube_issues}",
            f"  noise: {noise:.3f}",
            f"  enemies_killed: {enemies}",
            f"  coins_collected: {coins}",
            jump_line,
            f"  completable: {'YES' if completable else 'NO'}",
            "",
            "Target Gaps:",
        ]

        gap_metrics = ["complete", "tube_issues", "noise"]
        if "enemies_killed" in gaps:
            gap_metrics.append("enemies_killed")
        if "coins_collected" in gaps:
            gap_metrics.append("coins_collected")
        if "jumps_made" in gaps:
            gap_metrics.append("jumps_made")
        for metric in gap_metrics:
            gap_info = gaps[metric]
            summary_lines.append(f"  {metric}: {gap_info['status']} (gap: {gap_info['gap']})")

        return gaps, summary_lines

    def get_last_result(self) -> EvalResult | None:
        """Get the last evaluation result (includes paths for rendering)."""
        return self._last_result
