"""Scoring function for level evaluation (Binary, Zelda, etc.)."""

from __future__ import annotations

from ..config import Config, get_problem_type
from ..evaluator import EvalResult


class Scorer:
    """Computes optimization score from level metrics."""

    def __init__(self, config: Config):
        """Initialize scorer with config.

        Args:
            config: Configuration with scoring weights.
        """
        self.config = config
        self.problem_type = get_problem_type(config.env.name)
        self.w_path = config.scoring.w_path
        self.w_regions = config.scoring.w_regions

    def _metric_score(
        self,
        actual: float,
        target_cfg: "TargetConfig | None",
        weight: float,
        default_target: float | None = None,
    ) -> float:
        """Score a metric based on its target configuration.

        For mode="target": -weight * |actual - target_value|
        For mode="maximize": +weight * actual
        If target_cfg is None, uses default_target if provided, else maximizes.
        """
        if target_cfg is not None:
            if target_cfg.mode == "target":
                target_val = target_cfg.value if target_cfg.value is not None else default_target
                if target_val is not None:
                    return -weight * abs(actual - target_val)
            # mode == "maximize" (or target mode with no value and no default)
            return weight * actual
        # No target config
        if default_target is not None:
            return -weight * abs(actual - default_target)
        return weight * actual

    def score(self, result: EvalResult) -> float:
        """Compute score from evaluation result.

        Higher score is better.

        Args:
            result: EvalResult with metrics.

        Returns:
            Scalar score value.
        """
        if result.problem_type == 'zelda':
            return self._score_zelda(result)
        elif result.problem_type == 'sokoban':
            return self._score_sokoban(result)
        elif result.problem_type == 'loderunner':
            return self._score_loderunner(result)
        elif result.problem_type == 'smb':
            return self._score_smb(result)
        elif result.problem_type == 'binarydoor':
            return self._score_binarydoor(result)
        else:
            return self._score_binary(result)

    def _score_binary(self, result: EvalResult) -> float:
        """Score for binary maze (target-aware)."""
        path = result.metrics.get("path", 0)
        regions = result.metrics.get("num_connected_regions", 0)

        score = self._metric_score(path, self.config.targets.path, self.w_path)
        score += self._metric_score(regions, self.config.targets.num_connected_regions, self.w_regions, default_target=1)

        # Solvability penalty (no path means maze is untraversable)
        if path == 0:
            score -= self.config.scoring.w_playable

        # Graduated change penalty: penalize when >60% of the board differs
        # from the last accepted level.  Penalty grows linearly beyond the
        # threshold so that larger rewrites are punished more.
        change_ratio = result.metrics.get("change_ratio", 0.0)
        change_threshold = 0.6
        if change_ratio > change_threshold:
            excess = change_ratio - change_threshold
            score -= self.config.scoring.w_change_penalty * excess

        return score

    def _score_binarydoor(self, result: EvalResult) -> float:
        """Score for binary door maze (target-aware)."""
        door_path = result.metrics.get("door_path", 0)
        regions = result.metrics.get("num_connected_regions", 0)

        w_door_path = self.config.scoring.w_door_path
        w_regions = self.config.scoring.w_regions
        w_connected = self.config.scoring.w_connected

        score = self._metric_score(door_path, self.config.targets.door_path, w_door_path)
        score += self._metric_score(regions, self.config.targets.num_connected_regions, w_regions, default_target=1)

        # Connectivity bonus/penalty (doors must be connected)
        if door_path > 0:
            score += w_connected
        else:
            score -= w_connected

        # Graduated change penalty: penalize when >60% of the board differs
        # from the last accepted level.
        change_ratio = result.metrics.get("change_ratio", 0.0)
        change_threshold = 0.6
        if change_ratio > change_threshold:
            excess = change_ratio - change_threshold
            score -= self.config.scoring.w_change_penalty * excess

        return score

    def _score_zelda(self, result: EvalResult) -> float:
        """Score for Zelda level.

        Score components (using configurable weights):
        - Player: +w_player if exactly 1, else -w_player * |count - 1|
        - Key: +w_key if exactly 1, else -w_key * |count - 1|
        - Door: +w_door if exactly 1, else -w_door * |count - 1|
        - Enemy: -w_enemy * |count - target|
        - Playability: +w_playable if level is solvable (both paths exist)
        - Solution length: +w_path * solution_length
        - Region penalty: -w_regions * |regions - 1|
        """
        players = result.metrics.get("players", 0)
        keys = result.metrics.get("keys", 0)
        doors = result.metrics.get("doors", 0)
        enemies = result.metrics.get("enemies", 0)
        player_key = result.metrics.get("player_key", 0)
        key_door = result.metrics.get("key_door", 0)
        solution_length = result.metrics.get("solution_length", 0)
        regions = result.metrics.get("regions", 0)

        # Get weights from config
        w_player = self.config.scoring.w_player
        w_key = self.config.scoring.w_key
        w_door = self.config.scoring.w_door
        w_enemy = self.config.scoring.w_enemy
        w_playable = self.config.scoring.w_playable

        score = 0.0

        # Player tile scoring (must have exactly 1)
        if players == 1:
            score += w_player
        else:
            score -= w_player * abs(players - 1)

        # Key tile scoring (must have exactly 1)
        if keys == 1:
            score += w_key
        else:
            score -= w_key * abs(keys - 1)

        # Door tile scoring (must have exactly 1)
        if doors == 1:
            score += w_door
        else:
            score -= w_door * abs(doors - 1)

        # Enemy tile scoring (should match target)
        target_enemies = 3  # Default for zelda-v0
        if self.config.targets.enemies and self.config.targets.enemies.value:
            target_enemies = self.config.targets.enemies.value
        if enemies == target_enemies:
            score += w_enemy * target_enemies  # Bonus for exact match
        else:
            score -= w_enemy * abs(enemies - target_enemies)

        # Playability bonus/penalty (paths must exist)
        if player_key > 0 and key_door > 0:
            score += w_playable
        else:
            score -= w_playable

        # Path length scoring (target-aware)
        pk_cfg = self.config.targets.player_key
        kd_cfg = self.config.targets.key_door
        if pk_cfg is not None or kd_cfg is not None:
            # Score individual path components when they have target configs
            if pk_cfg is not None:
                score += self._metric_score(player_key, pk_cfg, self.w_path)
            if kd_cfg is not None:
                score += self._metric_score(key_door, kd_cfg, self.w_path)
        else:
            # Fall back to solution_length scoring
            score += self._metric_score(solution_length, self.config.targets.solution_length, self.w_path)

        # Region penalty (want exactly 1)
        score -= self.w_regions * abs(regions - 1)

        # Graduated change penalty
        change_ratio = result.metrics.get("change_ratio", 0.0)
        change_threshold = 0.6
        if change_ratio > change_threshold:
            excess = change_ratio - change_threshold
            score -= self.config.scoring.w_change_penalty * excess

        return score

    def _score_sokoban(self, result: EvalResult) -> float:
        """Score for Sokoban level.

        Score components (using configurable weights):
        - Player: +w_player if exactly 1, else -w_player * |count - 1|
        - Crates: -w_crate * |count - target|
        - Targets: -w_target * |count - target|
        - Balance: -w_balance * |crates - targets|
        - Solvability: +w_solvable if solvable (heuristic >= 0)
        - Solution length: +w_solution * solution_length
        """
        players = result.metrics.get("players", 0)
        crates = result.metrics.get("crates", 0)
        targets = result.metrics.get("targets", 0)
        heuristic = result.metrics.get("heuristic", -1)
        solution_length = result.metrics.get("solution_length", 0)

        # Get weights from config
        w_player = self.config.scoring.w_player
        w_crate = self.config.scoring.w_crate
        w_target = self.config.scoring.w_target
        w_balance = self.config.scoring.w_balance
        w_solvable = self.config.scoring.w_solvable
        w_solution = self.config.scoring.w_solution

        # Target crates
        target_crates = 3
        if self.config.targets.crates and self.config.targets.crates.value is not None:
            target_crates = self.config.targets.crates.value

        score = 0.0

        # Player tile scoring (must have exactly 1)
        if players == 1:
            score += w_player
        else:
            score -= w_player * abs(players - 1)

        # Crate count scoring (should match target)
        if crates == target_crates:
            score += w_crate
        else:
            score -= w_crate * abs(crates - target_crates)

        # Target count scoring (should match crate target)
        if targets == target_crates:
            score += w_target
        else:
            score -= w_target * abs(targets - target_crates)

        # Balance penalty (crates must equal targets)
        if crates == targets:
            score += w_balance
        else:
            score -= w_balance * abs(crates - targets)

        # Solvability bonus/penalty (solution_length > 0 means truly solvable)
        if solution_length > 0:
            score += w_solvable
        else:
            score -= w_solvable

        # Solution length (maximize)
        score += w_solution * solution_length

        # Graduated change penalty
        change_ratio = result.metrics.get("change_ratio", 0.0)
        change_threshold = 0.6
        if change_ratio > change_threshold:
            excess = change_ratio - change_threshold
            score -= self.config.scoring.w_change_penalty * excess

        return score

    def _score_loderunner(self, result: EvalResult) -> float:
        """Score for Lode Runner level.

        Score components:
        - Player: +w_player if exactly 1
        - Gold: -w_gold * |count - target|
        - Enemy: -w_enemy * |count - target|
        - Collectible: +w_collected * (collected_gold / total_gold)
        - Ladder: target-aware scoring
        - Rope: target-aware scoring
        """
        player = result.metrics.get("player", 0)
        gold = result.metrics.get("gold", 0)
        enemy = result.metrics.get("enemy", 0)
        collected_gold = result.metrics.get("collected_gold", 0)
        ladder = result.metrics.get("ladder", 0)
        rope = result.metrics.get("rope", 0)

        # Get weights from config
        w_player = self.config.scoring.w_player
        w_gold = self.config.scoring.w_gold
        w_enemy = self.config.scoring.w_enemy
        w_collected = self.config.scoring.w_collected
        w_ladder = self.config.scoring.w_ladder
        w_rope = self.config.scoring.w_rope

        # Target gold and enemies
        target_gold = 3
        target_enemy = 2
        if self.config.targets.gold and self.config.targets.gold.value is not None:
            target_gold = self.config.targets.gold.value
        if self.config.targets.enemies and self.config.targets.enemies.value is not None:
            target_enemy = self.config.targets.enemies.value

        score = 0.0

        # Player tile scoring (must have exactly 1)
        if player == 1:
            score += w_player
        else:
            score -= w_player * abs(player - 1)

        # Gold count scoring
        if gold == target_gold:
            score += w_gold * target_gold
        else:
            score -= w_gold * abs(gold - target_gold)

        # Enemy count scoring
        if enemy == target_enemy:
            score += w_enemy * target_enemy
        else:
            score -= w_enemy * abs(enemy - target_enemy)

        # Collected gold bonus (maximize)
        if gold > 0:
            collect_ratio = collected_gold / gold
            score += w_collected * collect_ratio

        # Ladder scoring (target-aware)
        score += self._metric_score(ladder, self.config.targets.ladder, w_ladder)

        # Rope scoring (target-aware)
        score += self._metric_score(rope, self.config.targets.rope, w_rope)

        # Playability penalty — graduated by fraction of uncollected gold.
        # Full penalty (-w_playable) when no gold is placed or player is missing.
        # Partial penalty scaled by unreachable ratio when some gold exists:
        #   penalty = w_playable * (uncollected / total)
        # This lets the optimizer make incremental progress (e.g. 2/3 reachable
        # is better than 0/3) rather than hitting a flat cliff.
        w_playable = self.config.scoring.w_playable
        if player != 1 or gold == 0:
            score -= w_playable
        elif collected_gold < gold:
            unreachable_ratio = (gold - collected_gold) / gold
            score -= w_playable * unreachable_ratio

        # Graduated change penalty
        change_ratio = result.metrics.get("change_ratio", 0.0)
        change_threshold = 0.6
        if change_ratio > change_threshold:
            excess = change_ratio - change_threshold
            score -= self.config.scoring.w_change_penalty * excess

        return score

    def _score_smb(self, result: EvalResult) -> float:
        """Score for Super Mario Bros level.

        Score components:
        - Complete: +w_complete if 100% completion
        - Enemies: +w_enemy * enemies_killed
        - Coins: +w_coins * coins_collected
        - Jumps: +w_jumps * num_jumps
        """
        complete = result.metrics.get("complete", 0.0)
        enemies = result.metrics.get("enemies", 0)
        coins = result.metrics.get("coins", 0)
        jumps = result.metrics.get("jumps", 0)
        tube_issues = result.metrics.get("tube_issues", 0)

        # Get weights from config
        w_complete = self.config.scoring.w_complete
        w_enemy = self.config.scoring.w_enemy
        w_coins = self.config.scoring.w_coins
        w_jumps = self.config.scoring.w_jumps

        score = 0.0

        # Completion score (continuous, no cliff at 1.0)
        score += w_complete * complete

        # Tube issues penalty
        score -= 50 * tube_issues

        # Enemies killed (target-aware)
        score += self._metric_score(enemies, self.config.targets.enemies, w_enemy)

        # Coins collected (target-aware)
        score += self._metric_score(coins, self.config.targets.coins, w_coins)

        # Jumps made (target-aware)
        score += self._metric_score(jumps, self.config.targets.jumps, w_jumps)

        # Graduated change penalty
        change_ratio = result.metrics.get("change_ratio", 0.0)
        change_threshold = 0.6
        if change_ratio > change_threshold:
            excess = change_ratio - change_threshold
            score -= self.config.scoring.w_change_penalty * excess

        return score

    def is_better(self, new_result: EvalResult, old_result: EvalResult) -> bool:
        """Check if new result is better than old.

        Args:
            new_result: New evaluation result.
            old_result: Previous evaluation result.

        Returns:
            True if new is strictly better.
        """
        return self.score(new_result) > self.score(old_result)

    def format_score(self, result: EvalResult) -> str:
        """Format score for display.

        Args:
            result: EvalResult to format.

        Returns:
            Human-readable score string.
        """
        score = self.score(result)

        if result.problem_type == 'zelda':
            players = result.metrics.get("players", 0)
            keys = result.metrics.get("keys", 0)
            doors = result.metrics.get("doors", 0)
            enemies = result.metrics.get("enemies", 0)
            solution_length = result.metrics.get("solution_length", 0)
            regions = result.metrics.get("regions", 0)
            playable = result.metrics.get("player_key", 0) > 0 and result.metrics.get("key_door", 0) > 0

            # Show status indicators for critical tiles
            p_ok = "✓" if players == 1 else "✗"
            k_ok = "✓" if keys == 1 else "✗"
            d_ok = "✓" if doors == 1 else "✗"

            target_enemies = 3
            if self.config.targets.enemies and self.config.targets.enemies.value:
                target_enemies = self.config.targets.enemies.value
            e_ok = "✓" if enemies == target_enemies else "~"

            return (
                f"Score: {score:.2f} "
                f"(P={players}{p_ok}, K={keys}{k_ok}, D={doors}{d_ok}, E={enemies}{e_ok}, "
                f"sol_len={solution_length}, regions={regions}, "
                f"playable={'Y' if playable else 'N'})"
            )
        elif result.problem_type == 'sokoban':
            players = result.metrics.get("players", 0)
            crates = result.metrics.get("crates", 0)
            targets = result.metrics.get("targets", 0)
            heuristic = result.metrics.get("heuristic", -1)
            solution_length = result.metrics.get("solution_length", 0)

            # Show status indicators
            p_ok = "✓" if players == 1 else "✗"
            bal_ok = "✓" if crates == targets else "✗"
            sol_ok = "✓" if solution_length > 0 else "✗"

            return (
                f"Score: {score:.2f} "
                f"(P={players}{p_ok}, crates={crates}, targets={targets}, "
                f"balanced={bal_ok}, solvable={sol_ok}, sol_len={solution_length})"
            )
        elif result.problem_type == 'loderunner':
            player = result.metrics.get("player", 0)
            gold = result.metrics.get("gold", 0)
            enemy = result.metrics.get("enemy", 0)
            collected_gold = result.metrics.get("collected_gold", 0)

            p_ok = "✓" if player == 1 else "✗"
            collect_ok = "✓" if gold > 0 and collected_gold == gold else "~"

            return (
                f"Score: {score:.2f} "
                f"(P={player}{p_ok}, gold={gold}, enemy={enemy}, "
                f"collected={collected_gold}/{gold}{collect_ok})"
            )
        elif result.problem_type == 'smb':
            complete = result.metrics.get("complete", 0.0)
            enemies = result.metrics.get("enemies", 0)
            coins = result.metrics.get("coins", 0)
            jumps = result.metrics.get("jumps", 0)
            tube_issues = result.metrics.get("tube_issues", 0)

            c_ok = "✓" if complete >= 1.0 else "✗"
            t_ok = "✓" if tube_issues == 0 else "✗"

            return (
                f"Score: {score:.2f} "
                f"(complete={complete*100:.0f}%{c_ok}, tubes={tube_issues}{t_ok}, "
                f"enemies_killed={enemies}, coins_collected={coins}, jumps={jumps})"
            )
        elif result.problem_type == 'binarydoor':
            door_path = result.metrics.get("door_path", 0)
            regions = result.metrics.get("num_connected_regions", 0)
            connected = "Y" if door_path > 0 else "N"

            return (
                f"Score: {score:.2f} "
                f"(door_path={door_path}, regions={regions}, "
                f"connected={connected})"
            )
        else:
            path = result.metrics.get("path", 0)
            regions = result.metrics.get("num_connected_regions", 0)

            return (
                f"Score: {score:.2f} "
                f"(path={path}, regions={regions}, "
                f"w_path={self.w_path}, w_regions={self.w_regions})"
            )
