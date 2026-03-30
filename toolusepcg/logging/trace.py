"""Trace logger for recording optimization runs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if TYPE_CHECKING:
    from ..level import Level
    from ..evaluator import EvalResult
    from ..config import Config


def create_edit_overlay(base_img: Image.Image, edits: list[dict], scale: int = 16,
                        color: tuple = (255, 0, 0, 180),
                        border_offset: tuple[int, int] = (1, 1)) -> Image.Image:
    """Create an image with edit locations highlighted.

    Args:
        base_img: Base level image.
        edits: List of edit dicts with 'y', 'x' keys.
        scale: Tile scale factor.
        color: RGBA color for edit highlight.
        border_offset: Tuple of (y_offset, x_offset) for border padding.
            Binary/Zelda/Sokoban: (1, 1) - 1-tile border on all sides
            LodeRunner: (0, 0) - only bottom padding, no top/left
            SMB: (0, 3) - 3-column padding on left, no row padding

    Returns:
        Image with edit overlay.
    """
    # Create a copy to avoid modifying original
    result = base_img.copy().convert('RGBA')
    overlay = Image.new('RGBA', result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    y_offset, x_offset = border_offset
    for edit in edits:
        y, x = edit['y'], edit['x']
        # Account for border padding in PCG benchmark render
        px = (x + x_offset) * scale
        py = (y + y_offset) * scale
        draw.rectangle([px, py, px + scale - 1, py + scale - 1], fill=color)

    result = Image.alpha_composite(result, overlay)
    return result


def create_heatmap(height: int, width: int, edit_counts: np.ndarray, scale: int = 16) -> Image.Image:
    """Create a heatmap image showing edit frequency.

    Args:
        height: Level height.
        width: Level width.
        edit_counts: 2D array of edit counts per tile.
        scale: Tile scale factor.

    Returns:
        Heatmap image.
    """
    # Normalize counts to 0-1
    max_count = edit_counts.max()
    if max_count == 0:
        normalized = np.zeros_like(edit_counts, dtype=float)
    else:
        normalized = edit_counts / max_count

    # Create image with border
    padded_h = height + 2
    padded_w = width + 2
    img = Image.new('RGBA', (padded_w * scale, padded_h * scale), (40, 40, 40, 255))
    draw = ImageDraw.Draw(img)

    # Draw heatmap tiles
    for y in range(height):
        for x in range(width):
            intensity = normalized[y, x]
            count = int(edit_counts[y, x])

            # Color gradient: blue (cold) -> red (hot)
            if intensity == 0:
                color = (60, 60, 60, 255)  # Dark gray for no edits
            else:
                # Interpolate from blue to yellow to red
                if intensity < 0.5:
                    # Blue to yellow
                    t = intensity * 2
                    r = int(255 * t)
                    g = int(255 * t)
                    b = int(255 * (1 - t))
                else:
                    # Yellow to red
                    t = (intensity - 0.5) * 2
                    r = 255
                    g = int(255 * (1 - t))
                    b = 0
                color = (r, g, b, 255)

            px = (x + 1) * scale
            py = (y + 1) * scale
            draw.rectangle([px, py, px + scale - 1, py + scale - 1], fill=color)

            # Draw count text for non-zero cells if scale is large enough
            if count > 0 and scale >= 16:
                text = str(count)
                # Center text in tile
                text_x = px + scale // 2
                text_y = py + scale // 2
                # Use contrasting color
                text_color = (0, 0, 0, 255) if intensity > 0.3 else (255, 255, 255, 255)
                draw.text((text_x, text_y), text, fill=text_color, anchor="mm")

    return img


class TraceLogger:
    """Logger for recording optimization traces."""

    def __init__(self, config: "Config", run_dir: str | Path | None = None, exp_name: str | None = None,
                 render_stats: bool = True, render_sim: bool = False,
                 runtime_args: dict[str, Any] | None = None):
        """Initialize trace logger.

        Args:
            config: Configuration object.
            run_dir: Directory for this run. If None, creates timestamped directory.
            exp_name: Optional experiment name to include in directory name.
            render_stats: Whether to overlay metrics/targets on rendered images.
            render_sim: Whether to render SMB simulation GIF showing AI agent playthrough.
            runtime_args: Optional dict of CLI/runtime arguments to save alongside config.
        """
        self.config = config
        self._runtime_args = runtime_args

        # Create run directory
        if run_dir is None:
            from ..config import sanitize_model_name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = sanitize_model_name(config.llm.model)
            problem_type = config.env.problem_type
            if exp_name:
                run_dir = Path("runs") / f"{timestamp}_{model_name}_{problem_type}_{exp_name}"
            else:
                run_dir = Path("runs") / f"{timestamp}_{model_name}_{problem_type}"
        else:
            run_dir = Path(run_dir)

        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.config_path = self.run_dir / "config.yaml"
        self.initial_level_path = self.run_dir / "initial_level.txt"
        self.final_level_path = self.run_dir / "final_level.txt"
        self.trace_path = self.run_dir / "trace.jsonl"
        self.summary_path = self.run_dir / "summary.json"

        # PCG env for rendering (lazy init)
        self._env = None

        # Step counter for images
        self._step_num = 0

        # Edit heatmap tracking
        self._edit_counts: np.ndarray | None = None
        self._level_height = config.env.height
        self._level_width = config.env.width

        # Problem type for level parsing
        self._problem_type = config.env.problem_type

        # Stats overlay on rendered images
        self._render_stats = render_stats

        # SMB simulation GIF rendering
        self._render_sim = render_sim

        # Write resolved config
        self._write_config()

    def _write_config(self) -> None:
        """Write resolved configuration to file.

        Includes both the resolved Config dataclass and any runtime arguments
        (max_steps, debug, verbose, etc.) under a 'runtime' key so each run
        has a complete record of what was executed.
        """
        from ..config import config_to_dict

        config_dict = config_to_dict(self.config)
        if self._runtime_args:
            config_dict["runtime"] = self._runtime_args
        with open(self.config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def set_env(self, env) -> None:
        """Set PCG environment for rendering.

        Args:
            env: PCG benchmark environment.
        """
        self._env = env

    def _render_level(self, level_data, info=None):
        """Render level, handling different render signatures.

        Some problem types (Binary, Zelda) accept info parameter,
        others (Sokoban, LodeRunner, SMB) only accept content.

        Note: We call _problem.render() directly for problem types that don't
        support info because PCGEnv.render() has a bug where it wraps None
        in a list making it truthy, causing it to pass info anyway.

        Args:
            level_data: Level data array.
            info: Optional info dict.

        Returns:
            PIL Image or None if rendering fails.
        """
        if self._env is None:
            return None

        try:
            # Problem types that don't support info parameter in render
            # Call _problem.render() directly to avoid PCGEnv wrapper bug
            if self._problem_type in ('sokoban', 'loderunner', 'smb'):
                return self._env._problem.render(level_data)
            else:
                # Binary, Zelda support info parameter
                return self._env.render(level_data, info)
        except Exception:
            return None

    def _get_border_offset(self) -> tuple[int, int]:
        """Get border offset for edit overlay based on problem type.

        Different problem types have different border padding in their render:
        - Binary, Zelda, Sokoban: 1-tile border padding on all sides
        - LodeRunner: Only 1 row at bottom (no top/left/right padding)
        - SMB: 3 columns on left/right, no row padding

        Returns:
            Tuple of (y_offset, x_offset) in tiles.
        """
        if self._problem_type == 'loderunner':
            return (0, 0)
        elif self._problem_type == 'smb':
            return (0, 3)  # SMB pads 3 columns on left
        return (1, 1)  # binary, zelda, sokoban

    # Metrics to display in stats overlay per problem type.
    # Each entry: (metric_key, config_target_attribute, hardcoded_fallback[, display_label])
    # When display_label (4th element) is present, it's used in the overlay instead of metric_key.
    _STATS_METRIC_CONFIG: dict[str, list[tuple]] = {
        'binary': [
            ('path', 'path', None),
            ('num_connected_regions', 'num_connected_regions', None),
        ],
        'binarydoor': [
            ('door_path', 'door_path', None),
            ('num_connected_regions', 'num_connected_regions', None),
        ],
        'zelda': [
            ('player_key', 'player_key', None),
            ('key_door', 'key_door', None),
            ('solution_length', 'solution_length', None),
            ('enemies', 'enemies', None),
            ('players', None, 1),
            ('keys', None, 1),
            ('doors', None, 1),
            ('regions', None, 1),
        ],
        'sokoban': [
            ('crates', 'crates', None),
            ('solution_length', 'solution_length', None),
            ('players', None, 1),
        ],
        'loderunner': [
            ('ladder', 'ladder', None),
            ('rope', 'rope', None),
            ('gold', 'gold', None),
            ('enemy', 'enemies', None),
            ('player', None, 1),
        ],
        'smb': [
            ('enemies', 'enemies', None, 'enemies_killed'),
            ('jumps', 'jumps', None, 'jumps'),
            ('coins', 'coins', None, 'coins_collected'),
            ('complete', 'complete', 1.0),
        ],
    }

    def _get_metric_targets(self) -> list[tuple[str, str, str, int | float | None]]:
        """Get metrics with target info for stats overlay.

        Returns:
            List of (metric_key, display_label, mode, target_value) tuples.
            Controllable (mode=target with value) metrics are listed first.
        """
        display_config = self._STATS_METRIC_CONFIG.get(self._problem_type, [])
        controllable: list[tuple[str, str, str, int | float | None]] = []
        other: list[tuple[str, str, str, int | float | None]] = []

        for entry in display_config:
            metric_key, config_attr, hardcoded = entry[0], entry[1], entry[2]
            display_label = entry[3] if len(entry) > 3 else metric_key
            cfg = getattr(self.config.targets, config_attr, None) if config_attr else None

            if cfg is not None:
                val = cfg.value if cfg.value is not None else cfg.min_threshold
                item = (metric_key, display_label, cfg.mode, val)
                if cfg.mode == 'target' and val is not None:
                    controllable.append(item)
                else:
                    other.append(item)
            elif hardcoded is not None:
                other.append((metric_key, display_label, 'target', hardcoded))

        return controllable + other

    def _build_stats_lines(self, metrics: dict) -> list[tuple[str, tuple[int, int, int, int]]]:
        """Build (text, RGBA_color) pairs for stats overlay.

        Args:
            metrics: Current metric values dict.

        Returns:
            List of (text, color) tuples for rendering.
        """
        green = (0, 220, 0, 255)
        orange = (255, 170, 40, 255)
        blue = (120, 180, 255, 255)
        gray = (200, 200, 200, 255)

        lines: list[tuple[str, tuple[int, int, int, int]]] = []
        target_info = self._get_metric_targets()

        for metric_key, display_label, mode, target_val in target_info:
            current = metrics.get(metric_key)
            if current is None:
                continue

            # Format current value
            if isinstance(current, float) and current != int(current):
                cur_str = f"{current:.2f}"
            else:
                cur_str = str(int(current) if isinstance(current, float) else current)

            if mode == 'target' and target_val is not None:
                delta = current - target_val

                # Format target value
                if isinstance(target_val, float) and target_val != int(target_val):
                    tgt_str = f"{target_val:.2f}"
                else:
                    tgt_str = str(int(target_val) if isinstance(target_val, float) else target_val)

                if current == target_val:
                    text = f"{display_label}: {cur_str}  [target: {tgt_str}] \u2713"
                    color = green
                elif delta > 0:
                    abs_d = abs(delta)
                    d_str = f"{abs_d:.1f}" if isinstance(abs_d, float) and abs_d != int(abs_d) else str(int(abs_d))
                    text = f"{display_label}: {cur_str}  [target: {tgt_str}] need -{d_str}"
                    color = orange
                else:
                    abs_d = abs(delta)
                    d_str = f"{abs_d:.1f}" if isinstance(abs_d, float) and abs_d != int(abs_d) else str(int(abs_d))
                    text = f"{display_label}: {cur_str}  [target: {tgt_str}] need +{d_str}"
                    color = orange
            elif mode == 'maximize':
                text = f"{display_label}: {cur_str}  (maximize)"
                color = blue
            elif mode == 'minimize':
                text = f"{display_label}: {cur_str}  (minimize)"
                color = blue
            else:
                text = f"{display_label}: {cur_str}"
                color = gray

            lines.append((text, color))

        return lines

    def _get_stats_font(self, size: int):
        """Get a font with Unicode support for stats rendering."""
        for path in [
            # macOS
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/SFMono-Regular.otf",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    def _create_stats_panel(self, metrics: dict, img_width: int) -> Image.Image | None:
        """Create a stats panel image to place below the level image.

        Args:
            metrics: Current metric values dict.
            img_width: Width of the level image (panel will match or exceed).

        Returns:
            Stats panel image, or None if no stats to show.
        """
        lines = self._build_stats_lines(metrics)
        if not lines:
            return None

        # Adaptive font size based on image width
        font_size = max(11, min(16, img_width // 20))
        font = self._get_stats_font(font_size)

        # Calculate panel dimensions
        padding = 6
        line_height = font_size + 4
        panel_height = padding * 2 + len(lines) * line_height

        # Measure max text width to ensure nothing is clipped
        max_text_width = 0
        for text, _ in lines:
            bbox = font.getbbox(text)
            max_text_width = max(max_text_width, bbox[2] - bbox[0])
        panel_width = max(img_width, max_text_width + padding * 2)

        # Create panel with dark background
        panel = Image.new('RGBA', (panel_width, panel_height), (30, 30, 30, 230))
        draw = ImageDraw.Draw(panel)

        y = padding
        for text, color in lines:
            draw.text((padding, y), text, fill=color, font=font)
            y += line_height

        return panel

    def _compose_with_stats(self, level_img: Image.Image, metrics: dict) -> Image.Image:
        """Compose level image with stats panel below.

        If render_stats is disabled or no metrics have targets, returns
        the original image unchanged.

        Args:
            level_img: Rendered level image.
            metrics: Current metric values dict.

        Returns:
            Composite image with stats panel appended below.
        """
        if not self._render_stats:
            return level_img

        stats_panel = self._create_stats_panel(metrics, level_img.width)
        if stats_panel is None:
            return level_img

        # Convert level image to RGBA if needed
        if level_img.mode != 'RGBA':
            level_img = level_img.convert('RGBA')

        # Composite: level on top, stats panel below
        # Panel may be wider than level image if text is long
        composite_width = max(level_img.width, stats_panel.width)
        total_height = level_img.height + stats_panel.height
        composite = Image.new('RGBA', (composite_width, total_height), (30, 30, 30, 255))
        composite.paste(level_img, (0, 0))
        composite.paste(stats_panel, (0, level_img.height))

        return composite

    def log_initial_level(self, level: "Level", result: "EvalResult") -> None:
        """Log initial level state.

        Args:
            level: Initial level.
            result: Initial evaluation result.
        """
        # Update dimensions from actual level
        self._level_height = level.height
        self._level_width = level.width

        # Write level text
        with open(self.initial_level_path, "w") as f:
            f.write(level.to_string())

        # Render image
        img = self._render_level(level.data, result.info)
        if img is not None:
            img = self._compose_with_stats(img, result.metrics)
            img.save(self.run_dir / "initial_level.png")

        # Log to trace
        self._log_trace({
            "event": "initial_level",
            "level": level.to_string(),
            "metrics": result.metrics,
            "valid": result.valid,
            "timestamp": datetime.now().isoformat(),
        })

    def log_final_level(self, level: "Level", result: "EvalResult") -> None:
        """Log final level state.

        Args:
            level: Final level.
            result: Final evaluation result.
        """
        # Write level text
        with open(self.final_level_path, "w") as f:
            f.write(level.to_string())

        # Render image
        img = self._render_level(level.data, result.info)
        if img is not None:
            img = self._compose_with_stats(img, result.metrics)
            img.save(self.run_dir / "final_level.png")

        # Save cumulative edit heatmap
        self.save_heatmap()

        # Log to trace
        self._log_trace({
            "event": "final_level",
            "level": level.to_string(),
            "metrics": result.metrics,
            "valid": result.valid,
            "timestamp": datetime.now().isoformat(),
        })

        # Render animated GIFs of the full run
        self.render_gif()
        self.render_per_tool_edit_gif()

        # Render simulation GIF and trajectory image if enabled (SMB/Zelda)
        if self._render_sim and result.info:
            if self._problem_type == 'smb':
                self.render_simulation_gif(level.data, result.info, metrics=result.metrics)
                self._render_trajectory_image(level.data, result.info)
            elif self._problem_type == 'zelda':
                self.render_zelda_simulation_gif(level.data, result.info, metrics=result.metrics)

    def log_step(self, record: Any) -> None:
        """Log a single optimization step.

        Args:
            record: StepRecord from optimizer.
        """
        self._step_num = record.step_num

        # Build trace entry
        entry = {
            "event": "step",
            "step_num": record.step_num,
            "message_type": record.message_type,
            "tool_results": record.tool_results,
            "metrics_before": record.metrics_before,
            "metrics_after": record.metrics_after,
            "score_before": record.score_before,
            "score_after": record.score_after,
            "accepted": record.accepted,
            "accept_reason": record.accept_reason,
            "num_tiles_changed": record.num_tiles_changed,
            "errors": record.errors,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "timestamp": datetime.now().isoformat(),
        }

        # Include parsed message (but not raw response to save space)
        if record.parsed_message:
            entry["parsed_message"] = record.parsed_message

        self._log_trace(entry)

        # Extract edits from tool results
        step_edits = self._extract_edits(record.tool_results)

        # Update cumulative heatmap if accepted
        if record.accepted and step_edits:
            self._update_edit_counts(step_edits)

        # Render step image if accepted and we have level
        if record.accepted and record.level_after and self._env is not None:
            try:
                from ..level import Level

                level = Level.from_string(record.level_after, problem_type=self._problem_type)
                info = self._env.info(level.data)
                base_img = self._render_level(level.data, info)

                if base_img is not None:
                    # Save base level image with stats overlay
                    step_metrics = record.metrics_after or {}
                    img_with_stats = self._compose_with_stats(base_img, step_metrics)
                    img_with_stats.save(self.run_dir / f"step_{record.step_num}.png")

                    # Save image with edit overlay and stats
                    if step_edits:
                        border_offset = self._get_border_offset()
                        edit_img = create_edit_overlay(base_img, step_edits, border_offset=border_offset)
                        edit_with_stats = self._compose_with_stats(edit_img, step_metrics)
                        edit_with_stats.save(self.run_dir / f"step_{record.step_num}_edits.png")

            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(
                    "Failed to render step %d image: %s", record.step_num, e
                )

    def _extract_edits(self, tool_results: list[dict]) -> list[dict]:
        """Extract edit locations from tool results.

        Any tool result with a successful diff is treated as an edit,
        regardless of which tool produced it.

        Args:
            tool_results: List of tool result dicts.

        Returns:
            List of edit dicts with 'y', 'x', 'old', 'new' keys.
        """
        edits = []
        for tr in tool_results:
            result = tr.get("result", {})
            if result.get("success"):
                diff = result.get("data", {}).get("diff", [])
                edits.extend(diff)
        return edits

    def _update_edit_counts(self, edits: list[dict]) -> None:
        """Update cumulative edit counts.

        Args:
            edits: List of edit dicts with 'y', 'x' keys.
        """
        # Initialize if needed
        if self._edit_counts is None:
            self._edit_counts = np.zeros((self._level_height, self._level_width), dtype=np.int32)

        for edit in edits:
            y, x = edit['y'], edit['x']
            if 0 <= y < self._level_height and 0 <= x < self._level_width:
                self._edit_counts[y, x] += 1

    def save_heatmap(self) -> None:
        """Save the cumulative edit heatmap."""
        if self._edit_counts is None or self._edit_counts.max() == 0:
            return

        heatmap_img = create_heatmap(self._level_height, self._level_width, self._edit_counts)
        heatmap_img.save(self.run_dir / "edit_heatmap.png")

        # Also save the raw counts as JSON
        heatmap_data = {
            "edit_counts": self._edit_counts.tolist(),
            "total_edits": int(self._edit_counts.sum()),
            "max_edits_per_tile": int(self._edit_counts.max()),
            "tiles_edited": int((self._edit_counts > 0).sum()),
        }
        with open(self.run_dir / "edit_heatmap.json", "w") as f:
            json.dump(heatmap_data, f, indent=2)

    def log_skill_proposal(self, skill_spec: dict) -> None:
        """Log a skill proposal.

        Args:
            skill_spec: Proposed skill specification.
        """
        self._log_trace({
            "event": "skill_proposal",
            "skill_spec": skill_spec,
            "timestamp": datetime.now().isoformat(),
        })

    def log_sampled_targets(self, targets: dict, seed: int | None = None) -> None:
        """Log sampled control targets for controllability testing.

        Args:
            targets: Dictionary of sampled control parameter targets.
            seed: Seed used for sampling (if any).
        """
        self._log_trace({
            "event": "sampled_targets",
            "targets": targets,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        })

    def log_controllability_result(self, result) -> None:
        """Log controllability result for a single trial.

        Args:
            result: ControllabilityResult object.
        """
        self._log_trace({
            "event": "controllability_result",
            "trial_id": result.trial_id,
            "sampled_control": result.sampled_control,
            "achieved_metrics": {k: v for k, v in result.achieved_metrics.items()
                                 if not isinstance(v, (list, dict))},
            "controllability_score": result.controllability_score,
            "exact_match": result.exact_match,
            "final_score": result.final_score,
            "total_steps": result.total_steps,
            "timestamp": datetime.now().isoformat(),
        })

    def log_summary(self, summary: dict) -> None:
        """Log final summary.

        Args:
            summary: Summary dictionary.
        """
        summary["run_dir"] = str(self.run_dir)
        summary["timestamp"] = datetime.now().isoformat()

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

        self._log_trace({
            "event": "summary",
            **summary,
        })

    def _log_trace(self, entry: dict) -> None:
        """Append entry to trace file.

        Args:
            entry: Dictionary to write as JSON line.
        """
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")

    def render_gif(self, frame_duration: int = 500, hold_duration: int = 1500) -> Path | None:
        """Render an animated GIF from all step images.

        Sequences initial level, per-step map states and edit overlays,
        and the final level into an animated GIF.

        Args:
            frame_duration: Duration per normal frame in ms.
            hold_duration: Duration for initial and final frames in ms.

        Returns:
            Path to the saved GIF, or None if not enough images.
        """
        frames: list[Image.Image] = []
        durations: list[int] = []

        def append_frame(img_path: Path, duration: int) -> None:
            if img_path.exists():
                img = Image.open(img_path)
                # GIF doesn't support RGBA; composite onto dark background
                if img.mode == 'RGBA':
                    bg = Image.new('RGB', img.size, (30, 30, 30))
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert('RGB')
                frames.append(img)
                durations.append(duration)

        # Initial level
        append_frame(self.run_dir / "initial_level.png", hold_duration)

        # Collect step images sorted by step number
        step_files = sorted(
            self.run_dir.glob("step_*.png"),
            key=lambda p: int(re.search(r'step_(\d+)', p.stem).group(1))  # type: ignore[union-attr]
        )

        # Group by step number: base image first, then edits overlay
        seen_steps: dict[int, list[Path]] = {}
        for f in step_files:
            m = re.search(r'step_(\d+)', f.stem)
            if m:
                step_num = int(m.group(1))
                seen_steps.setdefault(step_num, []).append(f)

        for step_num in sorted(seen_steps):
            paths = seen_steps[step_num]
            # Sort so _edits overlay comes before base image
            paths.sort(key=lambda p: '_edits' not in p.stem)
            for p in paths:
                append_frame(p, frame_duration)

        # Final level
        append_frame(self.run_dir / "final_level.png", hold_duration)

        if len(frames) < 2:
            return None

        # Normalize all frames to the same size to prevent ghosting artifacts.
        # Frames can differ in width when the stats panel text length varies.
        max_w = max(f.width for f in frames)
        max_h = max(f.height for f in frames)
        normalized: list[Image.Image] = []
        for f in frames:
            if f.size != (max_w, max_h):
                canvas = Image.new('RGB', (max_w, max_h), (30, 30, 30))
                canvas.paste(f, (0, 0))
                normalized.append(canvas)
            else:
                normalized.append(f)

        gif_path = self.run_dir / "run_animation.gif"
        normalized[0].save(
            gif_path,
            save_all=True,
            append_images=normalized[1:],
            duration=durations,
            loop=0,
        )
        return gif_path

    def render_per_tool_edit_gif(self, frame_duration: int = 400, hold_duration: int = 1500) -> Path | None:
        """Render per-tool-call animation GIF from trace.jsonl.

        Unlike render_gif() which shows one frame per step, this renders each
        individual tool call so the level is built up tool-by-tool.

        Args:
            frame_duration: Duration per tool-call frame in ms.
            hold_duration: Duration for initial and final frames in ms.

        Returns:
            Path to the saved GIF, or None if not enough data.
        """
        from ..level import Level

        if self._env is None:
            return None

        if not self.trace_path.exists():
            return None

        # Load trace entries
        entries = []
        with open(self.trace_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        initial_entry = next((e for e in entries if e["event"] == "initial_level"), None)
        if initial_entry is None:
            return None

        current_level = Level.from_string(initial_entry["level"], problem_type=self._problem_type)
        current_metrics = initial_entry.get("metrics", {})
        border_offset = self._get_border_offset()

        frames: list[Image.Image] = []
        durations: list[int] = []

        def add_frame(img: Image.Image, duration: int):
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (30, 30, 30))
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = img.convert('RGB')
            frames.append(img)
            durations.append(duration)

        def render_with_stats(level_data, metrics):
            base_img = self._render_level(level_data)
            if base_img is None:
                return None
            return self._compose_with_stats(base_img, metrics)

        # Initial level frame
        init_img = render_with_stats(current_level.data, current_metrics)
        if init_img is None:
            return None
        add_frame(init_img, hold_duration)

        # Process accepted steps only
        for entry in entries:
            if entry.get("event") != "step" or not entry.get("accepted", False):
                continue

            tool_results = entry.get("tool_results", [])
            if not tool_results:
                continue

            metrics_during = entry.get("metrics_before", current_metrics)
            metrics_after = entry.get("metrics_after", metrics_during)
            name_tiles = current_level.name_tiles

            for tr in tool_results:
                result = tr.get("result", {})
                if not result.get("success"):
                    continue
                diff = result.get("data", {}).get("diff", [])
                if not diff:
                    continue

                # Apply diff
                for d in diff:
                    tile_val = name_tiles.get(d["new"])
                    if tile_val is not None:
                        current_level.data[d["y"], d["x"]] = tile_val

                base_img = self._render_level(current_level.data)
                if base_img is None:
                    continue

                edit_img = create_edit_overlay(base_img, diff, border_offset=border_offset)
                img_with_stats = self._compose_with_stats(edit_img, metrics_during)
                add_frame(img_with_stats, frame_duration)

            # Clean step result frame
            result_img = render_with_stats(current_level.data, metrics_after)
            if result_img is not None:
                add_frame(result_img, frame_duration + 200)

            current_metrics = metrics_after

        # Final level frame
        final_img = render_with_stats(current_level.data, current_metrics)
        if final_img is not None:
            add_frame(final_img, hold_duration)

        if len(frames) < 2:
            return None

        # Normalize frame sizes
        max_w = max(f.width for f in frames)
        max_h = max(f.height for f in frames)
        normalized: list[Image.Image] = []
        for f in frames:
            if f.size != (max_w, max_h):
                canvas = Image.new('RGB', (max_w, max_h), (30, 30, 30))
                canvas.paste(f, (0, 0))
                normalized.append(canvas)
            else:
                normalized.append(f)

        gif_path = self.run_dir / "per_tool_edit_process.gif"
        normalized[0].save(
            gif_path,
            save_all=True,
            append_images=normalized[1:],
            duration=durations,
            loop=0,
        )
        return gif_path

    # SpriteType enum values from the SMB engine mapped to sprite image filenames
    _ENEMY_SPRITE_FILES: dict[int, str] = {
        2: "gomba.png",       # GOOMBA
        3: "gomba.png",       # GOOMBA_WINGED (use same sprite)
        4: "redkoopa.png",    # RED_KOOPA
        5: "redkoopa.png",    # RED_KOOPA_WINGED
        6: "greenkoopa.png",  # GREEN_KOOPA
        7: "greenkoopa.png",  # GREEN_KOOPA_WINGED
        8: "spiky.png",       # SPIKY
        9: "spiky.png",       # SPIKY_WINGED
    }

    # SMB engine camera viewport size in pixels
    _CAMERA_W = 256
    _CAMERA_H = 256
    # Tile index to SpriteType value mapping for smbtile content_space
    # symbols = ['-','X','#','S','Q','t','o','g','k','y'] → indices 0-9
    _TILE_TO_SPRITE_TYPE: dict[int, int] = {
        7: 2,   # g → GOOMBA
        8: 6,   # k → GREEN_KOOPA
        9: 8,   # y → SPIKY
    }

    def _render_trajectory_image(self, level_data, info: dict) -> Path | None:
        """Render a static trajectory PNG for SMB using render_trajectory module."""
        try:
            from render_trajectory import (
                _load_sprites, _make_transparent, _engine_to_image,
                _initial_enemy_spawns, _track_enemies, _mark_kills,
                _draw_dashed_path, _draw_kill_marker, _subsample,
                ENEMY_PALETTE, MARIO_COLOR, SPRITE_CENTER_OFFSET_Y,
                TILE_TO_SPRITE_TYPE, SKY_BLUE,
            )
        except Exception:
            return None

        locations = info.get("locations", [])
        if not locations or self._env is None:
            return None

        enemy_frames = info.get("enemy_frames", [])
        game_events = info.get("game_events", [])

        # Render clean background (enemies removed)
        level_arr = np.array(level_data)
        bg_level = level_arr.copy()
        bg_level[np.isin(bg_level, list(TILE_TO_SPRITE_TYPE.keys()))] = 0
        try:
            background = self._env._problem.render(bg_level)
        except Exception:
            return None
        if background is None:
            return None

        canvas = background.convert("RGBA")
        mario_sprite, enemy_sprites = _load_sprites()

        tracks = _track_enemies(enemy_frames)
        _mark_kills(tracks, game_events)
        initial_enemies = _initial_enemy_spawns(level_arr)
        draw = ImageDraw.Draw(canvas, "RGBA")

        # Semi-transparent enemy spawns
        for sprite_type, epx, epy in initial_enemies:
            ix, iy = _engine_to_image(epx, epy)
            esprite = enemy_sprites.get(sprite_type)
            if esprite:
                trans = _make_transparent(esprite, alpha=90)
                canvas.paste(trans, (int(ix) - trans.width // 2, int(iy) - trans.height), trans)

        # Semi-transparent Mario spawn
        mx0, my0 = _engine_to_image(float(locations[0][0]), float(locations[0][1]))
        trans_mario = _make_transparent(mario_sprite, alpha=90)
        canvas.paste(trans_mario, (int(mx0) - trans_mario.width // 2, int(my0) - trans_mario.height), trans_mario)

        draw = ImageDraw.Draw(canvas, "RGBA")
        subsample = 3

        # Enemy trajectories
        sorted_tracks = sorted(tracks, key=lambda t: t["positions"][0][0])
        for i, track in enumerate(sorted_tracks):
            color = ENEMY_PALETTE[i % len(ENEMY_PALETTE)]
            img_pos = [_engine_to_image(x, y + SPRITE_CENTER_OFFSET_Y) for x, y in track["positions"]]
            sampled = _subsample(img_pos, step=subsample)
            if len(sampled) >= 2:
                _draw_dashed_path(draw, sampled, color, dash=6, gap=4, width=2)

        # Mario trajectory
        mario_img = [_engine_to_image(float(l[0]), float(l[1]) + SPRITE_CENTER_OFFSET_Y) for l in locations]
        mario_sampled = _subsample(mario_img, step=subsample)
        if len(mario_sampled) >= 2:
            _draw_dashed_path(draw, mario_sampled, MARIO_COLOR, dash=8, gap=5, width=2)

        # Kill markers
        for track in tracks:
            if track["killed"] and track["kill_pos"]:
                kx, ky = _engine_to_image(*track["kill_pos"])
                _draw_kill_marker(draw, kx, ky + SPRITE_CENTER_OFFSET_Y)

        # Scale 2x
        canvas = canvas.resize((canvas.width * 2, canvas.height * 2), Image.NEAREST)

        out = Image.new("RGB", canvas.size, SKY_BLUE)
        out.paste(canvas, mask=canvas.split()[3])

        out_path = self.run_dir / "trajectory.png"
        out.save(out_path)
        return out_path

    def render_simulation_gif(self, level_data, info: dict, max_frames: int = 100,
                              frame_duration: int = 100, hold_duration: int = 1000,
                              metrics: dict | None = None) -> Path | None:
        """Render an animated GIF of the SMB AI agent playing through the level.

        Shows the camera viewport highlighted with a dim overlay on areas outside
        it. Enemies within the camera move at their simulation positions; enemies
        outside the camera are drawn frozen at their spawn tiles. When render_stats
        is enabled and metrics are provided, a stats panel is composited below
        each frame.

        Args:
            level_data: Level data array.
            info: Info dict from evaluator (must contain 'locations' and 'enemy_frames').
            max_frames: Maximum output frames (subsampled if exceeded).
            frame_duration: Duration per normal frame in ms.
            hold_duration: Duration for first and last frames in ms.

        Returns:
            Path to the saved GIF, or None if not enough data.
        """
        locations = info.get("locations", [])
        if not locations or self._env is None:
            return None

        enemy_frames = info.get("enemy_frames", [])

        # Render static background without enemies (clean terrain only)
        import numpy as np
        level_arr = np.array(level_data)
        bg_level = level_arr.copy()
        enemy_tile_set = set(self._TILE_TO_SPRITE_TYPE.keys())
        bg_level[np.isin(bg_level, list(enemy_tile_set))] = 0

        try:
            background = self._env._problem.render(bg_level)
        except Exception:
            return None
        if background is None:
            return None

        bg_rgba = background.convert('RGBA')
        img_w, img_h = bg_rgba.size

        # Level pixel dimensions (padded: 3 tiles each side)
        pad = 3
        level_pixel_w = (level_arr.shape[1] + 2 * pad) * 16
        level_pixel_h = level_arr.shape[0] * 16

        # Load sprites
        try:
            smbtile_dir = (Path(__file__).resolve().parents[2] / "submodules" /
                           "pcg_benchmark" / "pcg_benchmark" / "probs" / "smbtile" / "images")
            mario_sprite = Image.open(smbtile_dir / "mario.png").convert('RGBA')
            loaded_files: dict[str, Image.Image] = {}
            enemy_sprites: dict[int, Image.Image] = {}
            for type_val, filename in self._ENEMY_SPRITE_FILES.items():
                if filename not in loaded_files:
                    loaded_files[filename] = Image.open(smbtile_dir / filename).convert('RGBA')
                enemy_sprites[type_val] = loaded_files[filename]
        except Exception:
            return None

        # Collect initial enemy spawn positions from level_data.
        # Each enemy tile at (row, col) spawns at pixel coords:
        #   px = (col + pad) * 16 + 8,  py = row * 16 + 15
        # (matching spawnSprite in the engine)
        initial_enemies: list[tuple[int, int, int]] = []  # (sprite_type, px, py)
        for row in range(level_arr.shape[0]):
            for col in range(level_arr.shape[1]):
                tile_val = int(level_arr[row][col])
                if tile_val in self._TILE_TO_SPRITE_TYPE:
                    sprite_type = self._TILE_TO_SPRITE_TYPE[tile_val]
                    px = (col + pad) * 16 + 8
                    py = row * 16 + 15
                    initial_enemies.append((sprite_type, px, py))

        # Subsample frames if too many
        num_locations = len(locations)
        if num_locations > max_frames:
            indices = list(set([0, num_locations - 1]) |
                           set(int(round(i * (num_locations - 1) / (max_frames - 1)))
                               for i in range(max_frames)))
            indices.sort()
        else:
            indices = list(range(num_locations))

        sky_blue = (109, 143, 252)
        trail_length = 15
        dim_alpha = 100  # semi-transparent overlay outside camera

        # Track the furthest camera position reached to determine which
        # enemy spawn points have been activated (and thus shouldn't be
        # drawn as frozen anymore — they're either alive in enemy_frames
        # or already dead).
        max_camera_right = 0.0

        frames: list[Image.Image] = []
        durations: list[int] = []

        for frame_idx, loc_idx in enumerate(indices):
            mario_x = float(locations[loc_idx][0])
            mario_y = float(locations[loc_idx][1])

            # Compute camera viewport (same logic as MarioWorld.update)
            cam_x = mario_x - self._CAMERA_W / 2
            cam_x = max(0.0, min(cam_x, level_pixel_w - self._CAMERA_W))
            cam_y = mario_y - self._CAMERA_H / 2
            cam_y = max(0.0, min(cam_y, level_pixel_h - self._CAMERA_H))

            # Update furthest camera reach for spawn tracking.
            # Engine spawns enemies at tile x in [camX/16 - 1, (camX+256)/16 + 1]
            spawn_right_px = cam_x + self._CAMERA_W + 2 * 16
            if spawn_right_px > max_camera_right:
                max_camera_right = spawn_right_px

            frame = bg_rgba.copy()

            # --- Draw frozen (unspawned) enemies at their tile positions ---
            for sprite_type, epx, epy in initial_enemies:
                if epx >= max_camera_right:
                    # Not yet reached by camera — draw frozen
                    esprite = enemy_sprites.get(sprite_type)
                    if esprite is not None:
                        frame.paste(esprite,
                                    (epx - esprite.width // 2, epy - esprite.height),
                                    esprite)

            # --- Draw active (alive) enemies at simulation positions ---
            if loc_idx < len(enemy_frames):
                for enemy_info in enemy_frames[loc_idx]:
                    etype = int(enemy_info[0])
                    ex, ey = int(enemy_info[1]), int(enemy_info[2])
                    esprite = enemy_sprites.get(etype)
                    if esprite is not None:
                        frame.paste(esprite,
                                    (ex - esprite.width // 2, ey - esprite.height),
                                    esprite)

            # --- Apply dim overlay outside camera viewport ---
            overlay = Image.new('RGBA', frame.size, (0, 0, 0, 0))
            draw_ov = ImageDraw.Draw(overlay)
            cx, cy = int(cam_x), int(cam_y)
            cx2 = min(cx + self._CAMERA_W, img_w)
            cy2 = min(cy + self._CAMERA_H, img_h)
            dim = (0, 0, 0, dim_alpha)
            # Top strip
            if cy > 0:
                draw_ov.rectangle([0, 0, img_w, cy - 1], fill=dim)
            # Bottom strip
            if cy2 < img_h:
                draw_ov.rectangle([0, cy2, img_w, img_h], fill=dim)
            # Left strip (between top and bottom)
            if cx > 0:
                draw_ov.rectangle([0, cy, cx - 1, cy2 - 1], fill=dim)
            # Right strip (between top and bottom)
            if cx2 < img_w:
                draw_ov.rectangle([cx2, cy, img_w, cy2 - 1], fill=dim)
            frame = Image.alpha_composite(frame, overlay)

            # --- Draw trajectory trail ---
            draw = ImageDraw.Draw(frame, 'RGBA')
            trail_start = max(0, loc_idx - trail_length)
            for t in range(trail_start, loc_idx):
                tx, ty = int(locations[t][0]), int(locations[t][1])
                alpha = int(80 + 175 * (t - trail_start) / max(1, loc_idx - trail_start))
                r = 3
                draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=(255, 50, 50, alpha))

            # --- Draw Mario ---
            loc_xi, loc_yi = int(mario_x), int(mario_y)
            frame.paste(mario_sprite,
                        (loc_xi - mario_sprite.width // 2, loc_yi - mario_sprite.height),
                        mario_sprite)

            # Apply stats overlay if enabled
            if metrics is not None:
                frame = self._compose_with_stats(frame, metrics)

            # Convert RGBA to RGB for GIF
            rgb_frame = Image.new('RGB', frame.size, sky_blue)
            rgb_frame.paste(frame, mask=frame.split()[3])
            frames.append(rgb_frame)

            if frame_idx == 0 or frame_idx == len(indices) - 1:
                durations.append(hold_duration)
            else:
                durations.append(frame_duration)

        if len(frames) < 2:
            return None

        gif_path = self.run_dir / "simulation.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
        )
        return gif_path

    def render_zelda_simulation_gif(self, level_data, info: dict, max_frames: int = 100,
                                     frame_duration: int = 150, hold_duration: int = 800,
                                     metrics: dict | None = None) -> Path | None:
        """Render an animated GIF of a Zelda level playthrough.

        The player walks the solved path (player->key->door), enemies wander
        randomly bouncing off walls, and collisions show a kill effect before
        the enemy is removed.

        Args:
            level_data: Level data array.
            info: Info dict from evaluator (must contain 'pk_path' and 'kd_path').
            max_frames: Maximum output frames (subsampled if exceeded).
            frame_duration: Duration per normal frame in ms.
            hold_duration: Duration for first and last frames in ms.
            metrics: Optional metrics dict for stats overlay.

        Returns:
            Path to the saved GIF, or None if not enough data.
        """
        import random as random_mod

        pk_path = info.get("pk_path", [])
        kd_path = info.get("kd_path", [])
        if not pk_path or not kd_path or self._env is None:
            return None

        # Combine paths (they overlap at the key position)
        full_path = list(pk_path) + list(kd_path)[1:]
        if len(full_path) < 2:
            return None

        import numpy as np
        level_arr = np.array(level_data)
        scale = 16
        pad = 1  # Zelda render pads 1 tile on all sides

        # Render static background: replace player/enemy/key with empty tiles
        bg_level = level_arr.copy()
        bg_level[bg_level == 2] = 1  # PLAYER -> EMPTY
        bg_level[bg_level == 5] = 1  # ENEMY -> EMPTY
        bg_level[bg_level == 3] = 1  # KEY -> EMPTY

        try:
            background = self._env._problem.render(bg_level)
        except Exception:
            return None
        if background is None:
            return None

        bg_rgba = background.convert('RGBA')

        # Load sprites
        try:
            zelda_dir = (Path(__file__).resolve().parents[2] / "submodules" /
                         "pcg_benchmark" / "pcg_benchmark" / "probs" / "zelda" / "images")
            player_sprite = Image.open(zelda_dir / "player.png").convert('RGBA')
            bat_sprite = Image.open(zelda_dir / "bat.png").convert('RGBA')
            key_sprite = Image.open(zelda_dir / "key.png").convert('RGBA')
        except Exception:
            return None

        # Collect enemy starting positions from level_data (tile == 5)
        enemies: list[dict] = []
        for row in range(level_arr.shape[0]):
            for col in range(level_arr.shape[1]):
                if int(level_arr[row][col]) == 5:
                    enemies.append({
                        'x': col, 'y': row,
                        'prev_x': col, 'prev_y': row,
                        'dx': 0, 'dy': 0,
                        'alive': True,
                    })

        # Find key position from level_data (tile == 3)
        key_pos = None
        for row in range(level_arr.shape[0]):
            for col in range(level_arr.shape[1]):
                if int(level_arr[row][col]) == 3:
                    key_pos = (col, row)
                    break
            if key_pos is not None:
                break

        # Deterministic RNG for enemy movement
        rng = random_mod.Random(42)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # Initialize enemy directions
        for enemy in enemies:
            enemy['dx'], enemy['dy'] = rng.choice(directions)

        # Render every step; scale down frame duration for long paths
        num_steps = len(full_path)
        if num_steps > max_frames:
            frame_duration = max(30, frame_duration * max_frames // num_steps)

        # Kill effects: list of (x, y, ttl)
        kill_effects: list[list] = []
        key_collected = False
        trail: list[tuple[int, int]] = []
        trail_max = 8

        # Helper: check if grid cell is passable for enemies
        rows, cols = level_arr.shape
        def passable(x: int, y: int) -> bool:
            return 0 <= x < cols and 0 <= y < rows and level_arr[y][x] != 0

        frames: list[Image.Image] = []
        durations: list[int] = []

        # Simulate ALL steps; only render at selected indices
        for step_idx in range(num_steps):
            prev_px, prev_py = full_path[step_idx - 1] if step_idx > 0 else full_path[0]
            px, py = full_path[step_idx]  # path coords are (x, y)

            # Move enemies every 2nd step (simulation runs every step)
            if step_idx % 2 == 0:
                for enemy in enemies:
                    if not enemy['alive']:
                        continue
                    enemy['prev_x'], enemy['prev_y'] = enemy['x'], enemy['y']
                    nx = enemy['x'] + enemy['dx']
                    ny = enemy['y'] + enemy['dy']
                    if passable(nx, ny):
                        enemy['x'] = nx
                        enemy['y'] = ny
                    else:
                        enemy['dx'] = -enemy['dx']
                        enemy['dy'] = -enemy['dy']
                        new_dir = rng.choice(directions)
                        enemy['dx'], enemy['dy'] = new_dir

            # Collision check: same cell OR swap (player and enemy crossed paths)
            for enemy in enemies:
                if not enemy['alive']:
                    continue
                same_cell = (enemy['x'] == px and enemy['y'] == py)
                swapped = (enemy['x'] == prev_px and enemy['y'] == prev_py and
                           enemy.get('prev_x') == px and enemy.get('prev_y') == py)
                if same_cell or swapped:
                    enemy['alive'] = False
                    kill_effects.append([enemy['x'], enemy['y'], 3])

            # Key collection
            if key_pos is not None and px == key_pos[0] and py == key_pos[1]:
                key_collected = True

            # Update trail
            trail.append((px, py))
            if len(trail) > trail_max:
                trail = trail[-trail_max:]

            # Tick kill effect TTLs every step (not just rendered frames)
            new_effects = []
            for effect in kill_effects:
                if effect[2] > 0:
                    effect[2] -= 1
                    new_effects.append(effect)
            kill_effects = new_effects

            # Draw frame
            frame = bg_rgba.copy()
            draw = ImageDraw.Draw(frame, 'RGBA')

            # Draw key if not collected
            if not key_collected and key_pos is not None:
                kx, ky = key_pos
                frame.paste(key_sprite, ((kx + pad) * scale, (ky + pad) * scale), key_sprite)

            # Draw alive enemies
            for enemy in enemies:
                if enemy['alive']:
                    frame.paste(bat_sprite,
                                ((enemy['x'] + pad) * scale, (enemy['y'] + pad) * scale),
                                bat_sprite)

            # Draw player trail (green semi-transparent dots)
            for i, (tx, ty) in enumerate(trail[:-1]):
                alpha = int(80 + 140 * i / max(1, len(trail) - 1))
                cx = (tx + pad) * scale + scale // 2
                cy = (ty + pad) * scale + scale // 2
                r = 3
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             fill=(0, 200, 0, alpha))

            # Draw player
            frame.paste(player_sprite,
                        ((px + pad) * scale, (py + pad) * scale),
                        player_sprite)

            # Draw kill effects (red X) on top of everything
            for effect in kill_effects:
                ex, ey, ttl = effect
                cx = (ex + pad) * scale + scale // 2
                cy = (ey + pad) * scale + scale // 2
                alpha = int(255 * (ttl + 1) / 3)
                r = scale // 2 - 1
                draw.line([(cx - r, cy - r), (cx + r, cy + r)],
                          fill=(255, 50, 50, alpha), width=3)
                draw.line([(cx - r, cy + r), (cx + r, cy - r)],
                          fill=(255, 50, 50, alpha), width=3)

            # Stats overlay
            if metrics is not None:
                frame = self._compose_with_stats(frame, metrics)

            # Convert RGBA to RGB (black background)
            rgb_frame = Image.new('RGB', frame.size, (0, 0, 0))
            rgb_frame.paste(frame, mask=frame.split()[3])
            frames.append(rgb_frame)

            frame_idx = len(frames) - 1
            if frame_idx == 0 or step_idx == num_steps - 1:
                durations.append(hold_duration)
            else:
                durations.append(frame_duration)

        if len(frames) < 2:
            return None

        gif_path = self.run_dir / "simulation.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
        )
        return gif_path

    def get_run_dir(self) -> Path:
        """Get the run directory path."""
        return self.run_dir
