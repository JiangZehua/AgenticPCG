#!/usr/bin/env python3
"""Render a static trajectory image for SMB simulation runs.

Shows spawn positions (semi-transparent sprites), movement trajectories
(colored dashed lines for player and each enemy), and red X marks where
enemies are killed.

Usage:
    uv run python render_trajectory.py runs/some_smb_run/
    uv run python render_trajectory.py runs/run1/ runs/run2/ --scale 2
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from toolusepcg.config import load_config, get_problem_type
from toolusepcg.evaluator import StatsEvaluator
from toolusepcg.factory import LevelFactory
from toolusepcg.level import Level

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENGINE_PAD = 3
PAD_PX = ENGINE_PAD * 16

TILE_TO_SPRITE_TYPE = {7: 2, 8: 6, 9: 8}

ENEMY_SPRITE_FILES = {
    2: "gomba.png",
    3: "gomba.png",
    4: "redkoopa.png",
    5: "redkoopa.png",
    6: "greenkoopa.png",
    7: "greenkoopa.png",
    8: "spiky.png",
    9: "spiky.png",
}

KILL_EVENT_TYPES = {2, 3, 4}  # STOMP, FIRE, SHELL (exclude FALL_KILL=5)

MARIO_COLOR = (220, 40, 40, 220)
ENEMY_PALETTE = [
    (30, 180, 30, 200),
    (40, 100, 255, 200),
    (255, 150, 0, 200),
    (180, 40, 180, 200),
    (230, 230, 30, 200),
    (0, 190, 190, 200),
    (255, 80, 140, 200),
    (160, 100, 40, 200),
    (100, 255, 100, 200),
    (100, 100, 255, 200),
]

SKY_BLUE = (109, 143, 252)

# Sprites are 16x16, anchored at bottom-center. Shift trajectory lines up
# by half the sprite height so they pass through the sprite center.
SPRITE_CENTER_OFFSET_Y = -8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_sprites():
    """Load Mario and enemy sprites from the SMB tile images directory."""
    smbtile_dir = (
        Path(__file__).resolve().parent
        / "submodules"
        / "pcg_benchmark"
        / "pcg_benchmark"
        / "probs"
        / "smbtile"
        / "images"
    )
    mario = Image.open(smbtile_dir / "mario.png").convert("RGBA")
    loaded: dict[str, Image.Image] = {}
    enemies: dict[int, Image.Image] = {}
    for type_val, fname in ENEMY_SPRITE_FILES.items():
        if fname not in loaded:
            loaded[fname] = Image.open(smbtile_dir / fname).convert("RGBA")
        enemies[type_val] = loaded[fname]
    return mario, enemies


def _make_transparent(sprite: Image.Image, alpha: int = 80) -> Image.Image:
    """Return a copy of *sprite* with every pixel's alpha capped at *alpha*."""
    s = sprite.copy()
    r_ch, g_ch, b_ch, a_ch = s.split()
    a_ch = a_ch.point(lambda a: min(a, alpha))
    s = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch))
    return s


def _engine_to_image(px: float, py: float) -> tuple[float, float]:
    """Convert engine pixel coordinates to image coordinates.

    The rendered level image already includes 3-tile padding on each side
    (added by _convert2str in problem.py), so engine pixel coords map
    directly to image coords — no offset needed.
    """
    return px, py


def _initial_enemy_spawns(level_arr: np.ndarray) -> list[tuple[int, float, float]]:
    """Compute enemy spawn positions from level tiles.

    Returns list of (sprite_type, engine_px, engine_py).
    """
    spawns = []
    for row in range(level_arr.shape[0]):
        for col in range(level_arr.shape[1]):
            tile = int(level_arr[row][col])
            if tile in TILE_TO_SPRITE_TYPE:
                sprite_type = TILE_TO_SPRITE_TYPE[tile]
                px = (col + ENGINE_PAD) * 16 + 8
                py = row * 16 + 15
                spawns.append((sprite_type, px, py))
    return spawns


# ---------------------------------------------------------------------------
# Enemy tracking
# ---------------------------------------------------------------------------
def _track_enemies(enemy_frames: list) -> list[dict]:
    """Track individual enemies across frames using greedy nearest-neighbor.

    Returns a list of track dicts, each with keys:
        id, sprite_type, positions [(engine_x, engine_y), ...],
        first_frame, last_frame, killed, kill_pos
    """
    if not enemy_frames:
        return []

    tracks: list[dict] = []
    active: dict[int, tuple[int, float, float]] = {}  # tid -> (type, x, y)
    next_id = 0

    for fi, frame in enumerate(enemy_frames):
        # Build candidate (distance, tid, enemy_idx, ex, ey) pairs
        candidates: list[tuple[float, int, int, float, float]] = []
        for tid, (ttype, tx, ty) in active.items():
            for ei, einfo in enumerate(frame):
                etype = int(einfo[0])
                if etype != ttype:
                    continue
                ex, ey = float(einfo[1]), float(einfo[2])
                d = abs(ex - tx) + abs(ey - ty)
                if d < 80:
                    candidates.append((d, tid, ei, ex, ey))

        # Greedy match by increasing distance
        candidates.sort()
        used_tracks: set[int] = set()
        used_enemies: set[int] = set()
        for _d, tid, ei, ex, ey in candidates:
            if tid in used_tracks or ei in used_enemies:
                continue
            used_tracks.add(tid)
            used_enemies.add(ei)
            tracks[tid]["positions"].append((ex, ey))
            active[tid] = (active[tid][0], ex, ey)

        # End unmatched tracks
        for tid in list(active):
            if tid not in used_tracks:
                tracks[tid]["last_frame"] = fi - 1
                del active[tid]

        # Start new tracks for unmatched enemies
        for ei, einfo in enumerate(frame):
            if ei in used_enemies:
                continue
            etype = int(einfo[0])
            ex, ey = float(einfo[1]), float(einfo[2])
            tracks.append(
                {
                    "id": next_id,
                    "sprite_type": etype,
                    "positions": [(ex, ey)],
                    "first_frame": fi,
                    "last_frame": None,
                    "killed": False,
                    "kill_pos": None,
                }
            )
            active[next_id] = (etype, ex, ey)
            next_id += 1

    # Close remaining active tracks
    for tid in active:
        tracks[tid]["last_frame"] = len(enemy_frames) - 1

    return tracks


def _mark_kills(tracks: list[dict], game_events: list[dict]) -> None:
    """Cross-reference kill game_events with tracked enemies (in-place).

    Matches each kill event to the enemy track that:
      1. ended (disappeared) within a few frames of the kill time,
      2. matches the killed sprite type (via event ``param``), and
      3. was closest to Mario at the moment of the kill.
    """
    kill_events = sorted(
        [e for e in game_events if e["type"] in KILL_EVENT_TYPES],
        key=lambda e: e["time"],
    )
    killed_ids: set[int] = set()

    for ke in kill_events:
        t = ke["time"]
        mx, my = ke["mario_x"], ke["mario_y"]
        ke_sprite = ke.get("param")  # sprite type of the killed enemy

        best_track = None
        best_dist = float("inf")

        for track in tracks:
            if track["id"] in killed_ids:
                continue
            ff = track["first_frame"]
            lf = track["last_frame"]
            if lf is None:
                lf = ff + len(track["positions"]) - 1

            # Track must have started before the kill
            if ff > t:
                continue
            # Track must have ended near the kill time (enemy disappeared)
            if abs(lf - t) > 3:
                continue

            # Sprite type must match if available
            if ke_sprite is not None and track["sprite_type"] != ke_sprite:
                continue

            # Enemy position at kill time
            pos_idx = max(0, min(t - ff, len(track["positions"]) - 1))
            ex, ey = track["positions"][pos_idx]
            dist = abs(ex - mx) + abs(ey - my)

            if dist < best_dist:
                best_dist = dist
                best_track = track

        if best_track is not None and best_dist < 120:
            best_track["killed"] = True
            ff = best_track["first_frame"]
            pos_idx = max(0, min(t - ff, len(best_track["positions"]) - 1))
            best_track["kill_pos"] = best_track["positions"][pos_idx]
            killed_ids.add(best_track["id"])


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _draw_dashed_path(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple,
    dash: int = 8,
    gap: int = 5,
    width: int = 2,
) -> None:
    """Draw a dashed polyline through *points*."""
    dist_acc = 0.0
    cycle = dash + gap
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len < 0.5:
            dist_acc += seg_len
            continue
        ux, uy = dx / seg_len, dy / seg_len
        pos = 0.0
        while pos < seg_len:
            cycle_pos = (dist_acc + pos) % cycle
            if cycle_pos < dash:
                draw_len = min(dash - cycle_pos, seg_len - pos)
                sx = x1 + ux * pos
                sy = y1 + uy * pos
                ex = x1 + ux * (pos + draw_len)
                ey = y1 + uy * (pos + draw_len)
                draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
                pos += draw_len
            else:
                skip = min(cycle - cycle_pos, seg_len - pos)
                pos += skip
        dist_acc += seg_len


def _draw_kill_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    size: int = 6,
    color: tuple = (255, 0, 0, 255),
    width: int = 2,
) -> None:
    """Draw a red X at (x, y)."""
    draw.line(
        [(x - size, y - size), (x + size, y + size)], fill=color, width=width
    )
    draw.line(
        [(x - size, y + size), (x + size, y - size)], fill=color, width=width
    )


def _subsample(positions: list, step: int = 3) -> list:
    """Subsample positions, always keeping first and last."""
    if len(positions) <= 2:
        return list(positions)
    result = [positions[i] for i in range(0, len(positions), step)]
    if result[-1] != positions[-1]:
        result.append(positions[-1])
    return result


# ---------------------------------------------------------------------------
# Main rendering
# ---------------------------------------------------------------------------
def render_trajectory_for_run(
    run_dir: Path, scale: int = 2, subsample: int = 3
) -> bool:
    """Render a trajectory PNG for a single SMB run directory."""
    config_path = run_dir / "config.yaml"
    level_path = run_dir / "final_level.txt"

    if not config_path.exists() or not level_path.exists():
        print(f"  SKIP: missing config.yaml or final_level.txt in {run_dir}")
        return False

    config = load_config(str(config_path))
    problem_type = get_problem_type(config.env.name)

    if problem_type != "smb":
        print(f"  SKIP: problem type is '{problem_type}', not smb")
        return False

    factory = LevelFactory(config)
    env = factory.env
    evaluator = StatsEvaluator(config, env)
    level = Level.from_string(level_path.read_text(), problem_type=problem_type)
    result = evaluator.evaluate(level)
    info = result.info or {}

    locations = info.get("locations", [])
    if not locations:
        print(f"  SKIP: no simulation data (level may be unsolvable)")
        return False

    enemy_frames = info.get("enemy_frames", [])
    game_events = info.get("game_events", [])

    # Render clean background (enemies removed)
    level_arr = np.array(level.data)
    bg_level = level_arr.copy()
    bg_level[np.isin(bg_level, list(TILE_TO_SPRITE_TYPE.keys()))] = 0

    background = env._problem.render(bg_level)
    if background is None:
        print(f"  FAIL: could not render level")
        return False

    canvas = background.convert("RGBA")

    # Load sprites
    mario_sprite, enemy_sprites = _load_sprites()

    # Track enemies & mark kills
    tracks = _track_enemies(enemy_frames)
    _mark_kills(tracks, game_events)

    # Enemy spawn positions (engine coords)
    initial_enemies = _initial_enemy_spawns(level_arr)

    draw = ImageDraw.Draw(canvas, "RGBA")

    # --- 1. Semi-transparent enemy sprites at spawn positions ---
    for sprite_type, epx, epy in initial_enemies:
        ix, iy = _engine_to_image(epx, epy)
        esprite = enemy_sprites.get(sprite_type)
        if esprite:
            trans = _make_transparent(esprite, alpha=90)
            canvas.paste(
                trans,
                (int(ix) - trans.width // 2, int(iy) - trans.height),
                trans,
            )

    # --- 2. Semi-transparent Mario sprite at spawn ---
    mx0, my0 = _engine_to_image(
        float(locations[0][0]), float(locations[0][1])
    )
    trans_mario = _make_transparent(mario_sprite, alpha=90)
    canvas.paste(
        trans_mario,
        (int(mx0) - trans_mario.width // 2, int(my0) - trans_mario.height),
        trans_mario,
    )

    # Refresh draw handle after pastes
    draw = ImageDraw.Draw(canvas, "RGBA")

    # --- 3. Enemy trajectories (colored dashed lines) ---
    sorted_tracks = sorted(tracks, key=lambda t: t["positions"][0][0])
    for i, track in enumerate(sorted_tracks):
        color = ENEMY_PALETTE[i % len(ENEMY_PALETTE)]
        img_pos = [_engine_to_image(x, y + SPRITE_CENTER_OFFSET_Y) for x, y in track["positions"]]
        sampled = _subsample(img_pos, step=subsample)
        if len(sampled) >= 2:
            _draw_dashed_path(draw, sampled, color, dash=6, gap=4, width=2)

    # --- 4. Mario trajectory (red dashed line) ---
    mario_img = [
        _engine_to_image(float(l[0]), float(l[1]) + SPRITE_CENTER_OFFSET_Y) for l in locations
    ]
    mario_sampled = _subsample(mario_img, step=subsample)
    if len(mario_sampled) >= 2:
        _draw_dashed_path(
            draw, mario_sampled, MARIO_COLOR, dash=8, gap=5, width=2
        )

    # --- 5. Kill markers (red X) ---
    for track in tracks:
        if track["killed"] and track["kill_pos"]:
            kx, ky = _engine_to_image(*track["kill_pos"])
            _draw_kill_marker(draw, kx, ky + SPRITE_CENTER_OFFSET_Y)

    # Scale up for visibility
    if scale > 1:
        canvas = canvas.resize(
            (canvas.width * scale, canvas.height * scale), Image.NEAREST
        )

    # Save as PNG (convert RGBA -> RGB with sky-blue background)
    out = Image.new("RGB", canvas.size, SKY_BLUE)
    out.paste(canvas, mask=canvas.split()[3])

    out_path = run_dir / "trajectory.png"
    out.save(out_path)

    n_killed = sum(1 for t in tracks if t["killed"])
    print(
        f"  OK: {out_path} ({len(locations)} frames, "
        f"{len(tracks)} enemies tracked, {n_killed} killed)"
    )
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Render static trajectory image for SMB simulation runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Run directories containing config.yaml and final_level.txt",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Upscale factor for output image (default: 2)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=3,
        help="Draw every Nth trajectory position (default: 3)",
    )
    args = parser.parse_args()

    success = 0
    for run_dir in args.run_dirs:
        print(f"[{run_dir}]")
        try:
            if render_trajectory_for_run(
                run_dir, scale=args.scale, subsample=args.subsample
            ):
                success += 1
        except Exception as e:
            import traceback

            print(f"  ERROR: {e}")
            traceback.print_exc()

    if len(args.run_dirs) > 1:
        print(f"\nRendered {success}/{len(args.run_dirs)} trajectory images.")

    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
