"""Validation-only scaffold for the straight-line versus quadratic-Bezier study.

The primary comparison is an exact-pixel renderer-capacity experiment.  This
module deliberately provides no authorized comparative execution entry point.
Target hashes and the execution authorization must be frozen in a later commit
after the complete local validation gate passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil, hypot, isfinite, pi, sin
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TypeAlias

import numpy as np
from PIL import Image, ImageDraw

from .rgb_coarse_to_fine import (
    DEFAULT_STAGES,
    PainterConfig,
    RGBStroke,
    StageConfig,
    _pool_rng,
    _sample_geometry,
    _sample_midpoint,
    blank_rgb_canvas,
    file_sha256,
    pixel_mae,
    pixel_mse,
    propose_rgb_strokes,
    render_rgb_stroke,
)


PROTOCOL_ID = "quadratic_bezier_extension_v1"
BASE_COMMIT = "d5f1190ab9b62d5adff7fb56c5cc1ffd4d850177"
DEFAULT_CONFIG_PATH = Path("configs/quadratic-bezier-extension-2026-09-03.json")
PRIMITIVES = ("straight", "quadratic_bezier")
SEEDS = (73, 137, 211)
TARGET_IDS = (
    "01_ring_symbol",
    "02_curved_glyph",
    "03_organic_silhouette",
    "04_mixed_geometry",
    "05_layered_landscape",
    "06_dense_scene",
)
PrimitiveName = Literal["straight", "quadratic_bezier"]


@dataclass(frozen=True)
class QuadraticBezierStroke:
    """An opaque RGB quadratic Bezier stroke in normalized coordinates."""

    x0: float
    y0: float
    cx: float
    cy: float
    x1: float
    y1: float
    width: int
    color: tuple[int, int, int]

    def __post_init__(self) -> None:
        coordinates = (self.x0, self.y0, self.cx, self.cy, self.x1, self.y1)
        if any(not isfinite(value) for value in coordinates):
            raise ValueError("Bezier coordinates must be finite.")
        if any(not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError("Bezier coordinates must lie in [0, 1].")
        if self.width < 1:
            raise ValueError("Bezier width must be at least one pixel.")
        if len(self.color) != 3 or any(not 0 <= value <= 255 for value in self.color):
            raise ValueError("Bezier color must contain three values in [0, 255].")


Stroke: TypeAlias = RGBStroke | QuadraticBezierStroke


@dataclass(frozen=True)
class GeneratedTarget:
    target_id: str
    category: str
    provenance: str
    image: Image.Image


@dataclass
class PrimitivePlanResult:
    initial_canvas: Image.Image
    best_canvas: Image.Image
    final_canvas: Image.Image
    best_step: int
    initial_mse: float
    best_mse: float
    final_mse: float
    final_mae: float
    strokes: tuple[Stroke, ...]
    progress: tuple[dict[str, Any], ...]
    stage_stats: tuple[dict[str, Any], ...]


def _to_pixel(value: float, size: int) -> float:
    return max(0.0, min(1.0, float(value))) * (size - 1)


def quadratic_bezier_points(
    stroke: QuadraticBezierStroke,
    size: int,
) -> tuple[tuple[int, int], ...]:
    """Return a deterministic, adaptively sampled raster polyline."""

    if size < 8:
        raise ValueError("Bezier raster size must be at least 8.")
    p0 = (_to_pixel(stroke.x0, size), _to_pixel(stroke.y0, size))
    pc = (_to_pixel(stroke.cx, size), _to_pixel(stroke.cy, size))
    p1 = (_to_pixel(stroke.x1, size), _to_pixel(stroke.y1, size))
    control_length = hypot(pc[0] - p0[0], pc[1] - p0[1]) + hypot(
        p1[0] - pc[0], p1[1] - pc[1]
    )
    sample_count = max(8, min(2048, int(ceil(control_length * 2.0)) + 1))
    values = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    one_minus = 1.0 - values
    xs = one_minus * one_minus * p0[0] + 2.0 * one_minus * values * pc[0] + values * values * p1[0]
    ys = one_minus * one_minus * p0[1] + 2.0 * one_minus * values * pc[1] + values * values * p1[1]
    points: list[tuple[int, int]] = []
    for x_value, y_value in zip(xs, ys, strict=True):
        point = (int(round(float(x_value))), int(round(float(y_value))))
        if not points or point != points[-1]:
            points.append(point)
    if len(points) == 1:
        points.append(points[0])
    return tuple(points)


def render_quadratic_bezier(
    canvas: Image.Image,
    stroke: QuadraticBezierStroke,
) -> Image.Image:
    """Render one Bezier stroke without mutating the input image."""

    if canvas.mode != "RGB" or canvas.width != canvas.height:
        raise ValueError("Expected a square RGB canvas.")
    result = canvas.copy()
    draw = ImageDraw.Draw(result)
    draw.line(
        quadratic_bezier_points(stroke, result.width),
        fill=stroke.color,
        width=stroke.width,
    )
    return result


def quadratic_bezier_mask(
    stroke: QuadraticBezierStroke,
    size: int,
) -> np.ndarray:
    mask = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(mask)
    draw.line(
        quadratic_bezier_points(stroke, size),
        fill=255,
        width=stroke.width,
    )
    return np.asarray(mask, dtype=np.uint8) != 0


def render_extension_stroke(canvas: Image.Image, stroke: Stroke) -> Image.Image:
    if isinstance(stroke, RGBStroke):
        return render_rgb_stroke(canvas, stroke)
    if isinstance(stroke, QuadraticBezierStroke):
        return render_quadratic_bezier(canvas, stroke)
    raise TypeError(f"Unsupported stroke type: {type(stroke).__name__}")


def stroke_to_record(stroke: Stroke) -> dict[str, Any]:
    if isinstance(stroke, RGBStroke):
        return {
            "primitive": "straight",
            "x0": stroke.x0,
            "y0": stroke.y0,
            "x1": stroke.x1,
            "y1": stroke.y1,
            "width": stroke.width,
            "color": list(stroke.color),
        }
    if isinstance(stroke, QuadraticBezierStroke):
        return {
            "primitive": "quadratic_bezier",
            "x0": stroke.x0,
            "y0": stroke.y0,
            "cx": stroke.cx,
            "cy": stroke.cy,
            "x1": stroke.x1,
            "y1": stroke.y1,
            "width": stroke.width,
            "color": list(stroke.color),
        }
    raise TypeError(f"Unsupported stroke type: {type(stroke).__name__}")


def stroke_from_record(record: Mapping[str, Any]) -> Stroke:
    primitive = record.get("primitive")
    color_value = record.get("color")
    if not isinstance(color_value, Sequence) or len(color_value) != 3:
        raise ValueError("Serialized stroke has an invalid color.")
    color = tuple(int(value) for value in color_value)
    if primitive == "straight":
        return RGBStroke(
            float(record["x0"]),
            float(record["y0"]),
            float(record["x1"]),
            float(record["y1"]),
            int(record["width"]),
            color,
        )
    if primitive == "quadratic_bezier":
        return QuadraticBezierStroke(
            float(record["x0"]),
            float(record["y0"]),
            float(record["cx"]),
            float(record["cy"]),
            float(record["x1"]),
            float(record["y1"]),
            int(record["width"]),
            color,
        )
    raise ValueError(f"Unknown serialized primitive: {primitive!r}")


def _sample_curve_geometry(
    midpoint: tuple[float, float],
    stage: StageConfig,
    size: int,
    rng: np.random.Generator,
) -> QuadraticBezierStroke:
    base = _sample_geometry(midpoint, stage, size, rng)
    dx = base.x1 - base.x0
    dy = base.y1 - base.y0
    chord_length = hypot(dx, dy)
    if chord_length <= 0.0:
        raise RuntimeError("Sampled a degenerate Bezier chord.")
    along = float(rng.uniform(-0.15, 0.15))
    bend = float(rng.uniform(0.08, 0.45))
    side = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
    middle_x = 0.5 * (base.x0 + base.x1) + along * dx
    middle_y = 0.5 * (base.y0 + base.y1) + along * dy
    normal_x = -dy / chord_length
    normal_y = dx / chord_length
    cx = max(0.0, min(1.0, middle_x + side * bend * chord_length * normal_x))
    cy = max(0.0, min(1.0, middle_y + side * bend * chord_length * normal_y))
    return QuadraticBezierStroke(
        base.x0,
        base.y0,
        cx,
        cy,
        base.x1,
        base.y1,
        base.width,
        (0, 0, 0),
    )


def _fit_curve_color(
    target_values: np.ndarray,
    geometry: QuadraticBezierStroke,
) -> tuple[int, int, int]:
    mask = quadratic_bezier_mask(geometry, target_values.shape[0])
    if not bool(mask.any()):
        raise RuntimeError("A proposed Bezier stroke covered no pixels.")
    mean_color = np.rint(target_values[mask].mean(axis=0))
    clipped = np.clip(mean_color, 0, 255).astype(np.uint8)
    return (int(clipped[0]), int(clipped[1]), int(clipped[2]))


def propose_quadratic_bezier_strokes(
    current: Image.Image,
    target: Image.Image,
    stage: StageConfig,
    rng: np.random.Generator,
    *,
    count: int,
    error_guided_fraction: float,
    max_attempts_per_candidate: int,
) -> tuple[QuadraticBezierStroke, ...]:
    """Generate unique, changing, target-colored quadratic candidates."""

    if current.mode != "RGB" or target.mode != "RGB":
        raise ValueError("Bezier proposals require RGB images.")
    if current.size != target.size or current.width != current.height:
        raise ValueError("Bezier proposal images must be matching squares.")
    if count < 1:
        raise ValueError("Candidate count must be positive.")
    if not 0.0 <= error_guided_fraction <= 1.0:
        raise ValueError("error_guided_fraction must lie in [0, 1].")

    current_values = np.asarray(current, dtype=np.uint8)
    target_values = np.asarray(target, dtype=np.uint8)
    difference = current_values.astype(np.float64) - target_values.astype(np.float64)
    error = np.mean(difference * difference, axis=2)
    guided_count = int(round(count * error_guided_fraction))
    candidates: list[QuadraticBezierStroke] = []
    seen_outcomes: set[bytes] = set()
    attempts = 0
    maximum_attempts = count * max_attempts_per_candidate

    while len(candidates) < count and attempts < maximum_attempts:
        attempts += 1
        midpoint = _sample_midpoint(error, len(candidates) < guided_count, rng)
        geometry = _sample_curve_geometry(midpoint, stage, current.width, rng)
        color = _fit_curve_color(target_values, geometry)
        stroke = QuadraticBezierStroke(
            geometry.x0,
            geometry.y0,
            geometry.cx,
            geometry.cy,
            geometry.x1,
            geometry.y1,
            geometry.width,
            color,
        )
        rendered_values = np.asarray(
            render_quadratic_bezier(current, stroke),
            dtype=np.uint8,
        )
        if np.array_equal(rendered_values, current_values):
            continue
        signature = rendered_values.tobytes()
        if signature in seen_outcomes:
            continue
        seen_outcomes.add(signature)
        candidates.append(stroke)

    if len(candidates) != count:
        raise RuntimeError(
            f"Generated {len(candidates)} unique changing Bezier candidates; "
            f"required {count}."
        )
    return tuple(candidates)


def extension_stages() -> tuple[StageConfig, ...]:
    return tuple(
        StageConfig(
            stage.name,
            stage.max_steps * 2,
            stage.min_length,
            stage.max_length,
            stage.min_width,
            stage.max_width,
        )
        for stage in DEFAULT_STAGES
    )


def extension_painter_config(seed: int) -> PainterConfig:
    if seed not in SEEDS:
        raise ValueError(f"Seed {seed} is not in the frozen seed set.")
    config = PainterConfig(
        planning_size=128,
        replay_size=512,
        supersample=2,
        candidates_per_pool=64,
        error_guided_fraction=0.80,
        patience=12,
        min_improvement=1e-9,
        seed=seed,
        gif_stride=3,
        max_attempts_per_candidate=400,
        stages=extension_stages(),
    )
    config.validate()
    return config


def _propose(
    primitive: PrimitiveName,
    current: Image.Image,
    target: Image.Image,
    stage: StageConfig,
    rng: np.random.Generator,
    config: PainterConfig,
) -> tuple[Stroke, ...]:
    if primitive == "straight":
        return propose_rgb_strokes(
            current,
            target,
            stage,
            rng,
            count=config.candidates_per_pool,
            error_guided_fraction=config.error_guided_fraction,
            max_attempts_per_candidate=config.max_attempts_per_candidate,
        )
    if primitive == "quadratic_bezier":
        return propose_quadratic_bezier_strokes(
            current,
            target,
            stage,
            rng,
            count=config.candidates_per_pool,
            error_guided_fraction=config.error_guided_fraction,
            max_attempts_per_candidate=config.max_attempts_per_candidate,
        )
    raise ValueError(f"Unsupported primitive: {primitive}")


def plan_primitive_target(
    target: Image.Image,
    primitive: PrimitiveName,
    config: PainterConfig,
    *,
    target_stream: int,
) -> PrimitivePlanResult:
    """Run deterministic exact-pixel planning with one primitive family."""

    config.validate()
    if primitive not in PRIMITIVES:
        raise ValueError(f"Unsupported primitive: {primitive}")
    if target.mode != "RGB" or target.size != (
        config.planning_size,
        config.planning_size,
    ):
        raise ValueError("Planning target must be a matching square RGB image.")
    if target_stream < 0:
        raise ValueError("target_stream must be non-negative.")

    current = blank_rgb_canvas(config.planning_size)
    initial = current.copy()
    current_mse = pixel_mse(current, target)
    initial_mse = current_mse
    best_mse = current_mse
    best_canvas = current.copy()
    best_step = 0
    strokes: list[Stroke] = []
    progress: list[dict[str, Any]] = []
    stage_stats: list[dict[str, Any]] = []
    global_pool_index = 0

    for stage_index, stage in enumerate(config.stages):
        stage_start_mse = current_mse
        accepted = 0
        rejected_pools = 0
        consecutive_non_improving = 0
        stage_pool_count = 0
        stop_reason = "max_steps"
        while accepted < stage.max_steps:
            if current_mse <= config.min_improvement:
                stop_reason = "exact_match"
                break
            rng = _pool_rng(config.seed, target_stream, stage_index, global_pool_index)
            candidates = _propose(primitive, current, target, stage, rng, config)
            rendered = tuple(render_extension_stroke(current, item) for item in candidates)
            scores = np.asarray(
                [pixel_mse(canvas, target) for canvas in rendered],
                dtype=np.float64,
            )
            selected_index = int(np.argmin(scores))
            selected_mse = float(scores[selected_index])
            improvement = current_mse - selected_mse
            global_pool_index += 1
            stage_pool_count += 1
            if improvement <= config.min_improvement:
                rejected_pools += 1
                consecutive_non_improving += 1
                if consecutive_non_improving >= config.patience:
                    stop_reason = "patience"
                    break
                continue

            mse_before = current_mse
            current = rendered[selected_index]
            current_mse = selected_mse
            selected = candidates[selected_index]
            strokes.append(selected)
            accepted += 1
            consecutive_non_improving = 0
            step = len(strokes)
            if current_mse < best_mse:
                best_mse = current_mse
                best_canvas = current.copy()
                best_step = step
            progress.append(
                {
                    "step": step,
                    "stage": stage.name,
                    "stage_step": accepted,
                    "candidate_pool": global_pool_index,
                    "selected_candidate_index": selected_index,
                    "primitive": primitive,
                    "action": stroke_to_record(selected),
                    "mse_before": mse_before,
                    "mse_after": current_mse,
                    "best_mse": best_mse,
                    "improvement": improvement,
                }
            )

        stage_stats.append(
            {
                "stage": stage.name,
                "maximum_steps": stage.max_steps,
                "executed_steps": accepted,
                "candidate_pools": stage_pool_count,
                "rejected_pools": rejected_pools,
                "start_mse": stage_start_mse,
                "end_mse": current_mse,
                "stop_reason": stop_reason,
            }
        )

    return PrimitivePlanResult(
        initial_canvas=initial,
        best_canvas=best_canvas,
        final_canvas=current,
        best_step=best_step,
        initial_mse=initial_mse,
        best_mse=best_mse,
        final_mse=current_mse,
        final_mae=pixel_mae(current, target),
        strokes=tuple(strokes),
        progress=tuple(progress),
        stage_stats=tuple(stage_stats),
    )


def _render_target_curve(
    image: Image.Image,
    coordinates: tuple[float, float, float, float, float, float],
    *,
    width: int,
    color: tuple[int, int, int],
) -> Image.Image:
    return render_quadratic_bezier(
        image,
        QuadraticBezierStroke(*coordinates, width, color),
    )


def generate_rights_safe_targets(size: int = 512) -> tuple[GeneratedTarget, ...]:
    """Generate six deterministic, original/procedural RGB targets."""

    if size < 128:
        raise ValueError("Rights-safe targets must be at least 128 pixels.")

    targets: list[GeneratedTarget] = []

    ring = blank_rgb_canvas(size)
    draw = ImageDraw.Draw(ring)
    margin = int(round(size * 0.16))
    width = max(3, int(round(size * 0.035)))
    draw.ellipse((margin, margin, size - margin, size - margin), outline=(25, 35, 70), width=width)
    draw.ellipse((int(size * 0.34), int(size * 0.34), int(size * 0.66), int(size * 0.66)), outline=(220, 70, 65), width=max(2, width // 2))
    draw.line((size // 2, margin, size // 2, size - margin), fill=(25, 35, 70), width=max(2, width // 3))
    targets.append(GeneratedTarget(TARGET_IDS[0], "ring_symbol", "Original deterministic PIL drawing; CC0-equivalent project asset.", ring))

    glyph = blank_rgb_canvas(size)
    glyph = _render_target_curve(glyph, (0.72, 0.16, 0.24, 0.02, 0.28, 0.47), width=max(5, int(size * 0.075)), color=(38, 80, 165))
    glyph = _render_target_curve(glyph, (0.28, 0.47, 0.76, 0.54, 0.31, 0.84), width=max(5, int(size * 0.075)), color=(38, 80, 165))
    glyph = _render_target_curve(glyph, (0.31, 0.84, 0.48, 0.98, 0.72, 0.82), width=max(3, int(size * 0.035)), color=(238, 145, 45))
    targets.append(GeneratedTarget(TARGET_IDS[1], "curved_glyph", "Original deterministic Bezier glyph; CC0-equivalent project asset.", glyph))

    organic = Image.new("RGB", (size, size), color=(244, 239, 221))
    draw = ImageDraw.Draw(organic)
    left = []
    right = []
    for index in range(81):
        t = index / 80.0
        y = int(round(size * (0.10 + 0.78 * t)))
        half = size * 0.28 * sin(pi * t) ** 0.82
        center = size * (0.50 + 0.035 * sin(2.0 * pi * t))
        left.append((int(round(center - half)), y))
        right.append((int(round(center + half)), y))
    draw.polygon(left + list(reversed(right)), fill=(85, 156, 92), outline=(28, 88, 56))
    organic = _render_target_curve(organic, (0.50, 0.12, 0.43, 0.52, 0.52, 0.91), width=max(3, int(size * 0.018)), color=(246, 230, 168))
    targets.append(GeneratedTarget(TARGET_IDS[2], "organic_silhouette", "Original procedural leaf silhouette; CC0-equivalent project asset.", organic))

    mixed = Image.new("RGB", (size, size), color=(221, 238, 226))
    draw = ImageDraw.Draw(mixed)
    draw.rectangle((int(size * 0.08), int(size * 0.10), int(size * 0.45), int(size * 0.42)), fill=(239, 184, 72), outline=(45, 60, 70), width=max(2, size // 100))
    draw.ellipse((int(size * 0.54), int(size * 0.10), int(size * 0.90), int(size * 0.46)), fill=(105, 164, 220), outline=(45, 60, 70), width=max(2, size // 100))
    draw.polygon(((int(size * 0.10), int(size * 0.84)), (int(size * 0.38), int(size * 0.53)), (int(size * 0.53), int(size * 0.88))), fill=(215, 90, 105))
    mixed = _render_target_curve(mixed, (0.49, 0.62, 0.72, 0.43, 0.90, 0.82), width=max(4, int(size * 0.045)), color=(88, 62, 135))
    targets.append(GeneratedTarget(TARGET_IDS[3], "mixed_geometry", "Original deterministic geometric composition; CC0-equivalent project asset.", mixed))

    landscape = Image.new("RGB", (size, size), color=(145, 205, 239))
    draw = ImageDraw.Draw(landscape)
    draw.ellipse((int(size * 0.72), int(size * 0.08), int(size * 0.91), int(size * 0.27)), fill=(250, 208, 85))
    ridge = [(0, int(size * 0.58))]
    for x in range(size):
        y = size * (0.53 + 0.06 * sin(2.0 * pi * x / size) + 0.025 * sin(6.0 * pi * x / size))
        ridge.append((x, int(round(y))))
    ridge.extend(((size - 1, size - 1), (0, size - 1)))
    draw.polygon(ridge, fill=(75, 115, 112))
    foreground = [(0, int(size * 0.70))]
    for x in range(size):
        y = size * (0.70 + 0.045 * sin(2.0 * pi * x / size + 0.7))
        foreground.append((x, int(round(y))))
    foreground.extend(((size - 1, size - 1), (0, size - 1)))
    draw.polygon(foreground, fill=(74, 145, 78))
    landscape = _render_target_curve(landscape, (0.08, 0.92, 0.48, 0.66, 0.92, 0.90), width=max(4, int(size * 0.045)), color=(75, 145, 210))
    targets.append(GeneratedTarget(TARGET_IDS[4], "layered_landscape", "Original procedural landscape; CC0-equivalent project asset.", landscape))

    dense = Image.new("RGB", (size, size), color=(205, 218, 232))
    draw = ImageDraw.Draw(dense)
    building_colors = ((52, 63, 78), (76, 84, 96), (101, 88, 82), (62, 78, 94))
    for index in range(12):
        x0 = int(round(size * (0.025 + index * 0.081)))
        width_px = int(round(size * (0.055 + 0.012 * (index % 3))))
        top = int(round(size * (0.24 + 0.05 * ((index * 7) % 5))))
        draw.rectangle((x0, top, x0 + width_px, int(size * 0.88)), fill=building_colors[index % len(building_colors)])
        for row in range(4):
            for column in range(2):
                wx = x0 + int(width_px * (0.20 + 0.42 * column))
                wy = top + int(size * (0.07 + 0.11 * row))
                draw.rectangle((wx, wy, wx + max(2, width_px // 7), wy + max(3, size // 55)), fill=(235, 190, 92))
    draw.rectangle((0, int(size * 0.88), size, size), fill=(55, 70, 60))
    dense = _render_target_curve(dense, (0.04, 0.30, 0.50, 0.05, 0.96, 0.31), width=max(2, int(size * 0.012)), color=(38, 42, 50))
    dense = _render_target_curve(dense, (0.10, 0.48, 0.52, 0.24, 0.90, 0.50), width=max(2, int(size * 0.009)), color=(38, 42, 50))
    targets.append(GeneratedTarget(TARGET_IDS[5], "dense_scene", "Original procedural city scene; CC0-equivalent project asset.", dense))

    return tuple(targets)


def image_pixel_sha256(image: Image.Image) -> str:
    if image.mode != "RGB":
        raise ValueError("Target hash expects RGB pixels.")
    payload = image.width.to_bytes(4, "big") + image.height.to_bytes(4, "big") + image.tobytes()
    return sha256(payload).hexdigest()


def generated_target_manifest(size: int = 512) -> dict[str, Any]:
    targets = generate_rights_safe_targets(size)
    entries = []
    for target in targets:
        entries.append(
            {
                "target_id": target.target_id,
                "category": target.category,
                "provenance": target.provenance,
                "width": target.image.width,
                "height": target.image.height,
                "mode": target.image.mode,
                "pixel_sha256": image_pixel_sha256(target.image),
            }
        )
    ordered_payload = "\n".join(
        f"{item['target_id']}\0{item['pixel_sha256']}" for item in entries
    )
    return {
        "generator": "deterministic_procedural_pillow_v1",
        "target_count": len(entries),
        "target_set_sha256": sha256(ordered_payload.encode("utf-8")).hexdigest(),
        "targets": entries,
    }


def load_protocol_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Extension protocol config must be a JSON object.")
    return value


def validate_protocol_config(config: Mapping[str, Any]) -> dict[str, Any]:
    exact_values = {
        "protocol_id": PROTOCOL_ID,
        "status": "implementation_validation_only",
        "branch": "quadratic-bezier-extension",
        "base_commit": BASE_COMMIT,
        "conditions": list(PRIMITIVES),
        "planning_size": 128,
        "replay_size": 512,
        "accepted_strokes": 420,
        "stage_budgets": [80, 140, 200],
        "candidates_per_pool": 64,
        "error_guided_fraction": 0.80,
        "patience": 12,
        "min_improvement": 1e-9,
        "seeds": list(SEEDS),
        "target_count": 6,
        "target_source": "deterministic_procedural_rights_safe",
        "target_hashes_frozen": False,
        "target_set_sha256": None,
        "target_sha256": {},
        "execution_authorized": False,
        "maximum_completed_executions": 1,
        "completed_executions": 0,
        "learned_model_allowed": False,
    }
    for key, expected in exact_values.items():
        if config.get(key) != expected:
            raise ValueError(f"Frozen validation config mismatch for {key}.")
    if config.get("decision_rule") != {
        "minimum_mean_mse_improvement_fraction": 0.05,
        "minimum_improved_target_count": 4,
        "maximum_per_target_worsening_ratio": 1.05,
        "blinded_review_required": True,
    }:
        raise ValueError("Frozen decision rule changed.")
    for seed in SEEDS:
        current = extension_painter_config(seed)
        if [stage.max_steps for stage in current.stages] != [80, 140, 200]:
            raise RuntimeError("Extension stage budget changed.")
    return dict(config)


def _smoke_config(seed: int) -> PainterConfig:
    return PainterConfig(
        planning_size=32,
        replay_size=32,
        supersample=1,
        candidates_per_pool=8,
        error_guided_fraction=0.75,
        patience=3,
        min_improvement=1e-9,
        seed=seed,
        gif_stride=1,
        max_attempts_per_candidate=100,
        stages=(StageConfig("smoke", 3, 0.20, 0.80, 0.12, 0.30),),
    )


def validate_only_report(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Validate protocol, targets, renderers, proposals, and planners in memory."""

    config = validate_protocol_config(load_protocol_config(config_path))
    manifest = generated_target_manifest(512)
    targets = generate_rights_safe_targets(128)
    smoke_target = targets[3].image.resize((32, 32), Image.Resampling.LANCZOS)
    smoke: dict[str, Any] = {}
    for primitive in PRIMITIVES:
        painter = _smoke_config(73)
        first = plan_primitive_target(smoke_target, primitive, painter, target_stream=0)
        second = plan_primitive_target(smoke_target, primitive, painter, target_stream=0)
        deterministic = first.progress == second.progress and np.array_equal(
            np.asarray(first.final_canvas),
            np.asarray(second.final_canvas),
        )
        monotonic = bool(first.progress) and all(
            row["mse_after"] < row["mse_before"]
            and row["improvement"] > painter.min_improvement
            for row in first.progress
        )
        if not deterministic or not monotonic:
            raise RuntimeError(f"{primitive} smoke validation failed.")
        smoke[primitive] = {
            "executed_strokes": len(first.strokes),
            "initial_mse": first.initial_mse,
            "final_mse": first.final_mse,
            "deterministic": deterministic,
            "monotonic": monotonic,
        }

    import PIL

    return {
        "status": "quadratic_bezier_extension_valid_no_outputs",
        "protocol_id": PROTOCOL_ID,
        "config_sha256": file_sha256(config_path),
        "protocol": config,
        "proposed_target_manifest": manifest,
        "synthetic_smoke": smoke,
        "dependencies": {"numpy": np.__version__, "pillow": PIL.__version__},
        "output_side_effects": False,
        "comparative_outputs_viewed": False,
        "training_performed": False,
        "learned_model_used": False,
        "execution_authorized": False,
        "closed_experiments_changed": False,
    }
