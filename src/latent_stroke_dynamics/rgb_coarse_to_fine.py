"""Deterministic exact-pixel RGB coarse-to-fine stroke painter."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import cos, hypot, pi, sin
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps


TARGET_SET_SHA256 = (
    "31e1fcc2bf344f8b72d3f04dfbc9109c61c39fc8cbc10668c1b78d575a673b42"
)


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    category: str
    filename: str
    source_url: str
    sha256: str
    publication_note: str


TARGET_SPECS = (
    TargetSpec(
        target_id="01_symbol",
        category="simple_symbol",
        filename="mercedes-benz-logo-black.jpg",
        source_url="https://www.logodesignlove.com/circular-logos",
        sha256="94ed621d3f74ec56cddd1d59e6536dca0c61adb41c87e5e9d6510765656ad2a7",
        publication_note="Private test; final reproduction requires rights review.",
    ),
    TargetSpec(
        target_id="02_geometric",
        category="geometric_composition",
        filename="slice-vb00CqFW3YA-unsplash.jpg",
        source_url=(
            "https://unsplash.com/illustrations/"
            "abstract-geometric-shapes-in-various-colors-on-light-green-"
            "background-vb00CqFW3YA"
        ),
        sha256="159d5d7ed97542093a513f37097513f2316a146b00b7d1b289e1ce8de4b7fa1a",
        publication_note="Record contributor and item-specific Unsplash licence.",
    ),
    TargetSpec(
        target_id="03_object",
        category="isolated_object",
        filename="painted-vase-ideas-1.jpg",
        source_url="https://thenymelrosefamily.com/painted-vase-ideas/",
        sha256="429104e33e08e7998b0dcfc656873be6f0356cc72627be93ceb94b07700b3254",
        publication_note="Private test until creator and reuse rights are verified.",
    ),
    TargetSpec(
        target_id="04_landscape",
        category="landscape",
        filename=(
            "w1200_9d41_iPhone_14_Pro_Max_-_Telephoto_Lens_-_After_"
            "-1_copy__1_.jpg"
        ),
        source_url=(
            "https://www.sandmarc.com/blogs/iphone-photography/"
            "a-guide-to-landscape-photography"
        ),
        sha256="da60fea3d30ee85605a93537e9d6cf8fe5d12af62bb1dd778956eef55dde6b06",
        publication_note="Private test until photographer and reuse rights are verified.",
    ),
    TargetSpec(
        target_id="05_dense",
        category="dense_detailed_scene",
        filename="anor-londo-sq.jpg",
        source_url="http://darksouls3.wikidot.com/locationgroup:anor-londo",
        sha256="e9dc2de55d1a5fad6a6018de7acf681df8a6a632bfe394845cc84ea88003db52",
        publication_note="Private stress test; do not publish without a rights basis.",
    ),
)


@dataclass(frozen=True)
class RGBStroke:
    """An opaque RGB straight-line stroke."""

    x0: float
    y0: float
    x1: float
    y1: float
    width: int
    color: tuple[int, int, int]

    def __post_init__(self) -> None:
        coordinates = (self.x0, self.y0, self.x1, self.y1)
        if any(not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError("RGB stroke coordinates must lie in [0, 1].")
        if self.width < 1:
            raise ValueError("RGB stroke width must be at least one pixel.")
        if len(self.color) != 3 or any(not 0 <= value <= 255 for value in self.color):
            raise ValueError("RGB stroke color must contain three values in [0, 255].")


@dataclass(frozen=True)
class StageConfig:
    name: str
    max_steps: int
    min_length: float
    max_length: float
    min_width: float
    max_width: float


DEFAULT_STAGES = (
    StageConfig("global", 40, 0.25, 0.75, 0.10, 0.22),
    StageConfig("structure", 70, 0.08, 0.35, 0.035, 0.10),
    StageConfig("detail", 100, 0.02, 0.15, 0.012, 0.045),
)


@dataclass(frozen=True)
class PainterConfig:
    planning_size: int = 96
    replay_size: int = 512
    supersample: int = 2
    candidates_per_pool: int = 64
    error_guided_fraction: float = 0.80
    patience: int = 12
    min_improvement: float = 1e-9
    seed: int = 73
    gif_stride: int = 3
    max_attempts_per_candidate: int = 400
    stages: tuple[StageConfig, ...] = DEFAULT_STAGES

    def validate(self) -> None:
        if self.planning_size < 8:
            raise ValueError("planning_size must be at least 8.")
        if self.replay_size < self.planning_size:
            raise ValueError("replay_size must not be smaller than planning_size.")
        if not 1 <= self.supersample <= 4:
            raise ValueError("supersample must lie between 1 and 4.")
        if self.candidates_per_pool < 1:
            raise ValueError("candidates_per_pool must be positive.")
        if not 0.0 <= self.error_guided_fraction <= 1.0:
            raise ValueError("error_guided_fraction must lie in [0, 1].")
        if self.patience < 1 or self.gif_stride < 1:
            raise ValueError("patience and gif_stride must be positive.")
        if self.min_improvement <= 0.0:
            raise ValueError("min_improvement must be positive.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not self.stages:
            raise ValueError("At least one coarse-to-fine stage is required.")
        for stage in self.stages:
            if stage.max_steps < 1:
                raise ValueError("Every stage must permit at least one step.")
            if not 0.0 < stage.min_length <= stage.max_length <= 1.0:
                raise ValueError("Invalid stage length range.")
            if not 0.0 < stage.min_width <= stage.max_width <= 1.0:
                raise ValueError("Invalid stage width range.")


@dataclass
class PlanResult:
    initial_canvas: Image.Image
    best_canvas: Image.Image
    final_canvas: Image.Image
    best_step: int
    initial_mse: float
    best_mse: float
    final_mse: float
    final_mae: float
    strokes: tuple[RGBStroke, ...]
    progress: tuple[dict[str, Any], ...]
    stage_canvases: dict[str, Image.Image]
    stage_stats: tuple[dict[str, Any], ...]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_target_set_sha256(entries: Sequence[tuple[str, str]]) -> str:
    payload = "\n".join(f"{filename}\0{digest}" for filename, digest in entries)
    return sha256(payload.encode("utf-8")).hexdigest()


def validate_fixed_targets(input_dir: Path) -> dict[str, Any]:
    """Fail closed unless the exact frozen five-target set is present."""

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Target directory not found: {input_dir}")

    expected_names = {spec.filename for spec in TARGET_SPECS}
    actual_names = {
        path.name
        for path in input_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"Target-set mismatch. Missing={missing}; extra={extra}")

    entries: list[tuple[str, str]] = []
    metadata: list[dict[str, Any]] = []

    for spec in TARGET_SPECS:
        path = input_dir / spec.filename
        digest = file_sha256(path)
        if digest != spec.sha256:
            raise ValueError(f"SHA-256 mismatch for {spec.filename}")

        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError(f"{spec.filename} is not RGB.")
            width, height = image.size

        if width < 96 or height < 96:
            raise ValueError(f"{spec.filename} is smaller than 96 pixels.")

        entries.append((spec.filename, digest))
        metadata.append(
            {
                "target_id": spec.target_id,
                "category": spec.category,
                "filename": spec.filename,
                "source_url": spec.source_url,
                "publication_note": spec.publication_note,
                "source_width": width,
                "source_height": height,
                "mode": "RGB",
                "sha256": digest,
            }
        )

    combined = _ordered_target_set_sha256(entries)
    if combined != TARGET_SET_SHA256:
        raise ValueError("Combined ordered target-set SHA-256 mismatch.")

    return {"target_set_sha256": combined, "targets": metadata}


def blank_rgb_canvas(
    size: int,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if size < 8:
        raise ValueError("RGB canvas size must be at least 8.")
    return Image.new("RGB", (size, size), color=background)


def _to_pixel(value: float, size: int) -> int:
    return int(round(max(0.0, min(1.0, float(value))) * (size - 1)))


def render_rgb_stroke(canvas: Image.Image, stroke: RGBStroke) -> Image.Image:
    """Render one RGB stroke without mutating the input canvas."""

    if canvas.mode != "RGB" or canvas.width != canvas.height:
        raise ValueError("Expected a square RGB canvas.")

    result = canvas.copy()
    draw = ImageDraw.Draw(result)
    draw.line(
        (
            _to_pixel(stroke.x0, result.width),
            _to_pixel(stroke.y0, result.height),
            _to_pixel(stroke.x1, result.width),
            _to_pixel(stroke.y1, result.height),
        ),
        fill=stroke.color,
        width=stroke.width,
    )
    return result


def stroke_mask(stroke: RGBStroke, size: int) -> np.ndarray:
    mask = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(mask)
    draw.line(
        (
            _to_pixel(stroke.x0, size),
            _to_pixel(stroke.y0, size),
            _to_pixel(stroke.x1, size),
            _to_pixel(stroke.y1, size),
        ),
        fill=255,
        width=stroke.width,
    )
    return np.asarray(mask, dtype=np.uint8) != 0


def resize_with_padding(
    image: Image.Image,
    size: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Resize without distortion and center the result on a white square."""

    if size < 8:
        raise ValueError("Output size must be at least 8.")

    source = ImageOps.exif_transpose(image).convert("RGB")
    source_width, source_height = source.size
    scale = min(size / source_width, size / source_height)
    resized_size = (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )
    resized = source.resize(resized_size, resample=Image.Resampling.LANCZOS)
    offset = ((size - resized.width) // 2, (size - resized.height) // 2)
    output = blank_rgb_canvas(size)
    output.paste(resized, offset)

    return output, {
        "source_size": [source_width, source_height],
        "resized_size": [resized.width, resized.height],
        "offset": [offset[0], offset[1]],
        "padding_color": [255, 255, 255],
        "aspect_ratio_preserved": True,
    }


def pixel_mse(left: Image.Image, right: Image.Image) -> float:
    if left.mode != "RGB" or right.mode != "RGB" or left.size != right.size:
        raise ValueError("pixel_mse expects matching RGB images.")
    left_values = np.asarray(left, dtype=np.float64) / 255.0
    right_values = np.asarray(right, dtype=np.float64) / 255.0
    difference = left_values - right_values
    return float(np.mean(difference * difference))


def pixel_mae(left: Image.Image, right: Image.Image) -> float:
    if left.mode != "RGB" or right.mode != "RGB" or left.size != right.size:
        raise ValueError("pixel_mae expects matching RGB images.")
    left_values = np.asarray(left, dtype=np.float64) / 255.0
    right_values = np.asarray(right, dtype=np.float64) / 255.0
    return float(np.mean(np.abs(left_values - right_values)))


def _sample_midpoint(
    error: np.ndarray,
    guided: bool,
    rng: np.random.Generator,
) -> tuple[float, float]:
    height, width = error.shape
    if guided and float(error.sum()) > 0.0:
        probabilities = error.reshape(-1).astype(np.float64)
        probabilities /= probabilities.sum()
        flat_index = int(rng.choice(probabilities.size, p=probabilities))
        y_index, x_index = divmod(flat_index, width)
    else:
        x_index = int(rng.integers(0, width))
        y_index = int(rng.integers(0, height))

    x = (x_index + float(rng.uniform(-0.45, 0.45))) / max(1, width - 1)
    y = (y_index + float(rng.uniform(-0.45, 0.45))) / max(1, height - 1)
    return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))


def _sample_geometry(
    midpoint: tuple[float, float],
    stage: StageConfig,
    size: int,
    rng: np.random.Generator,
) -> RGBStroke:
    for _ in range(100):
        angle = float(rng.uniform(0.0, 2.0 * pi))
        length = float(rng.uniform(stage.min_length, stage.max_length))
        half_dx = 0.5 * length * cos(angle)
        half_dy = 0.5 * length * sin(angle)
        x0 = max(0.0, min(1.0, midpoint[0] - half_dx))
        y0 = max(0.0, min(1.0, midpoint[1] - half_dy))
        x1 = max(0.0, min(1.0, midpoint[0] + half_dx))
        y1 = max(0.0, min(1.0, midpoint[1] + half_dy))
        if hypot(x1 - x0, y1 - y0) < stage.min_length * 0.25:
            continue
        width_normalized = float(rng.uniform(stage.min_width, stage.max_width))
        width = max(1, int(round(width_normalized * size)))
        return RGBStroke(x0, y0, x1, y1, width, (0, 0, 0))
    raise RuntimeError("Could not sample a valid RGB stroke geometry.")


def _fit_target_color(
    target_values: np.ndarray,
    geometry: RGBStroke,
) -> tuple[int, int, int]:
    mask = stroke_mask(geometry, target_values.shape[0])
    if not bool(mask.any()):
        raise RuntimeError("A proposed RGB stroke covered no pixels.")
    mean_color = np.rint(target_values[mask].mean(axis=0))
    clipped = np.clip(mean_color, 0, 255).astype(np.uint8)
    return (int(clipped[0]), int(clipped[1]), int(clipped[2]))


def propose_rgb_strokes(
    current: Image.Image,
    target: Image.Image,
    stage: StageConfig,
    rng: np.random.Generator,
    *,
    count: int,
    error_guided_fraction: float,
    max_attempts_per_candidate: int,
) -> tuple[RGBStroke, ...]:
    """Generate unique changing RGB candidates for one exact-pixel pool."""

    if current.mode != "RGB" or target.mode != "RGB":
        raise ValueError("RGB proposals require RGB images.")
    if current.size != target.size or current.width != current.height:
        raise ValueError("RGB proposal images must be matching squares.")
    if count < 1:
        raise ValueError("Candidate count must be positive.")

    current_values = np.asarray(current, dtype=np.uint8)
    target_values = np.asarray(target, dtype=np.uint8)
    difference = current_values.astype(np.float64) - target_values.astype(np.float64)
    error = np.mean(difference * difference, axis=2)
    guided_count = int(round(count * error_guided_fraction))

    candidates: list[RGBStroke] = []
    seen_outcomes: set[bytes] = set()
    attempts = 0
    max_attempts = count * max_attempts_per_candidate

    while len(candidates) < count and attempts < max_attempts:
        attempts += 1
        guided = len(candidates) < guided_count
        midpoint = _sample_midpoint(error, guided, rng)
        geometry = _sample_geometry(midpoint, stage, current.width, rng)
        color = _fit_target_color(target_values, geometry)
        stroke = RGBStroke(
            geometry.x0,
            geometry.y0,
            geometry.x1,
            geometry.y1,
            geometry.width,
            color,
        )
        rendered = render_rgb_stroke(current, stroke)
        rendered_values = np.asarray(rendered, dtype=np.uint8)
        if np.array_equal(rendered_values, current_values):
            continue
        signature = rendered_values.tobytes()
        if signature in seen_outcomes:
            continue
        seen_outcomes.add(signature)
        candidates.append(stroke)

    if len(candidates) != count:
        raise RuntimeError(
            f"Generated {len(candidates)} unique changing RGB candidates; required {count}."
        )
    return tuple(candidates)


def _pool_rng(
    seed: int,
    target_stream: int,
    stage_index: int,
    pool_index: int,
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([seed, target_stream, stage_index, pool_index])
    )


def plan_rgb_target(
    target: Image.Image,
    config: PainterConfig | None = None,
    *,
    target_stream: int = 0,
) -> PlanResult:
    """Run fixed exact-pixel coarse-to-fine planning for one target."""

    config = config or PainterConfig()
    config.validate()
    if target.mode != "RGB":
        raise ValueError("The planning target must be RGB.")
    if target.size != (config.planning_size, config.planning_size):
        raise ValueError("The planning target has the wrong dimensions.")
    if target_stream < 0:
        raise ValueError("target_stream must be non-negative.")

    current = blank_rgb_canvas(config.planning_size)
    initial = current.copy()
    initial_mse = pixel_mse(current, target)
    current_mse = initial_mse
    best_mse = initial_mse
    best_canvas = current.copy()
    best_step = 0

    strokes: list[RGBStroke] = []
    progress: list[dict[str, Any]] = []
    stage_canvases: dict[str, Image.Image] = {}
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
            candidates = propose_rgb_strokes(
                current,
                target,
                stage,
                rng,
                count=config.candidates_per_pool,
                error_guided_fraction=config.error_guided_fraction,
                max_attempts_per_candidate=config.max_attempts_per_candidate,
            )
            rendered = tuple(render_rgb_stroke(current, stroke) for stroke in candidates)
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
                    "x0": selected.x0,
                    "y0": selected.y0,
                    "x1": selected.x1,
                    "y1": selected.y1,
                    "width": selected.width,
                    "red": selected.color[0],
                    "green": selected.color[1],
                    "blue": selected.color[2],
                    "mse_before": mse_before,
                    "mse_after": current_mse,
                    "best_mse": best_mse,
                    "improvement": improvement,
                }
            )

        stage_canvases[stage.name] = current.copy()
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

    return PlanResult(
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
        stage_canvases=stage_canvases,
        stage_stats=tuple(stage_stats),
    )


def replay_rgb_strokes_high_resolution(
    strokes: Sequence[RGBStroke],
    *,
    planning_size: int,
    output_size: int,
    supersample: int,
    best_step: int,
    capture_steps: set[int],
    gif_stride: int,
) -> tuple[Image.Image, Image.Image, tuple[Image.Image, ...]]:
    """Replay normalized RGB strokes with supersampling."""

    render_size = output_size * supersample
    if render_size > 4096:
        raise ValueError("Replay render size must not exceed 4096.")
    if not 0 <= best_step <= len(strokes):
        raise ValueError("best_step lies outside the RGB stroke sequence.")

    working = blank_rgb_canvas(render_size)
    width_scale = render_size / planning_size

    def presentation_frame(image: Image.Image) -> Image.Image:
        if render_size == output_size:
            return image.copy()
        return image.resize((output_size, output_size), Image.Resampling.LANCZOS)

    initial = presentation_frame(working)
    frames: list[Image.Image] = [initial]
    best = initial.copy()

    for step, stroke in enumerate(strokes, start=1):
        scaled = RGBStroke(
            stroke.x0,
            stroke.y0,
            stroke.x1,
            stroke.y1,
            max(1, int(round(stroke.width * width_scale))),
            stroke.color,
        )
        working = render_rgb_stroke(working, scaled)
        if step == best_step:
            best = presentation_frame(working)
        if step % gif_stride == 0 or step in capture_steps or step == len(strokes):
            frames.append(presentation_frame(working))

    return best, presentation_frame(working), tuple(frames)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_progress_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = (
        "step", "stage", "stage_step", "candidate_pool",
        "selected_candidate_index", "x0", "y0", "x1", "y1", "width",
        "red", "green", "blue", "mse_before", "mse_after", "best_mse",
        "improvement",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_montage(
    items: Sequence[tuple[str, Image.Image]],
    *,
    cell_size: int = 256,
    columns: int,
) -> Image.Image:
    if not items or columns < 1:
        raise ValueError("A montage needs items and at least one column.")
    label_height = 24
    rows = (len(items) + columns - 1) // columns
    montage = Image.new(
        "RGB",
        (columns * cell_size, rows * (cell_size + label_height)),
        color=(32, 32, 32),
    )
    draw = ImageDraw.Draw(montage)
    for index, (label, image) in enumerate(items):
        row, column = divmod(index, columns)
        x = column * cell_size
        y = row * (cell_size + label_height)
        resized = image.convert("RGB").resize(
            (cell_size, cell_size),
            Image.Resampling.LANCZOS,
        )
        montage.paste(resized, (x, y))
        draw.rectangle(
            (x, y + cell_size, x + cell_size, y + cell_size + label_height),
            fill=(0, 0, 0),
        )
        draw.text((x + 5, y + cell_size + 5), label, fill=(255, 255, 255))
    return montage


def _save_gif(frames: Sequence[Image.Image], path: Path) -> None:
    if not frames:
        raise ValueError("At least one GIF frame is required.")
    frames[0].save(
        path,
        save_all=True,
        append_images=list(frames[1:]),
        duration=100,
        loop=0,
        optimize=False,
    )


def _hash_tree(directory: Path, excluded: set[str] | None = None) -> dict[str, str]:
    excluded = excluded or set()
    hashes: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in excluded:
            hashes[str(path.relative_to(directory))] = file_sha256(path)
    return hashes


def _stage_end_steps(stage_stats: Sequence[dict[str, Any]]) -> set[int]:
    steps: set[int] = set()
    cumulative = 0
    for item in stage_stats:
        cumulative += int(item["executed_steps"])
        if cumulative > 0:
            steps.add(cumulative)
    return steps


def run_one_target(
    spec: TargetSpec,
    *,
    input_dir: Path,
    output_dir: Path,
    config: PainterConfig,
    target_stream: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    source_path = input_dir / spec.filename

    with Image.open(source_path) as source:
        source.load()
        planning_target, preprocessing = resize_with_padding(source, config.planning_size)
        replay_target, replay_preprocessing = resize_with_padding(source, config.replay_size)

    planning_target.save(output_dir / "processed_target.png")
    replay_target.save(output_dir / "processed_target_512.png")
    result = plan_rgb_target(planning_target, config, target_stream=target_stream)
    result.initial_canvas.save(output_dir / "initial.png")
    result.best_canvas.save(output_dir / "best.png")
    result.final_canvas.save(output_dir / "final.png")

    for stage_name, canvas in result.stage_canvases.items():
        canvas.save(output_dir / f"stage_{stage_name}.png")

    best_512, final_512, gif_frames = replay_rgb_strokes_high_resolution(
        result.strokes,
        planning_size=config.planning_size,
        output_size=config.replay_size,
        supersample=config.supersample,
        best_step=result.best_step,
        capture_steps=_stage_end_steps(result.stage_stats),
        gif_stride=config.gif_stride,
    )
    best_512.save(output_dir / "best_512.png")
    final_512.save(output_dir / "final_512.png")
    _save_gif(gif_frames, output_dir / "painting_512.gif")

    stage_items: list[tuple[str, Image.Image]] = [("target", planning_target)]
    for stage in config.stages:
        stage_items.append((stage.name, result.stage_canvases[stage.name]))
    _make_montage(stage_items, columns=len(stage_items)).save(
        output_dir / "stage_montage.png"
    )
    _make_montage(
        (("target", replay_target), ("best", best_512), ("final", final_512)),
        columns=3,
    ).save(output_dir / "comparison_512.png")

    stroke_records = []
    for progress_row, stroke in zip(result.progress, result.strokes, strict=True):
        stroke_records.append(
            {
                "step": progress_row["step"],
                "stage": progress_row["stage"],
                "x0": stroke.x0,
                "y0": stroke.y0,
                "x1": stroke.x1,
                "y1": stroke.y1,
                "width": stroke.width,
                "color": list(stroke.color),
                "mse_before": progress_row["mse_before"],
                "mse_after": progress_row["mse_after"],
                "improvement": progress_row["improvement"],
            }
        )

    _write_json(output_dir / "strokes.json", stroke_records)
    _write_progress_csv(output_dir / "progress.csv", result.progress)
    _write_json(
        output_dir / "run_config.json",
        {
            "schema_version": 1,
            "target": asdict(spec),
            "target_stream": target_stream,
            "target_set_sha256": TARGET_SET_SHA256,
            "painter_config": asdict(config),
            "selection": "exact_rendered_rgb_target_pixel_mse",
            "training_performed": False,
            "learned_model_used": False,
        },
    )

    monotonic = all(
        float(row["mse_after"]) < float(row["mse_before"])
        and float(row["improvement"]) > config.min_improvement
        for row in result.progress
    )
    summary: dict[str, Any] = {
        "status": "rgb_coarse_to_fine_target_complete",
        "schema_version": 1,
        "target_id": spec.target_id,
        "category": spec.category,
        "source_filename": spec.filename,
        "source_url": spec.source_url,
        "source_sha256": spec.sha256,
        "publication_note": spec.publication_note,
        "target_set_sha256": TARGET_SET_SHA256,
        "preprocessing": preprocessing,
        "replay_preprocessing": replay_preprocessing,
        "planning_size": config.planning_size,
        "replay_size": config.replay_size,
        "seed": config.seed,
        "target_stream": target_stream,
        "initial_mse": result.initial_mse,
        "best_mse": result.best_mse,
        "final_mse": result.final_mse,
        "final_mae": result.final_mae,
        "best_step": result.best_step,
        "executed_strokes": len(result.strokes),
        "maximum_strokes": sum(stage.max_steps for stage in config.stages),
        "stage_stats": list(result.stage_stats),
        "every_executed_stroke_improved": monotonic,
        "best_not_worse_than_final": result.best_mse <= result.final_mse,
        "best_equals_final": np.array_equal(
            np.asarray(result.best_canvas), np.asarray(result.final_canvas)
        ),
        "high_resolution_best_mse": pixel_mse(best_512, replay_target),
        "high_resolution_final_mse": pixel_mse(final_512, replay_target),
        "high_resolution_final_mae": pixel_mae(final_512, replay_target),
        "gif_frame_count": len(gif_frames),
        "runtime_seconds": time.perf_counter() - started,
        "training_performed": False,
        "learned_model_used": False,
        "frozen_phase_b0_decision_changed": False,
    }
    summary["artifact_sha256"] = _hash_tree(
        output_dir,
        excluded={"summary.json", "summary.sha256"},
    )
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.sha256").write_text(
        file_sha256(output_dir / "summary.json") + "\n",
        encoding="utf-8",
    )
    return summary


def _save_aggregate_progress_plot(
    output_dir: Path,
    summaries: Sequence[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    for summary in summaries:
        csv_path = output_dir / summary["target_id"] / "progress.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        steps = [0] + [int(row["step"]) for row in rows]
        values = [float(summary["initial_mse"])] + [
            float(row["mse_after"]) for row in rows
        ]
        axis.plot(steps, values, label=summary["target_id"])
    axis.set_xlabel("Executed stroke")
    axis.set_ylabel("Target RGB pixel MSE")
    axis.set_title("Fixed RGB coarse-to-fine trajectories")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "aggregate_progress.png", dpi=160)
    plt.close(figure)


def run_fixed_experiment(
    *,
    input_dir: Path,
    output_dir: Path,
    config: PainterConfig | None = None,
) -> dict[str, Any]:
    """Run all five frozen targets through one fixed configuration."""

    config = config or PainterConfig()
    config.validate()
    validation = validate_fixed_targets(input_dir)
    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {output_dir}")
    if incomplete.exists():
        raise FileExistsError(f"Preserve existing incomplete output: {incomplete}")

    incomplete.parent.mkdir(parents=True, exist_ok=True)
    incomplete.mkdir()
    started = time.perf_counter()

    try:
        _write_json(incomplete / "fixed_target_manifest.json", validation)
        _write_json(
            incomplete / "run_config.json",
            {
                "schema_version": 1,
                "target_set_sha256": TARGET_SET_SHA256,
                "painter_config": asdict(config),
                "selection": "exact_rendered_rgb_target_pixel_mse",
                "scope": "bounded_qualitative_engineering",
                "training_performed": False,
                "learned_model_used": False,
            },
        )

        summaries = []
        for target_stream, spec in enumerate(TARGET_SPECS):
            summaries.append(
                run_one_target(
                    spec,
                    input_dir=input_dir,
                    output_dir=incomplete / spec.target_id,
                    config=config,
                    target_stream=target_stream,
                )
            )

        aggregate_items: list[tuple[str, Image.Image]] = []
        for spec in TARGET_SPECS:
            target_dir = incomplete / spec.target_id
            for suffix, filename in (
                ("target", "processed_target_512.png"),
                ("best", "best_512.png"),
                ("final", "final_512.png"),
            ):
                with Image.open(target_dir / filename) as image:
                    aggregate_items.append(
                        (f"{spec.target_id} {suffix}", image.convert("RGB").copy())
                    )

        _make_montage(aggregate_items, columns=3).save(
            incomplete / "five_target_montage.png"
        )
        _save_aggregate_progress_plot(incomplete, summaries)

        aggregate = {
            "status": "rgb_coarse_to_fine_fixed_set_complete",
            "schema_version": 1,
            "target_set_sha256": TARGET_SET_SHA256,
            "painter_config": asdict(config),
            "completed_target_count": len(summaries),
            "targets": summaries,
            "acceptance_checks": {
                "all_five_targets_completed": len(summaries) == 5,
                "all_executed_strokes_improved": all(
                    item["every_executed_stroke_improved"] for item in summaries
                ),
                "all_best_frames_not_worse_than_final": all(
                    item["best_not_worse_than_final"] for item in summaries
                ),
                "all_frozen_decisions_preserved": all(
                    not item["frozen_phase_b0_decision_changed"] for item in summaries
                ),
            },
            "runtime_seconds": time.perf_counter() - started,
            "training_performed": False,
            "learned_model_used": False,
            "frozen_phase_b0_decision_changed": False,
            "output_directory": str(output_dir),
        }
        aggregate["artifact_sha256"] = _hash_tree(
            incomplete,
            excluded={
                "aggregate_summary.json",
                "aggregate_summary.sha256",
                "failure.json",
            },
        )
        _write_json(incomplete / "aggregate_summary.json", aggregate)
        (incomplete / "aggregate_summary.sha256").write_text(
            file_sha256(incomplete / "aggregate_summary.json") + "\n",
            encoding="utf-8",
        )
        incomplete.rename(output_dir)
        return aggregate

    except Exception as error:
        _write_json(
            incomplete / "failure.json",
            {
                "status": "rgb_coarse_to_fine_fixed_set_incomplete",
                "error_type": type(error).__name__,
                "error": str(error),
                "training_performed": False,
            },
        )
        raise


def validate_only_report(input_dir: Path) -> dict[str, Any]:
    """Validate targets and run a side-effect-free synthetic smoke test."""

    validation = validate_fixed_targets(input_dir)
    config = PainterConfig()
    config.validate()
    smoke_config = PainterConfig(
        planning_size=24,
        replay_size=32,
        supersample=1,
        candidates_per_pool=8,
        error_guided_fraction=0.75,
        patience=3,
        min_improvement=1e-9,
        seed=73,
        gif_stride=1,
        max_attempts_per_candidate=100,
        stages=(StageConfig("smoke", 2, 0.20, 0.80, 0.15, 0.35),),
    )
    smoke_target = Image.new("RGB", (24, 24), color=(64, 96, 128))
    first = plan_rgb_target(smoke_target, smoke_config, target_stream=0)
    second = plan_rgb_target(smoke_target, smoke_config, target_stream=0)
    deterministic = first.progress == second.progress and np.array_equal(
        np.asarray(first.final_canvas), np.asarray(second.final_canvas)
    )
    monotonic = bool(first.progress) and all(
        row["mse_after"] < row["mse_before"]
        and row["improvement"] > smoke_config.min_improvement
        for row in first.progress
    )
    if not deterministic:
        raise RuntimeError("Synthetic RGB smoke test was not deterministic.")
    if not monotonic:
        raise RuntimeError("Synthetic RGB smoke test was not monotonic.")

    import matplotlib
    import PIL

    return {
        "status": "rgb_coarse_to_fine_runner_valid_no_outputs",
        "target_validation": validation,
        "fixed_config": asdict(config),
        "synthetic_smoke": {
            "executed_strokes": len(first.strokes),
            "initial_mse": first.initial_mse,
            "final_mse": first.final_mse,
            "deterministic": deterministic,
            "monotonic": monotonic,
        },
        "dependencies": {
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "output_side_effects": False,
        "training_performed": False,
        "learned_model_used": False,
        "frozen_phase_b0_decision_changed": False,
    }
