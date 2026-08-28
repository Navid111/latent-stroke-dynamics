"""Qualitative high-resolution replay of an existing 64x64 stroke sequence."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .renderer import Stroke, blank_canvas, render_stroke


REQUIRED_SOURCE_FILES = (
    "processed_target.png",
    "strokes.json",
    "summary.json",
    "run_config.json",
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_strokes(path: Path) -> tuple[Stroke, ...]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError("strokes.json must contain a list.")
    strokes: list[Stroke] = []
    required = {"step", "x0", "y0", "x1", "y1", "width", "value"}
    for expected_step, item in enumerate(payload, start=1):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"Invalid stroke record at index {expected_step - 1}.")
        if item["step"] != expected_step:
            raise ValueError("Stroke steps must be contiguous and one-indexed.")
        strokes.append(
            Stroke(
                x0=float(item["x0"]),
                y0=float(item["y0"]),
                x1=float(item["x1"]),
                y1=float(item["y1"]),
                width=int(item["width"]),
                value=int(item["value"]),
            )
        )
    if not strokes:
        raise ValueError("At least one source stroke is required.")
    return tuple(strokes)


def _best_step_from_progress(path: Path, stroke_count: int) -> int:
    if not path.is_file():
        raise ValueError(
            "Source summary has no best_step and progress.csv is unavailable."
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != stroke_count:
        raise ValueError("progress.csv length does not match the stroke sequence.")

    best_step = 0
    best_mse: float | None = None
    for expected_step, row in enumerate(rows, start=1):
        try:
            step = int(row["step"])
            mse_before = float(row["mse_before"])
            mse_after = float(row["mse_after"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("progress.csv has invalid best-step fields.") from exc
        if step != expected_step or not isfinite(mse_before) or not isfinite(mse_after):
            raise ValueError("progress.csv has invalid best-step values.")
        if best_mse is None:
            best_mse = mse_before
        if mse_after < best_mse:
            best_mse = mse_after
            best_step = step
    return best_step


def _resolve_best_step(
    summary: dict[str, Any],
    source: Path,
    stroke_count: int,
) -> tuple[int, str]:
    raw = summary.get("best_step")
    if isinstance(raw, bool):
        raise ValueError("Source summary has an invalid best_step.")
    if isinstance(raw, int):
        best_step = raw
        source_name = "summary.json"
    elif isinstance(raw, float) and isfinite(raw) and raw.is_integer():
        best_step = int(raw)
        source_name = "summary.json"
    elif raw is None:
        best_step = _best_step_from_progress(source / "progress.csv", stroke_count)
        source_name = "progress.csv"
    else:
        raise ValueError("Source summary has an invalid best_step.")
    if not 0 <= best_step <= stroke_count:
        raise ValueError("Source best_step lies outside the stroke sequence.")
    return best_step, source_name


def replay_strokes_high_resolution(
    strokes: Sequence[Stroke],
    *,
    output_size: int = 512,
    planning_size: int = 64,
    supersample: int = 2,
) -> tuple[Image.Image, ...]:
    """Replay fixed normalized strokes at a larger antialiased resolution."""

    if planning_size != 64:
        raise ValueError("The frozen qualitative painter plans at 64x64.")
    if output_size < planning_size or output_size > 2048:
        raise ValueError("output_size must lie between 64 and 2048.")
    if supersample < 1 or supersample > 4:
        raise ValueError("supersample must lie between 1 and 4.")
    if not strokes:
        raise ValueError("At least one stroke is required for replay.")

    render_size = output_size * supersample
    if render_size > 4096:
        raise ValueError("output_size times supersample must not exceed 4096.")
    width_scale = render_size / planning_size
    working = blank_canvas(render_size)

    def presentation_frame(image: Image.Image) -> Image.Image:
        if render_size == output_size:
            return image.copy()
        return image.resize(
            (output_size, output_size),
            resample=Image.Resampling.LANCZOS,
        )

    frames = [presentation_frame(working)]
    for stroke in strokes:
        scaled = Stroke(
            x0=stroke.x0,
            y0=stroke.y0,
            x1=stroke.x1,
            y1=stroke.y1,
            width=max(1, int(round(stroke.width * width_scale))),
            value=stroke.value,
        )
        working = render_stroke(working, scaled)
        frames.append(presentation_frame(working))
    return tuple(frames)


def _save_gif(frames: Sequence[Image.Image], path: Path) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=list(frames[1:]),
        duration=140,
        loop=0,
        optimize=False,
    )


def _validate_directories(source: Path, output: Path) -> Path:
    if not source.is_dir():
        raise FileNotFoundError(f"Completed painting directory not found: {source}")
    if source.name.endswith(".incomplete"):
        raise ValueError("Refusing to replay an incomplete painting output.")
    for filename in REQUIRED_SOURCE_FILES:
        if not (source / filename).is_file():
            raise FileNotFoundError(f"Missing source painting artifact: {filename}")
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite replay output: {output}")
    if incomplete.exists():
        raise FileExistsError(f"Preserve existing incomplete replay: {incomplete}")
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    try:
        output_resolved.relative_to(source_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("Replay output must be outside the source painting directory.")
    return incomplete


def replay_existing_painting(
    painting_dir: str | Path,
    output_dir: str | Path,
    *,
    output_size: int = 512,
    supersample: int = 2,
) -> tuple[Path, dict[str, Any]]:
    """Create presentation-resolution artifacts without changing stroke decisions."""

    source = Path(painting_dir).expanduser()
    output = Path(output_dir).expanduser()
    incomplete = _validate_directories(source, output)
    source_paths = {name: source / name for name in REQUIRED_SOURCE_FILES}
    progress_path = source / "progress.csv"
    if progress_path.is_file():
        source_paths["progress.csv"] = progress_path
    hashes_before = {
        name: _file_sha256(path) for name, path in source_paths.items()
    }

    run_config = _load_json(source / "run_config.json")
    summary = _load_json(source / "summary.json")
    if not isinstance(run_config, dict) or not isinstance(summary, dict):
        raise ValueError("Source configuration and summary must be JSON objects.")
    processing = run_config.get("target_processing")
    if not isinstance(processing, dict) or processing.get("canvas_size") != 64:
        raise ValueError("Source painting was not planned under the 64x64 qualitative protocol.")
    strokes = _load_strokes(source / "strokes.json")
    best_step, best_step_source = _resolve_best_step(summary, source, len(strokes))

    with Image.open(source / "processed_target.png") as image:
        target = image.convert("L")
    if target.size != (64, 64):
        raise ValueError("Source processed target must be 64x64.")

    frames = replay_strokes_high_resolution(
        strokes,
        output_size=output_size,
        planning_size=64,
        supersample=supersample,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    incomplete.mkdir(parents=False, exist_ok=False)
    target.resize(
        (output_size, output_size),
        resample=Image.Resampling.LANCZOS,
    ).save(incomplete / "reference.png")
    frames[0].save(incomplete / "initial.png")
    frames[best_step].save(incomplete / "best.png")
    frames[-1].save(incomplete / "final.png")
    _save_gif(frames, incomplete / "painting.gif")

    hashes_after = {
        name: _file_sha256(path) for name, path in source_paths.items()
    }
    if hashes_before != hashes_after:
        raise RuntimeError("A source painting artifact changed during replay.")
    replay_config: dict[str, Any] = {
        "status": "qualitative_high_resolution_replay_complete",
        "source_painting_directory": str(source.resolve()),
        "source_artifact_sha256": hashes_before,
        "source_artifacts_unchanged": True,
        "planning_size": 64,
        "output_size": output_size,
        "supersample": supersample,
        "internal_render_size": output_size * supersample,
        "stroke_count": len(strokes),
        "source_best_step": best_step,
        "source_best_step_source": best_step_source,
        "stroke_sequence_changed": False,
        "candidate_selection_repeated": False,
        "evaluation_metrics_recomputed": False,
        "models_loaded": False,
        "models_trained": False,
        "controlled_results_changed": False,
        "qualitative_presentation_artifact_only": True,
    }
    (incomplete / "replay_config.json").write_text(
        json.dumps(replay_config, indent=2),
        encoding="utf-8",
    )
    incomplete.replace(output)
    return output, replay_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--painting-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--supersample", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output, config = replay_existing_painting(
        args.painting_dir,
        args.output_dir,
        output_size=args.size,
        supersample=args.supersample,
    )
    print("\nHigh-resolution qualitative replay complete\n")
    print(json.dumps(config, indent=2))
    print(f"\nSaved replay to: {output.resolve()}")
    print("The original stroke sequence and controlled results remain unchanged.")


if __name__ == "__main__":
    main()
