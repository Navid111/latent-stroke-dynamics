"""User-facing qualitative image-to-strokes painting command."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .learned_pixel_planner import (
    LearnedPlanningRun,
    PixelCheckpointMetadata,
    load_pixel_checkpoint,
    run_learned_planner,
    state_dict_sha256,
)
from .planning import (
    PlanningRun,
    ProposalConfig,
    load_target,
    pixel_mae,
    pixel_mse,
    run_planner,
)


PaintingMethod = Literal["random", "exact", "learned"]
RunLike = PlanningRun | LearnedPlanningRun
DEFAULT_CHECKPOINT = Path("checkpoints/stage3-pixel-mlp-seed11.pt")
FROZEN_CHECKPOINT_SHA256 = (
    "e32f3612f7a184e4e9b58f95a987551bd25cdb17ff1bf2b6be40fcf5781ea472"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paint a center-cropped 64x64 grayscale approximation of an input "
            "image with sequential straight-line strokes."
        )
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("random", "exact", "learned"),
        default="learned",
    )
    parser.add_argument("--strokes", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    parser.add_argument("--gif-scale", type=int, default=6)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _validate_request(
    target_path: Path,
    output_dir: Path,
    method: str,
    strokes: int,
    candidates: int,
    seed: int,
    prediction_batch_size: int,
    gif_scale: int,
    checkpoint: Path,
) -> Path:
    if method not in ("random", "exact", "learned"):
        raise ValueError("method must be 'random', 'exact', or 'learned'.")
    if not target_path.is_file():
        raise FileNotFoundError(f"Target image does not exist: {target_path}")
    if strokes < 1:
        raise ValueError("strokes must be positive.")
    if candidates < 2:
        raise ValueError("candidates must be at least 2.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if prediction_batch_size < 1 or gif_scale < 1:
        raise ValueError("prediction batch size and GIF scale must be positive.")
    if not output_dir.name:
        raise ValueError("output_dir must name a directory.")

    incomplete_dir = output_dir.with_name(f"{output_dir.name}.incomplete")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    if incomplete_dir.exists():
        raise FileExistsError(
            "A previous incomplete painting directory exists. Preserve it for "
            f"diagnosis or move it explicitly: {incomplete_dir}"
        )
    if method == "learned" and not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing frozen Stage 3 checkpoint: {checkpoint}. "
            "Do not retrain it; restore the validated local checkpoint."
        )
    return incomplete_dir


def _validate_checkpoint(
    metadata: PixelCheckpointMetadata,
    digest: str,
) -> None:
    expected = {
        "canvas_size": 64,
        "hidden_dim": 64,
        "parameter_count": 833,
        "model_seed": 11,
        "train_seed": 20260824,
        "validation_seed": 20260825,
        "train_samples": 1000,
        "validation_samples": 200,
        "test_rows_used_for_training_or_selection": False,
    }
    for name, expected_value in expected.items():
        observed = getattr(metadata, name)
        if observed != expected_value:
            raise RuntimeError(
                f"Frozen checkpoint metadata mismatch for {name}: "
                f"expected {expected_value!r}, received {observed!r}."
            )
    if digest != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Frozen Stage 3 checkpoint digest mismatch: "
            f"expected {FROZEN_CHECKPOINT_SHA256}, received {digest}."
        )


def _method_name(run: RunLike) -> str:
    return "learned" if isinstance(run, LearnedPlanningRun) else run.method


def _step_rows(run: RunLike) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in run.steps:
        row: dict[str, Any] = {"method": _method_name(run), **asdict(record)}
        stroke = row.pop("stroke")
        if not isinstance(stroke, dict):
            raise RuntimeError("Serialized stroke record is invalid.")
        for key, value in stroke.items():
            row[f"stroke_{key}"] = value
        rows.append(row)
    return rows


def _summary(run: RunLike, elapsed_seconds: float) -> dict[str, Any]:
    initial_mse = pixel_mse(run.initial_canvas, run.target)
    final_mse = pixel_mse(run.final_canvas, run.target)
    if initial_mse <= 1e-12:
        raise ValueError("The processed target is effectively blank white; no painting is needed.")
    result: dict[str, Any] = {
        "method": _method_name(run),
        "strokes": len(run.steps),
        "candidates_per_step": run.steps[0].candidate_count,
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "final_mae": pixel_mae(run.final_canvas, run.target),
        "relative_mse_improvement": (initial_mse - final_mse) / initial_mse,
        "improved_steps": sum(int(record.improved) for record in run.steps),
        "elapsed_seconds": float(elapsed_seconds),
        "exact_top1_rate": None,
        "exact_top5_rate": None,
        "mean_exact_rank": None,
        "mean_exact_regret": None,
        "max_exact_regret": None,
    }
    if isinstance(run, LearnedPlanningRun):
        result.update(
            {
                "exact_top1_rate": float(
                    np.mean([record.exact_top1 for record in run.steps])
                ),
                "exact_top5_rate": float(
                    np.mean([record.exact_top5 for record in run.steps])
                ),
                "mean_exact_rank": float(
                    np.mean([record.exact_selected_rank for record in run.steps])
                ),
                "mean_exact_regret": float(
                    np.mean([record.exact_regret for record in run.steps])
                ),
                "max_exact_regret": float(
                    np.max([record.exact_regret for record in run.steps])
                ),
            }
        )
    for name, value in result.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise RuntimeError(f"Painting summary contains non-finite metric: {name}")
    return result


def _save_frames(frames: tuple[Image.Image, ...], output_dir: Path) -> None:
    if not frames:
        raise ValueError("Painting output requires captured frames.")
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=False)
    digits = max(4, len(str(len(frames) - 1)))
    for index, frame in enumerate(frames):
        frame.save(frame_dir / f"frame_{index:0{digits}d}.png")


def _save_gif(
    frames: tuple[Image.Image, ...],
    output_path: Path,
    scale: int,
) -> None:
    scaled = [
        frame.resize(
            (frame.width * scale, frame.height * scale),
            resample=Image.Resampling.NEAREST,
        )
        for frame in frames
    ]
    scaled[0].save(
        output_path,
        save_all=True,
        append_images=scaled[1:],
        duration=140,
        loop=0,
        optimize=False,
    )


def _save_progress_plot(run: RunLike, output_path: Path) -> None:
    values = [pixel_mse(run.initial_canvas, run.target)] + [
        record.mse_after for record in run.steps
    ]
    best_step = int(np.argmin(values))
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(range(len(values)), values, color="tab:green", linewidth=2)
    axis.scatter(
        [best_step],
        [values[best_step]],
        color="black",
        s=24,
        zorder=3,
        label=f"best step {best_step}",
    )
    axis.set_xlabel("Executed strokes")
    axis.set_ylabel("Target pixel MSE")
    axis.set_title(f"{_method_name(run).title()} painting progress")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_comparison_plot(run: RunLike, output_path: Path) -> None:
    target_values = np.asarray(run.target, dtype=np.int16)
    final_values = np.asarray(run.final_canvas, dtype=np.int16)
    absolute_error = np.abs(final_values - target_values)
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    axes[0].imshow(target_values, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Processed target")
    axes[1].imshow(final_values, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"Final painting\nMSE {pixel_mse(run.final_canvas, run.target):.5f}"
    )
    axes[2].imshow(absolute_error, cmap="magma", vmin=0, vmax=255)
    axes[2].set_title("Absolute error (fixed 0–255)")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_artifacts(
    run: RunLike,
    output_dir: Path,
    summary: dict[str, Any],
    config: dict[str, Any],
    gif_scale: int,
) -> None:
    run.target.save(output_dir / "processed_target.png")
    run.initial_canvas.save(output_dir / "initial_canvas.png")
    run.final_canvas.save(output_dir / "final_painting.png")

    rows = _step_rows(run)
    progress = pd.DataFrame(rows)
    progress.to_csv(output_dir / "progress.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    strokes = [
        {
            "step": int(row["step"]),
            "selected_index": int(row["selected_index"]),
            "x0": float(row["stroke_x0"]),
            "y0": float(row["stroke_y0"]),
            "x1": float(row["stroke_x1"]),
            "y1": float(row["stroke_y1"]),
            "width": int(row["stroke_width"]),
            "value": int(row["stroke_value"]),
        }
        for row in rows
    ]
    (output_dir / "strokes.json").write_text(
        json.dumps(strokes, indent=2),
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    _save_frames(run.frames, output_dir)
    _save_gif(run.frames, output_dir / "painting.gif", scale=gif_scale)
    _save_progress_plot(run, output_dir / "progress.png")
    _save_comparison_plot(run, output_dir / "comparison.png")


def paint_target(
    target_path: str | Path,
    output_dir: str | Path,
    method: PaintingMethod = "learned",
    strokes: int = 100,
    candidates: int = 128,
    seed: int = 20261001,
    prediction_batch_size: int = 32,
    gif_scale: int = 6,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
) -> tuple[Path, dict[str, Any]]:
    """Run one qualitative painting and atomically publish all artifacts."""

    target_path = Path(target_path)
    output_dir = Path(output_dir)
    checkpoint = Path(checkpoint)
    incomplete_dir = _validate_request(
        target_path=target_path,
        output_dir=output_dir,
        method=method,
        strokes=strokes,
        candidates=candidates,
        seed=seed,
        prediction_batch_size=prediction_batch_size,
        gif_scale=gif_scale,
        checkpoint=checkpoint,
    )

    target = load_target(target_path, size=64)
    initial_target_mse = pixel_mse(Image.new("L", target.size, color=255), target)
    if initial_target_mse <= 1e-12:
        raise ValueError("The processed target is effectively blank white; no painting is needed.")

    model = None
    metadata: PixelCheckpointMetadata | None = None
    checkpoint_digest: str | None = None
    if method == "learned":
        model, metadata = load_pixel_checkpoint(checkpoint, device="cpu")
        checkpoint_digest = state_dict_sha256(model)
        _validate_checkpoint(metadata, checkpoint_digest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir(parents=False, exist_ok=False)
    proposal_config = ProposalConfig(count=candidates)
    started = time.perf_counter()
    if method == "learned":
        if model is None:
            raise RuntimeError("Learned painting requires a loaded model.")
        run: RunLike = run_learned_planner(
            target,
            model,
            steps=strokes,
            seed=seed,
            proposal_config=proposal_config,
            prediction_batch_size=prediction_batch_size,
            device="cpu",
            capture_frames=True,
        )
    else:
        run = run_planner(
            target,
            method,
            steps=strokes,
            seed=seed,
            proposal_config=proposal_config,
            capture_frames=True,
        )
    elapsed_seconds = time.perf_counter() - started
    summary = _summary(run, elapsed_seconds)
    config: dict[str, Any] = {
        "qualitative_demo": True,
        "controlled_stage3_result_unchanged": True,
        "formal_gate_or_model_selection": False,
        "retraining_performed": False,
        "input_target": str(target_path.resolve()),
        "target_processing": {
            "exif_transpose": True,
            "center_crop_square": True,
            "grayscale": True,
            "canvas_size": 64,
            "resize": "Pillow LANCZOS",
        },
        "method": method,
        "strokes": strokes,
        "candidates_per_step": candidates,
        "planner_seed": seed,
        "prediction_batch_size": prediction_batch_size,
        "gif_scale": gif_scale,
        "proposal": asdict(proposal_config),
        "checkpoint_path": str(checkpoint) if method == "learned" else None,
        "checkpoint_state_dict_sha256": checkpoint_digest,
        "checkpoint_metadata": metadata.to_dict() if metadata is not None else None,
    }
    _save_artifacts(
        run,
        output_dir=incomplete_dir,
        summary=summary,
        config=config,
        gif_scale=gif_scale,
    )
    incomplete_dir.replace(output_dir)
    return output_dir, summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir, summary = paint_target(
        target_path=args.target,
        output_dir=args.output_dir,
        method=args.method,
        strokes=args.strokes,
        candidates=args.candidates,
        seed=args.seed,
        prediction_batch_size=args.prediction_batch_size,
        gif_scale=args.gif_scale,
        checkpoint=args.checkpoint,
    )
    print("\nImage-to-strokes painting complete\n")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved qualitative artifacts to: {output_dir.resolve()}")
    print("The frozen controlled Stage 3 result remains unchanged.")


if __name__ == "__main__":
    main()
