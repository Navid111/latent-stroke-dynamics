"""Stage 3 smoke comparing random, exact, and learned pixel planners."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from latent_stroke_dynamics.learned_pixel_planner import (
    LearnedPlanningRun,
    PixelCheckpointMetadata,
    load_pixel_checkpoint,
    run_learned_planner,
    state_dict_sha256,
)
from latent_stroke_dynamics.planning import (
    PlanningRun,
    ProposalConfig,
    pixel_mae,
    pixel_mse,
    run_planner,
)
from latent_stroke_dynamics.renderer import random_base_canvas


RunLike = PlanningRun | LearnedPlanningRun


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/stage3-pixel-mlp-seed11.pt"),
    )
    parser.add_argument("--canvas-size", type=int, default=64)
    parser.add_argument("--target-strokes", type=int, default=20)
    parser.add_argument("--target-seed", type=int, default=20260901)
    parser.add_argument("--planner-seed", type=int, default=20260822)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    parser.add_argument("--gif-scale", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage3-all-methods-smoke-1"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing demonstration checkpoint: {args.checkpoint}. "
            "Run experiments/05_train_pixel_planner_checkpoint.py first."
        )
    if args.canvas_size < 8 or args.target_strokes < 1:
        raise ValueError("Canvas size and target strokes are invalid.")
    if args.target_seed < 0 or args.planner_seed < 0:
        raise ValueError("Seeds must be non-negative.")
    if args.steps < 1 or args.candidates < 2:
        raise ValueError("--steps must be positive and --candidates at least 2.")
    if args.prediction_batch_size < 1 or args.gif_scale < 1:
        raise ValueError("Batch size and GIF scale must be positive.")


def validate_checkpoint_scope(
    metadata: PixelCheckpointMetadata,
    canvas_size: int,
) -> None:
    expected = {
        "canvas_size": canvas_size,
        "hidden_dim": 64,
        "parameter_count": 833,
        "model_seed": 11,
        "train_seed": 20260824,
        "validation_seed": 20260825,
        "train_samples": 1000,
        "validation_samples": 200,
        "crowding": (0, 5, 15),
        "epochs": 30,
        "patience": 6,
        "batch_size": 16,
    }
    for name, value in expected.items():
        if getattr(metadata, name) != value:
            raise RuntimeError(
                f"Checkpoint metadata mismatch for {name}: "
                f"expected {value!r}, received {getattr(metadata, name)!r}."
            )
    if not np.isclose(metadata.learning_rate, 0.001):
        raise RuntimeError("Checkpoint learning rate does not match Stage 3.")
    if not np.isclose(metadata.weight_decay, 0.0001):
        raise RuntimeError("Checkpoint weight decay does not match Stage 3.")
    if metadata.test_rows_used_for_training_or_selection:
        raise RuntimeError("Checkpoint metadata reports forbidden test-row use.")


def method_name(run: RunLike) -> str:
    return "learned" if isinstance(run, LearnedPlanningRun) else run.method


def save_gif(frames: tuple[Image.Image, ...], output_path: Path, scale: int) -> None:
    if not frames:
        raise ValueError("GIF output requires captured frames.")
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
        duration=180,
        loop=0,
        optimize=False,
    )


def step_rows(run: RunLike) -> list[dict[str, Any]]:
    name = method_name(run)
    rows: list[dict[str, Any]] = []
    for record in run.steps:
        row = {"method": name, **asdict(record)}
        stroke = row.pop("stroke")
        for key, value in stroke.items():
            row[f"stroke_{key}"] = value
        rows.append(row)
    return rows


def save_run(run: RunLike, output_dir: Path, gif_scale: int) -> None:
    name = method_name(run)
    method_dir = output_dir / name
    method_dir.mkdir(parents=True, exist_ok=True)
    run.initial_canvas.save(method_dir / "initial_canvas.png")
    run.final_canvas.save(method_dir / "final_canvas.png")
    rows = step_rows(run)
    pd.DataFrame(rows).to_csv(method_dir / "progress.csv", index=False)
    stroke_payload = [
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
    (method_dir / "strokes.json").write_text(
        json.dumps(stroke_payload, indent=2),
        encoding="utf-8",
    )
    save_gif(run.frames, method_dir / "painting.gif", scale=gif_scale)


def summary_row(run: RunLike, elapsed_seconds: float) -> dict[str, Any]:
    name = method_name(run)
    initial_mse = pixel_mse(run.initial_canvas, run.target)
    final_mse = pixel_mse(run.final_canvas, run.target)
    row: dict[str, Any] = {
        "method": name,
        "steps": len(run.steps),
        "candidates_per_step": run.steps[0].candidate_count,
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "final_mae": pixel_mae(run.final_canvas, run.target),
        "relative_mse_improvement": (initial_mse - final_mse) / initial_mse,
        "improved_steps": sum(int(record.improved) for record in run.steps),
        "elapsed_seconds": elapsed_seconds,
        "exact_top1_rate": None,
        "exact_top5_rate": None,
        "mean_exact_rank": None,
        "mean_exact_regret": None,
        "max_exact_regret": None,
    }
    if isinstance(run, LearnedPlanningRun):
        row.update(
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
    return row


def validate_summary_metrics(summaries: pd.DataFrame) -> None:
    """Require finite metrics while allowing non-applicable baseline diagnostics."""

    shared_columns = [
        "steps",
        "candidates_per_step",
        "initial_mse",
        "final_mse",
        "final_mae",
        "relative_mse_improvement",
        "improved_steps",
        "elapsed_seconds",
    ]
    shared = summaries[shared_columns].to_numpy(dtype=float)
    if not bool(np.isfinite(shared).all()):
        raise RuntimeError("All-method smoke contains a non-finite shared metric.")

    learned_columns = [
        "exact_top1_rate",
        "exact_top5_rate",
        "mean_exact_rank",
        "mean_exact_regret",
        "max_exact_regret",
    ]
    learned = summaries.loc[
        summaries["method"] == "learned",
        learned_columns,
    ].to_numpy(dtype=float)
    if learned.shape != (1, len(learned_columns)) or not bool(
        np.isfinite(learned).all()
    ):
        raise RuntimeError("Learned-planner diagnostics contain a non-finite metric.")


def save_progress_plot(runs: tuple[RunLike, ...], output_path: Path) -> None:
    colors = {"random": "tab:blue", "exact": "tab:orange", "learned": "tab:green"}
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for run in runs:
        name = method_name(run)
        values = [pixel_mse(run.initial_canvas, run.target)] + [
            record.mse_after for record in run.steps
        ]
        axis.plot(
            range(len(values)),
            values,
            marker="o",
            markersize=3,
            label=name,
            color=colors[name],
        )
    axis.set_xlabel("Executed strokes")
    axis.set_ylabel("Target pixel MSE")
    axis.set_title("Stage 3 all-method smoke: target error by step")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_comparison_plot(
    target: Image.Image,
    runs: tuple[RunLike, ...],
    output_path: Path,
) -> None:
    target_values = np.asarray(target, dtype=np.int16)
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes[0, 0].imshow(target_values, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Synthetic target")
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.5,
        0.5,
        "Absolute-error panels\nuse fixed scale 0–255",
        ha="center",
        va="center",
        fontsize=11,
    )
    for column, run in enumerate(runs, start=1):
        name = method_name(run)
        final_values = np.asarray(run.final_canvas, dtype=np.int16)
        error = np.abs(final_values - target_values)
        axes[0, column].imshow(final_values, cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(
            f"{name.title()}\nMSE {pixel_mse(run.final_canvas, target):.5f}"
        )
        axes[1, column].imshow(error, cmap="magma", vmin=0, vmax=255)
        axes[1, column].set_title(f"{name.title()} absolute error")
    for axis in axes.ravel():
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def timed_run(function: Any, *args: Any, **kwargs: Any) -> tuple[RunLike, float]:
    started = time.perf_counter()
    run = function(*args, **kwargs)
    return run, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, metadata = load_pixel_checkpoint(args.checkpoint, device="cpu")
    validate_checkpoint_scope(metadata, args.canvas_size)
    checkpoint_digest = state_dict_sha256(model)

    target = random_base_canvas(
        size=args.canvas_size,
        prior_strokes=args.target_strokes,
        rng=np.random.default_rng(args.target_seed),
    )
    target.save(args.output_dir / "target.png")
    proposal_config = ProposalConfig(count=args.candidates)

    random_run, random_elapsed = timed_run(
        run_planner,
        target,
        "random",
        steps=args.steps,
        seed=args.planner_seed,
        proposal_config=proposal_config,
        capture_frames=True,
    )
    exact_run, exact_elapsed = timed_run(
        run_planner,
        target,
        "exact",
        steps=args.steps,
        seed=args.planner_seed,
        proposal_config=proposal_config,
        capture_frames=True,
    )
    learned_run, learned_elapsed = timed_run(
        run_learned_planner,
        target,
        model,
        steps=args.steps,
        seed=args.planner_seed,
        proposal_config=proposal_config,
        prediction_batch_size=args.prediction_batch_size,
        device="cpu",
        capture_frames=True,
    )
    learned_replay = run_learned_planner(
        target,
        model,
        steps=args.steps,
        seed=args.planner_seed,
        proposal_config=proposal_config,
        prediction_batch_size=args.prediction_batch_size,
        device="cpu",
        capture_frames=False,
    )
    deterministic_learned_replay = bool(
        learned_run.steps == learned_replay.steps
        and np.array_equal(
            np.asarray(learned_run.final_canvas),
            np.asarray(learned_replay.final_canvas),
        )
    )
    if not deterministic_learned_replay:
        raise RuntimeError("Learned-planner deterministic replay failed.")

    runs: tuple[RunLike, ...] = (random_run, exact_run, learned_run)
    for run in runs:
        save_run(run, args.output_dir, gif_scale=args.gif_scale)
    summaries = pd.DataFrame(
        [
            summary_row(random_run, random_elapsed),
            summary_row(exact_run, exact_elapsed),
            summary_row(learned_run, learned_elapsed),
        ]
    )
    validate_summary_metrics(summaries)
    summaries.to_csv(args.output_dir / "summary.csv", index=False)
    learned_diagnostics = pd.DataFrame(step_rows(learned_run))
    learned_diagnostics.to_csv(
        args.output_dir / "learned_step_diagnostics.csv",
        index=False,
    )
    save_progress_plot(runs, args.output_dir / "progress_curves.png")
    save_comparison_plot(target, runs, args.output_dir / "final_comparison.png")

    summary_by_method = summaries.set_index("method")
    random_final = float(summary_by_method.loc["random", "final_mse"])
    exact_final = float(summary_by_method.loc["exact", "final_mse"])
    learned_final = float(summary_by_method.loc["learned", "final_mse"])
    config = {
        "diagnostic_only": True,
        "formal_stage3_decision_made": False,
        "canvas_size": args.canvas_size,
        "target_strokes": args.target_strokes,
        "target_seed": args.target_seed,
        "planner_seed": args.planner_seed,
        "steps": args.steps,
        "candidates_per_step": args.candidates,
        "prediction_batch_size": args.prediction_batch_size,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_state_dict_sha256": checkpoint_digest,
        "checkpoint_metadata": metadata.to_dict(),
        "deterministic_learned_replay_passed": deterministic_learned_replay,
        "learned_final_mse_no_worse_than_random": learned_final <= random_final,
        "learned_final_mse_ratio_to_exact": learned_final / max(exact_final, 1e-12),
        "formal_paired_control_result_unchanged": True,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nStage 3 random/exact/learned engineering smoke\n")
    print(summaries.to_string(index=False))
    print("\nLearned-planner diagnostics\n")
    print(
        learned_diagnostics[
            [
                "step",
                "mse_before",
                "mse_after",
                "exact_selected_rank",
                "exact_top1",
                "exact_top5",
                "exact_regret",
                "improved",
            ]
        ].to_string(index=False)
    )
    print(f"\nDeterministic learned replay passed: {deterministic_learned_replay}")
    print(f"Learned final MSE no worse than random: {learned_final <= random_final}")
    print(f"Learned/exact final MSE ratio: {learned_final / max(exact_final, 1e-12):.6f}")
    print(f"Saved all-method smoke artifacts to: {args.output_dir.resolve()}")
    print(
        "This is a development-only engineering diagnostic. "
        "It does not make the controlled Stage 3 decision."
    )


if __name__ == "__main__":
    main()
