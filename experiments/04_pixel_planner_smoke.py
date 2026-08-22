"""Stage 3 engineering smoke for random and exact-greedy pixel planners."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from latent_stroke_dynamics.planning import (
    PlanningRun,
    ProposalConfig,
    pixel_mae,
    pixel_mse,
    run_planner,
)
from latent_stroke_dynamics.renderer import random_base_canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canvas-size", type=int, default=64)
    parser.add_argument("--target-strokes", type=int, default=20)
    parser.add_argument("--target-seed", type=int, default=20260901)
    parser.add_argument("--planner-seed", type=int, default=20260822)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--gif-scale", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage3-smoke-1"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.canvas_size < 8:
        raise ValueError("--canvas-size must be at least 8.")
    if args.target_strokes < 1:
        raise ValueError("--target-strokes must be positive.")
    if args.target_seed < 0 or args.planner_seed < 0:
        raise ValueError("Seeds must be non-negative.")
    if args.steps < 1 or args.candidates < 2:
        raise ValueError("--steps must be positive and --candidates at least 2.")
    if args.gif_scale < 1:
        raise ValueError("--gif-scale must be positive.")


def step_rows(run: PlanningRun) -> list[dict[str, int | float | bool | None]]:
    rows: list[dict[str, int | float | bool | None]] = []
    for record in run.steps:
        rows.append(
            {
                "method": run.method,
                "step": record.step,
                "selected_index": record.selected_index,
                "candidate_count": record.candidate_count,
                "mse_before": record.mse_before,
                "mse_after": record.mse_after,
                "mae_after": record.mae_after,
                "best_candidate_mse": record.best_candidate_mse,
                "improved": record.improved,
                "stroke_x0": record.stroke.x0,
                "stroke_y0": record.stroke.y0,
                "stroke_x1": record.stroke.x1,
                "stroke_y1": record.stroke.y1,
                "stroke_width": record.stroke.width,
                "stroke_value": record.stroke.value,
            }
        )
    return rows


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


def save_run(run: PlanningRun, output_dir: Path, gif_scale: int) -> None:
    method_dir = output_dir / run.method
    method_dir.mkdir(parents=True, exist_ok=True)
    run.initial_canvas.save(method_dir / "initial_canvas.png")
    run.final_canvas.save(method_dir / "final_canvas.png")
    pd.DataFrame(step_rows(run)).to_csv(method_dir / "progress.csv", index=False)
    stroke_payload = [
        {
            "step": record.step,
            "selected_index": record.selected_index,
            **asdict(record.stroke),
        }
        for record in run.steps
    ]
    (method_dir / "strokes.json").write_text(
        json.dumps(stroke_payload, indent=2),
        encoding="utf-8",
    )
    save_gif(run.frames, method_dir / "painting.gif", scale=gif_scale)


def summary_row(run: PlanningRun) -> dict[str, int | float | str]:
    initial_mse = pixel_mse(run.initial_canvas, run.target)
    final_mse = pixel_mse(run.final_canvas, run.target)
    return {
        "method": run.method,
        "steps": len(run.steps),
        "candidates_per_step": run.steps[0].candidate_count,
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "final_mae": pixel_mae(run.final_canvas, run.target),
        "relative_mse_improvement": (initial_mse - final_mse) / initial_mse,
        "improved_steps": sum(int(record.improved) for record in run.steps),
    }


def save_progress_plot(
    runs: tuple[PlanningRun, ...],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for run in runs:
        values = [pixel_mse(run.initial_canvas, run.target)] + [
            record.mse_after for record in run.steps
        ]
        axis.plot(range(len(values)), values, marker="o", markersize=3, label=run.method)
    axis.set_xlabel("Executed strokes")
    axis.set_ylabel("Target pixel MSE")
    axis.set_title("Stage 3 smoke: target error by planning step")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_comparison_plot(
    target: Image.Image,
    random_run: PlanningRun,
    exact_run: PlanningRun,
    output_path: Path,
) -> None:
    target_values = np.asarray(target, dtype=np.int16)
    random_values = np.asarray(random_run.final_canvas, dtype=np.int16)
    exact_values = np.asarray(exact_run.final_canvas, dtype=np.int16)
    random_error = np.abs(random_values - target_values)
    exact_error = np.abs(exact_values - target_values)

    figure, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    axes[0].imshow(target_values, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Synthetic target")
    axes[1].imshow(random_values, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Random\nMSE {pixel_mse(random_run.final_canvas, target):.5f}")
    axes[2].imshow(exact_values, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"Exact greedy\nMSE {pixel_mse(exact_run.final_canvas, target):.5f}")
    axes[3].imshow(random_error, cmap="magma", vmin=0, vmax=255)
    axes[3].set_title("Random absolute error\nfixed scale 0–255")
    axes[4].imshow(exact_error, cmap="magma", vmin=0, vmax=255)
    axes[4].set_title("Exact absolute error\nfixed scale 0–255")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_rng = np.random.default_rng(args.target_seed)
    target = random_base_canvas(
        size=args.canvas_size,
        prior_strokes=args.target_strokes,
        rng=target_rng,
    )
    target.save(args.output_dir / "target.png")

    proposal_config = ProposalConfig(count=args.candidates)
    random_run = run_planner(
        target,
        "random",
        steps=args.steps,
        seed=args.planner_seed,
        proposal_config=proposal_config,
        capture_frames=True,
    )
    exact_run = run_planner(
        target,
        "exact",
        steps=args.steps,
        seed=args.planner_seed,
        proposal_config=proposal_config,
        capture_frames=True,
    )
    exact_replay = run_planner(
        target,
        "exact",
        steps=args.steps,
        seed=args.planner_seed,
        proposal_config=proposal_config,
        capture_frames=False,
    )

    deterministic_replay_passed = bool(
        exact_run.steps == exact_replay.steps
        and np.array_equal(
            np.asarray(exact_run.final_canvas),
            np.asarray(exact_replay.final_canvas),
        )
    )
    if not deterministic_replay_passed:
        raise RuntimeError("Exact-planner deterministic replay failed.")

    save_run(random_run, args.output_dir, gif_scale=args.gif_scale)
    save_run(exact_run, args.output_dir, gif_scale=args.gif_scale)
    summaries = pd.DataFrame([summary_row(random_run), summary_row(exact_run)])
    if not bool(np.isfinite(summaries.select_dtypes(include=[np.number])).all().all()):
        raise RuntimeError("Smoke summary contains a non-finite metric.")
    summaries.to_csv(args.output_dir / "summary.csv", index=False)
    save_progress_plot((random_run, exact_run), args.output_dir / "progress_curves.png")
    save_comparison_plot(
        target,
        random_run,
        exact_run,
        args.output_dir / "final_comparison.png",
    )

    random_final = float(
        summaries.loc[summaries["method"] == "random", "final_mse"].iloc[0]
    )
    exact_final = float(
        summaries.loc[summaries["method"] == "exact", "final_mse"].iloc[0]
    )
    config = {
        "diagnostic_only": True,
        "canvas_size": args.canvas_size,
        "target_strokes": args.target_strokes,
        "target_seed": args.target_seed,
        "planner_seed": args.planner_seed,
        "steps": args.steps,
        "candidates_per_step": args.candidates,
        "proposal": {
            "error_guided_fraction": proposal_config.error_guided_fraction,
            "min_length": proposal_config.min_length,
            "max_length": proposal_config.max_length,
            "width_choices": list(proposal_config.width_choices),
            "value_choices": list(proposal_config.value_choices),
        },
        "deterministic_replay_passed": deterministic_replay_passed,
        "exact_final_mse_no_worse_than_random": exact_final <= random_final,
        "formal_stage3_decision_made": False,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nStage 3 random/exact engineering smoke\n")
    print(summaries.to_string(index=False))
    print(f"\nDeterministic replay passed: {deterministic_replay_passed}")
    print(f"Exact final MSE no worse than random: {exact_final <= random_final}")
    print(f"Saved smoke artifacts to: {args.output_dir.resolve()}")
    print(
        "This is a development-only engineering diagnostic. "
        "It does not make the controlled Stage 3 decision."
    )


if __name__ == "__main__":
    main()
