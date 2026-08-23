#!/usr/bin/env python3
"""Guarded five-method latent-planner smoke runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from latent_stroke_dynamics.latent_planner import (
    DEFAULT_LATENT_PLANNER_CONFIG,
    load_latent_planner_config,
    load_latent_predictor_ensembles,
    load_task_latent_resources,
)
from latent_stroke_dynamics.latent_smoke import (
    LatentPlanningRun,
    SMOKE_METHODS,
    require_smoke_authorized,
    require_smoke_outputs_absent,
    run_latent_planner,
    smoke_output_paths,
    validate_smoke_runner_request,
)
from latent_stroke_dynamics.learned_pixel_planner import (
    LearnedPlanningRun,
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


RunLike = PlanningRun | LearnedPlanningRun | LatentPlanningRun


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_LATENT_PLANNER_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-run", action="store_true")
    return parser.parse_args()


def method_name(run: RunLike) -> str:
    if isinstance(run, LatentPlanningRun):
        return run.method
    if isinstance(run, LearnedPlanningRun):
        return "learned_pixel"
    return "exact_pixel" if run.method == "exact" else "random"


def timed(function: Any, *args: Any, **kwargs: Any) -> tuple[RunLike, float]:
    started = time.perf_counter()
    run = function(*args, **kwargs)
    return run, time.perf_counter() - started


def replay_matches(first: RunLike, second: RunLike) -> bool:
    return bool(
        first.steps == second.steps
        and np.array_equal(
            np.asarray(first.final_canvas),
            np.asarray(second.final_canvas),
        )
    )


def trajectory_values(run: RunLike) -> np.ndarray:
    return np.asarray(
        [pixel_mse(run.initial_canvas, run.target)]
        + [record.mse_after for record in run.steps],
        dtype=np.float64,
    )


def summary_row(run: RunLike, elapsed_seconds: float) -> dict[str, Any]:
    values = trajectory_values(run)
    best_step = int(np.argmin(values))
    row: dict[str, Any] = {
        "method": method_name(run),
        "steps": len(run.steps),
        "candidates_per_step": run.steps[0].candidate_count,
        "initial_mse": float(values[0]),
        "final_mse": float(values[-1]),
        "best_mse": float(values[best_step]),
        "best_step": best_step,
        "final_mae": pixel_mae(run.final_canvas, run.target),
        "relative_final_mse_improvement": float(
            (values[0] - values[-1]) / max(values[0], 1e-12)
        ),
        "relative_best_mse_improvement": float(
            (values[0] - values[best_step]) / max(values[0], 1e-12)
        ),
        "improved_steps": sum(int(record.improved) for record in run.steps),
        "elapsed_seconds": float(elapsed_seconds),
        "exact_top1_rate": None,
        "exact_top5_rate": None,
        "mean_exact_rank": None,
        "mean_exact_regret": None,
        "max_exact_regret": None,
        "mean_score_exact_spearman": None,
    }
    if isinstance(run, (LearnedPlanningRun, LatentPlanningRun)):
        row.update(
            {
                "exact_top1_rate": float(np.mean([item.exact_top1 for item in run.steps])),
                "exact_top5_rate": float(np.mean([item.exact_top5 for item in run.steps])),
                "mean_exact_rank": float(
                    np.mean([item.exact_selected_rank for item in run.steps])
                ),
                "mean_exact_regret": float(np.mean([item.exact_regret for item in run.steps])),
                "max_exact_regret": float(np.max([item.exact_regret for item in run.steps])),
            }
        )
    if isinstance(run, LatentPlanningRun):
        row["mean_score_exact_spearman"] = float(
            np.mean([item.score_exact_spearman for item in run.steps])
        )
    return row


def step_rows(run: RunLike) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in run.steps:
        row: dict[str, Any] = {"method": method_name(run), **asdict(record)}
        stroke = row.pop("stroke")
        for name, value in stroke.items():
            row[f"stroke_{name}"] = value
        rows.append(row)
    return rows


def save_gif(frames: tuple[Image.Image, ...], path: Path) -> None:
    scaled = [
        frame.resize((384, 384), resample=Image.Resampling.NEAREST)
        for frame in frames
    ]
    scaled[0].save(
        path,
        save_all=True,
        append_images=scaled[1:],
        duration=180,
        loop=0,
        optimize=False,
    )


def save_run(run: RunLike, root: Path) -> None:
    method_dir = root / method_name(run)
    method_dir.mkdir()
    values = trajectory_values(run)
    best_step = int(np.argmin(values))
    if len(run.frames) != len(run.steps) + 1:
        raise RuntimeError("Smoke artifact generation requires every exact frame.")
    run.initial_canvas.save(method_dir / "initial_canvas.png")
    run.frames[best_step].save(method_dir / "best_canvas.png")
    run.final_canvas.save(method_dir / "final_canvas.png")
    rows = step_rows(run)
    pd.DataFrame(rows).to_csv(method_dir / "progress.csv", index=False)
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
    (method_dir / "strokes.json").write_text(
        json.dumps(strokes, indent=2),
        encoding="utf-8",
    )
    save_gif(run.frames, method_dir / "painting.gif")


def validate_summaries(summary: pd.DataFrame) -> None:
    if tuple(summary["method"]) != SMOKE_METHODS:
        raise RuntimeError("Smoke summary methods or order changed.")
    shared = [
        "steps",
        "candidates_per_step",
        "initial_mse",
        "final_mse",
        "best_mse",
        "best_step",
        "final_mae",
        "relative_final_mse_improvement",
        "relative_best_mse_improvement",
        "improved_steps",
        "elapsed_seconds",
    ]
    if not bool(np.isfinite(summary[shared].to_numpy(dtype=float)).all()):
        raise RuntimeError("Smoke shared summary contains a non-finite value.")
    learned = summary["method"].isin(("learned_pixel", "latent_mse", "latent_ranking"))
    diagnostics = [
        "exact_top1_rate",
        "exact_top5_rate",
        "mean_exact_rank",
        "mean_exact_regret",
        "max_exact_regret",
    ]
    if not bool(np.isfinite(summary.loc[learned, diagnostics].to_numpy(dtype=float)).all()):
        raise RuntimeError("Smoke learned-method diagnostics are non-finite.")
    latent = summary["method"].isin(("latent_mse", "latent_ranking"))
    if not bool(
        np.isfinite(
            summary.loc[latent, ["mean_score_exact_spearman"]].to_numpy(dtype=float)
        ).all()
    ):
        raise RuntimeError("Smoke latent rank correlations are non-finite.")


def save_progress_plot(runs: tuple[RunLike, ...], path: Path) -> None:
    colors = {
        "random": "tab:gray",
        "exact_pixel": "black",
        "learned_pixel": "tab:blue",
        "latent_mse": "tab:orange",
        "latent_ranking": "tab:green",
    }
    figure, axis = plt.subplots(figsize=(8, 5))
    for run in runs:
        name = method_name(run)
        axis.plot(trajectory_values(run), label=name, color=colors[name])
    axis.set_xlabel("Executed strokes")
    axis.set_ylabel("Target pixel MSE")
    axis.set_title("Latent-planner implementation smoke")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_montage(target: Image.Image, runs: tuple[RunLike, ...], path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10, 7))
    entries: list[tuple[str, Image.Image]] = [("Target", target)] + [
        (method_name(run), run.final_canvas) for run in runs
    ]
    for axis, (title, image) in zip(axes.ravel(), entries, strict=True):
        axis.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.validate_only == args.smoke_run:
        raise ValueError("Choose exactly one of --validate-only or --smoke-run.")
    config = load_latent_planner_config(args.config)
    paths = smoke_output_paths(config)
    require_smoke_outputs_absent(paths)

    if args.validate_only:
        print(json.dumps(validate_smoke_runner_request(config), indent=2, sort_keys=True))
        return

    require_smoke_authorized(config)

    autoencoder, statistics = load_task_latent_resources(config)
    ensembles = load_latent_predictor_ensembles(config)
    pixel_model, pixel_metadata = load_pixel_checkpoint(
        config["pixel_predictor"]["path"],
        device="cpu",
    )
    pixel_digest = state_dict_sha256(pixel_model)
    if pixel_digest != config["pixel_predictor"]["state_sha256"]:
        raise RuntimeError("Frozen pixel checkpoint digest mismatch.")
    if pixel_metadata.model_seed != 11 or pixel_metadata.parameter_count != 833:
        raise RuntimeError("Frozen pixel checkpoint metadata mismatch.")
    for parameter in pixel_model.parameters():
        parameter.requires_grad_(False)

    smoke = config["smoke"]
    planner = config["planner"]
    proposal = planner["proposal"]
    proposal_config = ProposalConfig(
        count=smoke["candidates_per_step"],
        error_guided_fraction=proposal["error_guided_fraction"],
        min_length=proposal["min_length"],
        max_length=proposal["max_length"],
        width_choices=tuple(proposal["width_choices"]),
        value_choices=tuple(proposal["value_choices"]),
    )

    paths.incomplete.mkdir(parents=True)
    target = random_base_canvas(
        size=config["canvas_size"],
        prior_strokes=planner["target_strokes"],
        rng=np.random.default_rng(smoke["target_seed"]),
    )
    target.save(paths.incomplete / "target.png")

    random_run, random_time = timed(
        run_planner,
        target,
        "random",
        steps=smoke["steps"],
        seed=smoke["planner_seed"],
        proposal_config=proposal_config,
        capture_frames=True,
    )
    exact_run, exact_time = timed(
        run_planner,
        target,
        "exact",
        steps=smoke["steps"],
        seed=smoke["planner_seed"],
        proposal_config=proposal_config,
        capture_frames=True,
    )
    learned_run, learned_time = timed(
        run_learned_planner,
        target,
        pixel_model,
        steps=smoke["steps"],
        seed=smoke["planner_seed"],
        proposal_config=proposal_config,
        prediction_batch_size=planner["prediction_batch_size"],
        device="cpu",
        capture_frames=True,
    )
    latent_mse_run, latent_mse_time = timed(
        run_latent_planner,
        target,
        "latent_mse",
        autoencoder,
        statistics,
        [item.model for item in ensembles["mse_only"]],
        steps=smoke["steps"],
        seed=smoke["planner_seed"],
        proposal_config=proposal_config,
        prediction_batch_size=planner["prediction_batch_size"],
        capture_frames=True,
    )
    latent_ranking_run, latent_ranking_time = timed(
        run_latent_planner,
        target,
        "latent_ranking",
        autoencoder,
        statistics,
        [item.model for item in ensembles["ranking_aware"]],
        steps=smoke["steps"],
        seed=smoke["planner_seed"],
        proposal_config=proposal_config,
        prediction_batch_size=planner["prediction_batch_size"],
        capture_frames=True,
    )

    learned_replay = run_learned_planner(
        target,
        pixel_model,
        steps=smoke["steps"],
        seed=smoke["planner_seed"],
        proposal_config=proposal_config,
        prediction_batch_size=planner["prediction_batch_size"],
        device="cpu",
        capture_frames=False,
    )
    latent_mse_replay = run_latent_planner(
        target,
        "latent_mse",
        autoencoder,
        statistics,
        [item.model for item in ensembles["mse_only"]],
        steps=smoke["steps"],
        seed=smoke["planner_seed"],
        proposal_config=proposal_config,
        prediction_batch_size=planner["prediction_batch_size"],
        capture_frames=False,
    )
    latent_ranking_replay = run_latent_planner(
        target,
        "latent_ranking",
        autoencoder,
        statistics,
        [item.model for item in ensembles["ranking_aware"]],
        steps=smoke["steps"],
        seed=smoke["planner_seed"],
        proposal_config=proposal_config,
        prediction_batch_size=planner["prediction_batch_size"],
        capture_frames=False,
    )
    replay = {
        "learned_pixel": replay_matches(learned_run, learned_replay),
        "latent_mse": replay_matches(latent_mse_run, latent_mse_replay),
        "latent_ranking": replay_matches(latent_ranking_run, latent_ranking_replay),
    }
    if not all(replay.values()):
        raise RuntimeError("A learned-method deterministic replay failed.")

    runs: tuple[RunLike, ...] = (
        random_run,
        exact_run,
        learned_run,
        latent_mse_run,
        latent_ranking_run,
    )
    elapsed = (
        random_time,
        exact_time,
        learned_time,
        latent_mse_time,
        latent_ranking_time,
    )
    summaries = pd.DataFrame(
        [summary_row(run, seconds) for run, seconds in zip(runs, elapsed, strict=True)]
    )
    validate_summaries(summaries)
    for run in runs:
        save_run(run, paths.incomplete)
    summaries.to_csv(paths.incomplete / "summary.csv", index=False)
    pd.DataFrame(
        [row for run in runs for row in step_rows(run)]
    ).to_csv(paths.incomplete / "step_diagnostics.csv", index=False)
    save_progress_plot(runs, paths.incomplete / "progress_curves.png")
    save_montage(target, runs, paths.incomplete / "final_montage.png")

    run_config = {
        "status": "latent_planner_smoke_complete_integrity_passed",
        "diagnostic_only": True,
        "smoke_authorized": True,
        "controlled_authorized": False,
        "methods": list(SMOKE_METHODS),
        "target_seed": smoke["target_seed"],
        "planner_seed": smoke["planner_seed"],
        "steps": smoke["steps"],
        "candidates_per_step": smoke["candidates_per_step"],
        "proposal": proposal,
        "latent_score": planner["latent_score"],
        "latent_predictor_state_sha256": {
            method: {str(item.seed): item.state_sha256 for item in group}
            for method, group in ensembles.items()
        },
        "pixel_predictor_state_sha256": pixel_digest,
        "target_encoded_once_per_latent_method": all(
            run.target_encoding_count == 1
            for run in (latent_mse_run, latent_ranking_run)
        ),
        "observed_canvas_reencoded_every_step": all(
            run.observed_canvas_encoding_count == smoke["steps"]
            for run in (latent_mse_run, latent_ranking_run)
        ),
        "predicted_latent_rolled_forward": False,
        "deterministic_replay": replay,
        "models_trained_or_finetuned": False,
        "historical_results_unchanged": True,
    }
    (paths.incomplete / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    paths.incomplete.rename(paths.final)
    print(summaries.to_string(index=False))
    print(json.dumps(run_config, indent=2))
    print(f"Saved smoke artifacts to: {paths.final.resolve()}")
    print("This smoke is diagnostic only. Controlled planning remains unauthorized.")


if __name__ == "__main__":
    main()
