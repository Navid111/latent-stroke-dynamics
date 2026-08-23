#!/usr/bin/env python3
"""Guarded six-target five-method latent-planner controlled comparison."""

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

from latent_stroke_dynamics.latent_controlled import (
    CONTROLLED_METHODS,
    CONTROLLED_PLANNER_SEEDS,
    CONTROLLED_TARGET_SEEDS,
    aggregate_controlled_summary,
    controlled_output_paths,
    make_controlled_decision,
    require_controlled_authorized,
    require_controlled_outputs_absent,
    validate_controlled_runner_request,
    validate_controlled_summary,
)
from latent_stroke_dynamics.latent_planner import (
    DEFAULT_LATENT_PLANNER_CONFIG,
    load_latent_planner_config,
    load_latent_predictor_ensembles,
    load_task_latent_resources,
)
from latent_stroke_dynamics.latent_smoke import LatentPlanningRun, run_latent_planner
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
    parser.add_argument("--controlled-run", action="store_true")
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


def save_run(run: RunLike, root: Path) -> None:
    method_dir = root / method_name(run)
    method_dir.mkdir()
    values = trajectory_values(run)
    best_step = int(np.argmin(values))
    if len(run.frames) != len(run.steps) + 1:
        raise RuntimeError("Controlled artifact generation requires every exact frame.")
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


def run_five_methods(
    target: Image.Image,
    planner_seed: int,
    config: dict[str, Any],
    pixel_model: Any,
    autoencoder: Any,
    statistics: Any,
    ensembles: dict[str, Any],
    proposal_config: ProposalConfig,
) -> tuple[tuple[RunLike, ...], tuple[float, ...], dict[str, bool]]:
    planner = config["planner"]
    steps = planner["steps"]
    prediction_batch_size = planner["prediction_batch_size"]
    random_run, random_time = timed(
        run_planner,
        target,
        "random",
        steps=steps,
        seed=planner_seed,
        proposal_config=proposal_config,
        capture_frames=True,
    )
    exact_run, exact_time = timed(
        run_planner,
        target,
        "exact",
        steps=steps,
        seed=planner_seed,
        proposal_config=proposal_config,
        capture_frames=True,
    )
    learned_run, learned_time = timed(
        run_learned_planner,
        target,
        pixel_model,
        steps=steps,
        seed=planner_seed,
        proposal_config=proposal_config,
        prediction_batch_size=prediction_batch_size,
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
        steps=steps,
        seed=planner_seed,
        proposal_config=proposal_config,
        prediction_batch_size=prediction_batch_size,
        capture_frames=True,
    )
    latent_ranking_run, latent_ranking_time = timed(
        run_latent_planner,
        target,
        "latent_ranking",
        autoencoder,
        statistics,
        [item.model for item in ensembles["ranking_aware"]],
        steps=steps,
        seed=planner_seed,
        proposal_config=proposal_config,
        prediction_batch_size=prediction_batch_size,
        capture_frames=True,
    )

    learned_replay = run_learned_planner(
        target,
        pixel_model,
        steps=steps,
        seed=planner_seed,
        proposal_config=proposal_config,
        prediction_batch_size=prediction_batch_size,
        device="cpu",
        capture_frames=False,
    )
    latent_mse_replay = run_latent_planner(
        target,
        "latent_mse",
        autoencoder,
        statistics,
        [item.model for item in ensembles["mse_only"]],
        steps=steps,
        seed=planner_seed,
        proposal_config=proposal_config,
        prediction_batch_size=prediction_batch_size,
        capture_frames=False,
    )
    latent_ranking_replay = run_latent_planner(
        target,
        "latent_ranking",
        autoencoder,
        statistics,
        [item.model for item in ensembles["ranking_aware"]],
        steps=steps,
        seed=planner_seed,
        proposal_config=proposal_config,
        prediction_batch_size=prediction_batch_size,
        capture_frames=False,
    )
    replay = {
        "learned_pixel": replay_matches(learned_run, learned_replay),
        "latent_mse": replay_matches(latent_mse_run, latent_mse_replay),
        "latent_ranking": replay_matches(latent_ranking_run, latent_ranking_replay),
    }
    if not all(replay.values()):
        raise RuntimeError("A controlled learned-method deterministic replay failed.")
    return (
        (
            random_run,
            exact_run,
            learned_run,
            latent_mse_run,
            latent_ranking_run,
        ),
        (
            random_time,
            exact_time,
            learned_time,
            latent_mse_time,
            latent_ranking_time,
        ),
        replay,
    )


def save_progress_plot(progress: pd.DataFrame, path: Path) -> None:
    colors = {
        "random": "tab:gray",
        "exact_pixel": "black",
        "learned_pixel": "tab:blue",
        "latent_mse": "tab:orange",
        "latent_ranking": "tab:green",
    }
    figure, axis = plt.subplots(figsize=(9, 5.5))
    grouped = progress.groupby(["method", "step"])["mse"].agg(["mean", "std"])
    for method in CONTROLLED_METHODS:
        values = grouped.loc[method].reset_index()
        standard_deviation = values["std"].fillna(0.0)
        axis.plot(values["step"], values["mean"], label=method, color=colors[method])
        axis.fill_between(
            values["step"],
            values["mean"] - standard_deviation,
            values["mean"] + standard_deviation,
            color=colors[method],
            alpha=0.12,
        )
    axis.set_xlabel("Executed strokes")
    axis.set_ylabel("Mean target pixel MSE")
    axis.set_title("Controlled latent planning across six frozen targets")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_montage(targets_root: Path, path: Path) -> None:
    titles = ("Target",) + CONTROLLED_METHODS
    figure, axes = plt.subplots(6, 6, figsize=(17, 17))
    for row in range(6):
        target_dir = targets_root / f"target_{row + 1:02d}"
        image_paths = [target_dir / "target.png"] + [
            target_dir / method / "final_canvas.png" for method in CONTROLLED_METHODS
        ]
        for column, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                axes[row, column].imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
            if row == 0:
                axes[row, column].set_title(titles[column])
            if column == 0:
                axes[row, column].set_ylabel(f"Target {row + 1}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.validate_only == args.controlled_run:
        raise ValueError("Choose exactly one of --validate-only or --controlled-run.")
    config = load_latent_planner_config(args.config)
    paths = controlled_output_paths(config)
    require_controlled_outputs_absent(paths)

    if args.validate_only:
        print(
            json.dumps(
                validate_controlled_runner_request(config),
                indent=2,
                sort_keys=True,
            )
        )
        return

    require_controlled_authorized(config)

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

    planner = config["planner"]
    proposal = planner["proposal"]
    proposal_config = ProposalConfig(
        count=planner["candidates_per_step"],
        error_guided_fraction=proposal["error_guided_fraction"],
        min_length=proposal["min_length"],
        max_length=proposal["max_length"],
        width_choices=tuple(proposal["width_choices"]),
        value_choices=tuple(proposal["value_choices"]),
    )

    paths.incomplete.mkdir(parents=True)
    targets_root = paths.incomplete / "targets"
    targets_root.mkdir()
    summary_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    integrity_by_target: dict[str, Any] = {}

    for index, (target_seed, planner_seed) in enumerate(
        zip(CONTROLLED_TARGET_SEEDS, CONTROLLED_PLANNER_SEEDS, strict=True),
        start=1,
    ):
        target_id = f"target_{index:02d}"
        target_dir = targets_root / target_id
        target_dir.mkdir()
        print(f"Running controlled target {index}/6...")
        target = random_base_canvas(
            size=config["canvas_size"],
            prior_strokes=planner["target_strokes"],
            rng=np.random.default_rng(target_seed),
        )
        target.save(target_dir / "target.png")
        runs, elapsed, replay = run_five_methods(
            target,
            planner_seed,
            config,
            pixel_model,
            autoencoder,
            statistics,
            ensembles,
            proposal_config,
        )
        target_summary_rows: list[dict[str, Any]] = []
        target_step_rows: list[dict[str, Any]] = []
        for run, seconds in zip(runs, elapsed, strict=True):
            row = {
                "target_id": target_id,
                "target_seed": target_seed,
                "planner_seed": planner_seed,
                **summary_row(run, seconds),
            }
            target_summary_rows.append(row)
            summary_rows.append(row)
            for item in step_rows(run):
                expanded = {
                    "target_id": target_id,
                    "target_seed": target_seed,
                    "planner_seed": planner_seed,
                    **item,
                }
                target_step_rows.append(expanded)
                all_step_rows.append(expanded)
            values = trajectory_values(run)
            for step, value in enumerate(values):
                progress_rows.append(
                    {
                        "target_id": target_id,
                        "method": method_name(run),
                        "step": step,
                        "mse": float(value),
                    }
                )
            save_run(run, target_dir)

        latent_runs = [
            run for run in runs if isinstance(run, LatentPlanningRun)
        ]
        target_integrity = {
            "deterministic_replay": replay,
            "target_encoded_once_per_latent_method": all(
                run.target_encoding_count == 1 for run in latent_runs
            ),
            "observed_canvas_reencoded_every_step": all(
                run.observed_canvas_encoding_count == planner["steps"]
                for run in latent_runs
            ),
            "predicted_latent_rolled_forward": False,
        }
        if not (
            all(replay.values())
            and target_integrity["target_encoded_once_per_latent_method"]
            and target_integrity["observed_canvas_reencoded_every_step"]
        ):
            raise RuntimeError(f"Controlled integrity failed for {target_id}.")
        integrity_by_target[target_id] = target_integrity
        pd.DataFrame(target_summary_rows).to_csv(
            target_dir / "summary.csv",
            index=False,
        )
        pd.DataFrame(target_step_rows).to_csv(
            target_dir / "step_diagnostics.csv",
            index=False,
        )
        (target_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "target_id": target_id,
                    "target_seed": target_seed,
                    "planner_seed": planner_seed,
                    "steps": planner["steps"],
                    "candidates_per_step": planner["candidates_per_step"],
                    **target_integrity,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  target {index} complete")

    summary = pd.DataFrame(summary_rows)
    validate_controlled_summary(summary, config)
    aggregate = aggregate_controlled_summary(summary)
    progress = pd.DataFrame(progress_rows)
    all_replays = all(
        all(payload["deterministic_replay"].values())
        for payload in integrity_by_target.values()
    )
    target_encoding_passed = all(
        payload["target_encoded_once_per_latent_method"]
        for payload in integrity_by_target.values()
    )
    observed_encoding_passed = all(
        payload["observed_canvas_reencoded_every_step"]
        for payload in integrity_by_target.values()
    )
    all_models_frozen = bool(
        not any(parameter.requires_grad for parameter in autoencoder.parameters())
        and not any(parameter.requires_grad for parameter in pixel_model.parameters())
        and not any(
            parameter.requires_grad
            for group in ensembles.values()
            for item in group
            for parameter in item.model.parameters()
        )
    )
    implementation_integrity_passed = bool(
        all_replays
        and target_encoding_passed
        and observed_encoding_passed
        and all_models_frozen
    )
    decision = make_controlled_decision(
        summary,
        config,
        implementation_integrity_passed=implementation_integrity_passed,
    )

    summary.to_csv(paths.incomplete / "per_target_summary.csv", index=False)
    pd.DataFrame(all_step_rows).to_csv(
        paths.incomplete / "step_diagnostics.csv",
        index=False,
    )
    progress.to_csv(paths.incomplete / "progress_by_step.csv", index=False)
    aggregate.to_csv(paths.incomplete / "aggregate_summary.csv", index=False)
    pd.DataFrame([decision]).to_csv(paths.incomplete / "decision.csv", index=False)
    (paths.incomplete / "decision.json").write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )
    (paths.incomplete / "integrity_by_target.json").write_text(
        json.dumps(integrity_by_target, indent=2),
        encoding="utf-8",
    )
    save_progress_plot(progress, paths.incomplete / "aggregate_progress.png")
    save_montage(targets_root, paths.incomplete / "final_montage.png")

    run_config = {
        "status": (
            "latent_planner_controlled_complete_integrity_passed_"
            + decision["status"]
        ),
        "controlled_authorized": True,
        "smoke_authorized": False,
        "single_run": True,
        "methods": list(CONTROLLED_METHODS),
        "target_seeds": list(CONTROLLED_TARGET_SEEDS),
        "planner_seeds": list(CONTROLLED_PLANNER_SEEDS),
        "steps": planner["steps"],
        "candidates_per_step": planner["candidates_per_step"],
        "prediction_batch_size": planner["prediction_batch_size"],
        "proposal": proposal,
        "latent_score": planner["latent_score"],
        "latent_predictor_state_sha256": {
            method: {str(item.seed): item.state_sha256 for item in group}
            for method, group in ensembles.items()
        },
        "pixel_predictor_state_sha256": pixel_digest,
        "implementation_integrity_passed": implementation_integrity_passed,
        "all_deterministic_replays_passed": all_replays,
        "target_encoded_once_per_latent_method": target_encoding_passed,
        "observed_canvas_reencoded_every_step": observed_encoding_passed,
        "predicted_latent_rolled_forward": False,
        "all_models_frozen": all_models_frozen,
        "models_trained_or_finetuned": False,
        "criteria_frozen_before_controlled_data": True,
        "smoke_result_unchanged": True,
        "historical_results_unchanged": True,
        "decision": decision,
    }
    (paths.incomplete / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    paths.incomplete.rename(paths.final)

    print("\nControlled latent-planner aggregate summary\n")
    print(aggregate.to_string(index=False))
    print("\nControlled decision\n")
    print(json.dumps(decision, indent=2))
    print(f"\nSaved controlled artifacts to: {paths.final.resolve()}")
    print("Do not rerun or retune this controlled comparison.")


if __name__ == "__main__":
    main()
