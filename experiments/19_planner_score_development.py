#!/usr/bin/env python3
"""Guarded three-target long-horizon planner-score development comparison."""

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
from latent_stroke_dynamics.planner_score_alignment import (
    DEFAULT_SCORE_ALIGNMENT_CONFIG,
    load_score_alignment_config,
)
from latent_stroke_dynamics.planner_score_development import (
    EXPECTED_PIXEL_SHA256,
    PLANNER_DEVELOPMENT_METHODS,
    PLANNER_DEVELOPMENT_SEEDS,
    PLANNER_DEVELOPMENT_TARGET_SEEDS,
    SELECTED_PREDICTOR_FAMILY,
    SELECTED_SCORE_NAME,
    SelectedScorePlanningRun,
    aggregate_planner_development_summary,
    load_frozen_development_selection,
    make_planner_development_decision,
    planner_development_output_paths,
    require_planner_development_authorized,
    require_planner_development_outputs_absent,
    run_selected_score_planner,
    validate_planner_development_resources,
    validate_planner_development_runner_request,
    validate_planner_development_summary,
)
from latent_stroke_dynamics.planning import (
    PlanningRun,
    ProposalConfig,
    pixel_mae,
    pixel_mse,
    run_planner,
)
from latent_stroke_dynamics.renderer import random_base_canvas


RunLike = PlanningRun | LearnedPlanningRun | LatentPlanningRun | SelectedScorePlanningRun


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SCORE_ALIGNMENT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--planner-development", action="store_true")
    return parser.parse_args()


def method_name(run: RunLike) -> str:
    if isinstance(run, SelectedScorePlanningRun):
        return run.method
    if isinstance(run, LatentPlanningRun):
        return "current_latent_mse_forced"
    if isinstance(run, LearnedPlanningRun):
        return "learned_pixel"
    return "exact_pixel"


def timed(function: Any, *args: Any, **kwargs: Any) -> tuple[RunLike, float]:
    started = time.perf_counter()
    run = function(*args, **kwargs)
    return run, time.perf_counter() - started


def replay_matches(first: RunLike, second: RunLike) -> bool:
    return bool(
        first.steps == second.steps
        and np.array_equal(np.asarray(first.final_canvas), np.asarray(second.final_canvas))
        and (
            not isinstance(first, SelectedScorePlanningRun)
            or (
                isinstance(second, SelectedScorePlanningRun)
                and first.stop_decision == second.stop_decision
            )
        )
    )


def trajectory_values(run: RunLike) -> np.ndarray:
    return np.asarray(
        [pixel_mse(run.initial_canvas, run.target)]
        + [record.mse_after for record in run.steps],
        dtype=np.float64,
    )


def summary_row(
    run: RunLike,
    elapsed_seconds: float,
    *,
    maximum_steps: int,
    candidates_per_step: int,
) -> dict[str, Any]:
    values = trajectory_values(run)
    best_step = int(np.argmin(values))
    stop_decision = (
        run.stop_decision if isinstance(run, SelectedScorePlanningRun) else None
    )
    row: dict[str, Any] = {
        "method": method_name(run),
        "maximum_steps": maximum_steps,
        "executed_steps": len(run.steps),
        "candidates_per_step": candidates_per_step,
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
        "stopped_early": stop_decision is not None,
        "stop_round": stop_decision.proposal_round if stop_decision else None,
        "current_score_at_stop": stop_decision.current_score if stop_decision else None,
        "best_candidate_score_at_stop": (
            stop_decision.best_candidate_score if stop_decision else None
        ),
        "exact_top1_rate": None,
        "exact_top5_rate": None,
        "mean_exact_rank": None,
        "mean_exact_regret": None,
        "max_exact_regret": None,
        "mean_score_exact_spearman": None,
    }
    if isinstance(run, (LearnedPlanningRun, LatentPlanningRun, SelectedScorePlanningRun)) and run.steps:
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
    if isinstance(run, (LatentPlanningRun, SelectedScorePlanningRun)) and run.steps:
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
        raise RuntimeError("Planner-development artifacts require every exact frame.")
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
    if isinstance(run, SelectedScorePlanningRun):
        (method_dir / "stop_decision.json").write_text(
            json.dumps(
                asdict(run.stop_decision) if run.stop_decision else None,
                indent=2,
            ),
            encoding="utf-8",
        )


def run_five_methods(
    target: Image.Image,
    planner_seed: int,
    config: dict[str, Any],
    closed_config: dict[str, Any],
    pixel_model: Any,
    autoencoder: Any,
    statistics: Any,
    ensembles: dict[str, Any],
    proposal_config: ProposalConfig,
) -> tuple[tuple[RunLike, ...], tuple[float, ...], dict[str, bool]]:
    phase = config["planner_development"]
    steps = phase["maximum_steps"]
    batch_size = phase["prediction_batch_size"]
    mse_models = [item.model for item in ensembles[SELECTED_PREDICTOR_FAMILY]]

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
        prediction_batch_size=batch_size,
        device="cpu",
        capture_frames=True,
    )
    current_mse_run, current_mse_time = timed(
        run_latent_planner,
        target,
        "latent_mse",
        autoencoder,
        statistics,
        mse_models,
        steps=steps,
        seed=planner_seed,
        proposal_config=proposal_config,
        prediction_batch_size=batch_size,
        capture_frames=True,
    )
    selected_forced_run, selected_forced_time = timed(
        run_selected_score_planner,
        target,
        autoencoder,
        statistics,
        mse_models,
        maximum_steps=steps,
        seed=planner_seed,
        proposal_config=proposal_config,
        prediction_batch_size=batch_size,
        allow_no_op=False,
        no_op_margin=phase["no_op_margin"],
        capture_frames=True,
    )
    selected_no_op_run, selected_no_op_time = timed(
        run_selected_score_planner,
        target,
        autoencoder,
        statistics,
        mse_models,
        maximum_steps=steps,
        seed=planner_seed,
        proposal_config=proposal_config,
        prediction_batch_size=batch_size,
        allow_no_op=True,
        no_op_margin=phase["no_op_margin"],
        capture_frames=True,
    )

    replay_specs = {
        "learned_pixel": (
            run_learned_planner,
            (target, pixel_model),
            {
                "steps": steps,
                "seed": planner_seed,
                "proposal_config": proposal_config,
                "prediction_batch_size": batch_size,
                "device": "cpu",
                "capture_frames": False,
            },
            learned_run,
        ),
        "current_latent_mse_forced": (
            run_latent_planner,
            (target, "latent_mse", autoencoder, statistics, mse_models),
            {
                "steps": steps,
                "seed": planner_seed,
                "proposal_config": proposal_config,
                "prediction_batch_size": batch_size,
                "capture_frames": False,
            },
            current_mse_run,
        ),
        "development_selected_score_forced": (
            run_selected_score_planner,
            (target, autoencoder, statistics, mse_models),
            {
                "maximum_steps": steps,
                "seed": planner_seed,
                "proposal_config": proposal_config,
                "prediction_batch_size": batch_size,
                "allow_no_op": False,
                "no_op_margin": phase["no_op_margin"],
                "capture_frames": False,
            },
            selected_forced_run,
        ),
        "development_selected_score_no_op": (
            run_selected_score_planner,
            (target, autoencoder, statistics, mse_models),
            {
                "maximum_steps": steps,
                "seed": planner_seed,
                "proposal_config": proposal_config,
                "prediction_batch_size": batch_size,
                "allow_no_op": True,
                "no_op_margin": phase["no_op_margin"],
                "capture_frames": False,
            },
            selected_no_op_run,
        ),
    }
    replay: dict[str, bool] = {}
    for name, (function, positional, keyword, original) in replay_specs.items():
        repeated = function(*positional, **keyword)
        replay[name] = replay_matches(original, repeated)
    if not all(replay.values()):
        raise RuntimeError("A planner-development deterministic replay failed.")

    return (
        (
            exact_run,
            learned_run,
            current_mse_run,
            selected_forced_run,
            selected_no_op_run,
        ),
        (
            exact_time,
            learned_time,
            current_mse_time,
            selected_forced_time,
            selected_no_op_time,
        ),
        replay,
    )


def save_progress_plot(progress: pd.DataFrame, path: Path) -> None:
    colors = {
        "exact_pixel": "black",
        "learned_pixel": "tab:blue",
        "current_latent_mse_forced": "tab:orange",
        "development_selected_score_forced": "tab:purple",
        "development_selected_score_no_op": "tab:green",
    }
    figure, axis = plt.subplots(figsize=(10, 5.8))
    grouped = progress.groupby(["method", "step"])["mse"].agg(["mean", "std"])
    for method in PLANNER_DEVELOPMENT_METHODS:
        values = grouped.loc[method].reset_index()
        deviation = values["std"].fillna(0.0)
        axis.plot(values["step"], values["mean"], label=method, color=colors[method])
        axis.fill_between(
            values["step"],
            values["mean"] - deviation,
            values["mean"] + deviation,
            color=colors[method],
            alpha=0.12,
        )
    axis.set_xlabel("Executed strokes / held final state after no-op")
    axis.set_ylabel("Mean target pixel MSE")
    axis.set_title("Planner-score development across three reserved targets")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_montage(targets_root: Path, path: Path) -> None:
    titles = ("Target",) + PLANNER_DEVELOPMENT_METHODS
    figure, axes = plt.subplots(3, 6, figsize=(17, 9))
    for row in range(3):
        target_dir = targets_root / f"target_{row + 1:02d}"
        paths = [target_dir / "target.png"] + [
            target_dir / method / "final_canvas.png"
            for method in PLANNER_DEVELOPMENT_METHODS
        ]
        for column, image_path in enumerate(paths):
            with Image.open(image_path) as image:
                axes[row, column].imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=8)
            if column == 0:
                axes[row, column].set_ylabel(f"Target {row + 1}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.validate_only == args.planner_development:
        raise ValueError("Choose exactly one runner mode.")

    config = load_score_alignment_config(args.config)
    selection = load_frozen_development_selection()
    closed_config = load_latent_planner_config(
        config["frozen_resources"]["latent_planner_config"]
    )
    resource_checks = validate_planner_development_resources(
        config,
        closed_config,
        selection,
    )
    paths = planner_development_output_paths(config)
    require_planner_development_outputs_absent(paths)

    if args.validate_only:
        result = validate_planner_development_runner_request(config, selection)
        result.update(resource_checks)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    require_planner_development_authorized(config)
    autoencoder, statistics = load_task_latent_resources(closed_config)
    ensembles = load_latent_predictor_ensembles(closed_config)
    pixel_model, pixel_metadata = load_pixel_checkpoint(
        closed_config["pixel_predictor"]["path"],
        device="cpu",
    )
    pixel_digest = state_dict_sha256(pixel_model)
    if pixel_digest != EXPECTED_PIXEL_SHA256:
        raise RuntimeError("Frozen pixel checkpoint digest mismatch.")
    if pixel_metadata.model_seed != 11 or pixel_metadata.parameter_count != 833:
        raise RuntimeError("Frozen pixel checkpoint metadata mismatch.")
    for parameter in pixel_model.parameters():
        parameter.requires_grad_(False)

    phase = config["planner_development"]
    proposal = config["proposal"]
    proposal_config = ProposalConfig(
        count=phase["candidates_per_step"],
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
    step_diagnostics: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    integrity_by_target: dict[str, Any] = {}

    for index, (target_seed, planner_seed) in enumerate(
        zip(
            PLANNER_DEVELOPMENT_TARGET_SEEDS,
            PLANNER_DEVELOPMENT_SEEDS,
            strict=True,
        ),
        start=1,
    ):
        target_id = f"target_{index:02d}"
        print(f"Running planner-development target {index}/3...")
        target_dir = targets_root / target_id
        target_dir.mkdir()
        target = random_base_canvas(
            size=config["canvas_size"],
            prior_strokes=phase["target_strokes"],
            rng=np.random.default_rng(target_seed),
        )
        target.save(target_dir / "target.png")
        runs, elapsed, replay = run_five_methods(
            target,
            planner_seed,
            config,
            closed_config,
            pixel_model,
            autoencoder,
            statistics,
            ensembles,
            proposal_config,
        )
        target_summary: list[dict[str, Any]] = []
        for run, seconds in zip(runs, elapsed, strict=True):
            row = {
                "target_id": target_id,
                "target_seed": target_seed,
                "planner_seed": planner_seed,
                **summary_row(
                    run,
                    seconds,
                    maximum_steps=phase["maximum_steps"],
                    candidates_per_step=phase["candidates_per_step"],
                ),
            }
            target_summary.append(row)
            summary_rows.append(row)
            for item in step_rows(run):
                step_diagnostics.append(
                    {
                        "target_id": target_id,
                        "target_seed": target_seed,
                        "planner_seed": planner_seed,
                        **item,
                    }
                )
            values = trajectory_values(run)
            held = np.full(phase["maximum_steps"] + 1, values[-1], dtype=np.float64)
            held[: len(values)] = values
            for step, value in enumerate(held):
                progress_rows.append(
                    {
                        "target_id": target_id,
                        "method": method_name(run),
                        "step": step,
                        "mse": float(value),
                        "state_held_after_no_op": step >= len(values),
                    }
                )
            save_run(run, target_dir)

        current_latent = next(
            run for run in runs if isinstance(run, LatentPlanningRun)
        )
        selected_runs = [
            run for run in runs if isinstance(run, SelectedScorePlanningRun)
        ]
        target_integrity = {
            "deterministic_replay": replay,
            "current_latent_target_encoded_once": current_latent.target_encoding_count == 1,
            "current_latent_observed_every_step": (
                current_latent.observed_canvas_encoding_count == phase["maximum_steps"]
            ),
            "selected_target_encoded_once": all(
                run.target_encoding_count == 1 for run in selected_runs
            ),
            "selected_observed_every_proposal_round": all(
                run.observed_canvas_encoding_count == run.proposal_rounds_evaluated
                for run in selected_runs
            ),
            "selected_predictor_family": SELECTED_PREDICTOR_FAMILY,
            "selected_score_name": SELECTED_SCORE_NAME,
            "no_op_margin": phase["no_op_margin"],
            "predicted_latent_rolled_forward": False,
        }
        if not (
            all(replay.values())
            and target_integrity["current_latent_target_encoded_once"]
            and target_integrity["current_latent_observed_every_step"]
            and target_integrity["selected_target_encoded_once"]
            and target_integrity["selected_observed_every_proposal_round"]
        ):
            raise RuntimeError(f"Planner-development integrity failed for {target_id}.")
        integrity_by_target[target_id] = target_integrity
        pd.DataFrame(target_summary).to_csv(target_dir / "summary.csv", index=False)
        (target_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "target_id": target_id,
                    "target_seed": target_seed,
                    "planner_seed": planner_seed,
                    "maximum_steps": phase["maximum_steps"],
                    "candidates_per_step": phase["candidates_per_step"],
                    **target_integrity,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  target {index} complete")

    summary = pd.DataFrame(summary_rows)
    validate_planner_development_summary(summary, config)
    aggregate = aggregate_planner_development_summary(summary)
    progress = pd.DataFrame(progress_rows)
    all_replays = all(
        all(payload["deterministic_replay"].values())
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
    implementation_integrity_passed = bool(all_replays and all_models_frozen)
    decision = make_planner_development_decision(
        summary,
        config,
        implementation_integrity_passed=implementation_integrity_passed,
        selected_pair_matches=True,
    )

    summary.to_csv(paths.incomplete / "per_target_summary.csv", index=False)
    pd.DataFrame(step_diagnostics).to_csv(
        paths.incomplete / "step_diagnostics.csv",
        index=False,
    )
    progress.to_csv(paths.incomplete / "progress_by_step.csv", index=False)
    aggregate.to_csv(paths.incomplete / "aggregate_summary.csv", index=False)
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
        "status": "planner_score_planner_development_complete",
        "single_run": True,
        "methods": list(PLANNER_DEVELOPMENT_METHODS),
        "target_seeds": list(PLANNER_DEVELOPMENT_TARGET_SEEDS),
        "planner_seeds": list(PLANNER_DEVELOPMENT_SEEDS),
        "maximum_steps": phase["maximum_steps"],
        "candidates_per_step": phase["candidates_per_step"],
        "prediction_batch_size": phase["prediction_batch_size"],
        "selected_predictor_family": SELECTED_PREDICTOR_FAMILY,
        "selected_score_name": SELECTED_SCORE_NAME,
        "no_op_margin": phase["no_op_margin"],
        "resource_checks": resource_checks,
        "pixel_predictor_state_sha256": pixel_digest,
        "implementation_integrity_passed": implementation_integrity_passed,
        "all_deterministic_replays_passed": all_replays,
        "all_models_frozen": all_models_frozen,
        "models_trained_or_finetuned": False,
        "closed_targets_reused": False,
        "score_audit_result_unchanged": True,
        "controlled_result_unchanged": True,
        "historical_results_unchanged": True,
        "decision": decision,
    }
    (paths.incomplete / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    paths.incomplete.rename(paths.final)

    print("\nPlanner-development aggregate summary\n")
    print(aggregate.to_string(index=False))
    print("\nPlanner-development decision\n")
    print(json.dumps(decision, indent=2))
    print(f"\nSaved planner-development artifacts to: {paths.final.resolve()}")
    print("Do not rerun or tune against this planner-development result.")


if __name__ == "__main__":
    main()
