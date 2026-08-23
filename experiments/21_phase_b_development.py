#!/usr/bin/env python3
"""Guarded single-run Phase B0 training and long-horizon development study."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from latent_stroke_dynamics.extension_training import model_state_sha256
from latent_stroke_dynamics.latent_planner import (
    load_latent_planner_config,
    load_latent_predictor_ensembles,
    load_task_latent_resources,
)
from latent_stroke_dynamics.learned_pixel_planner import (
    LearnedPlanningRun,
    load_pixel_checkpoint,
    run_learned_planner,
    state_dict_sha256,
)
from latent_stroke_dynamics.phase_b_data import (
    build_planner_payload,
    build_transition_payload,
    fit_progress_statistics,
)
from latent_stroke_dynamics.phase_b_development import (
    DEFAULT_PHASE_B_CONFIG,
    LONG_HORIZON_METHODS,
    load_phase_b_development_config,
    phase_b_output_paths,
    require_phase_b_development_authorized,
    validate_phase_b_development_runner_request,
)
from latent_stroke_dynamics.phase_b_planning import (
    PhaseBPlanningRun,
    run_phase_b_planner,
)
from latent_stroke_dynamics.phase_b_training import (
    feature_statistics,
    four_way_retrieval,
    freeze_phase_b_model,
    planner_candidate_metrics,
    save_phase_b_checkpoint,
    train_phase_b_variant,
)
from latent_stroke_dynamics.planner_score_development import (
    SelectedScorePlanningRun,
    run_selected_score_planner,
)
from latent_stroke_dynamics.planning import (
    PlanningRun,
    ProposalConfig,
    pixel_mae,
    pixel_mse,
    run_planner,
)
from latent_stroke_dynamics.renderer import random_base_canvas


RunLike = PlanningRun | LearnedPlanningRun | SelectedScorePlanningRun | PhaseBPlanningRun


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PHASE_B_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if args.validate_only == args.development:
        parser.error("Choose exactly one of --validate-only or --development.")
    return args


def _proposal(count: int) -> ProposalConfig:
    return ProposalConfig(
        count=count,
        error_guided_fraction=0.8,
        min_length=0.1,
        max_length=0.6,
        width_choices=(1, 2, 3, 4),
        value_choices=(0, 32, 64, 96, 128),
    )


def _trajectory(run: RunLike) -> np.ndarray:
    return np.asarray(
        [pixel_mse(run.initial_canvas, run.target)]
        + [float(step.mse_after) for step in run.steps],
        dtype=np.float64,
    )


def _summary_row(
    run: RunLike,
    method: str,
    elapsed: float,
    maximum_steps: int,
    candidates: int,
) -> dict[str, Any]:
    values = _trajectory(run)
    best_step = int(np.argmin(values))
    records = tuple(run.steps)
    stop = run.stop_decision if isinstance(run, PhaseBPlanningRun) else None
    row: dict[str, Any] = {
        "method": method,
        "maximum_steps": maximum_steps,
        "executed_steps": len(records),
        "candidates_per_step": candidates,
        "initial_mse": float(values[0]),
        "final_mse": float(values[-1]),
        "best_mse": float(values[best_step]),
        "best_step": best_step,
        "final_mae": pixel_mae(run.final_canvas, run.target),
        "stopped_early": stop is not None,
        "premature_stop": bool(stop.premature) if stop else False,
        "elapsed_seconds": elapsed,
        "exact_top1_rate": None,
        "exact_top5_rate": None,
        "mean_exact_rank": None,
        "mean_exact_regret": None,
        "mean_score_exact_spearman": None,
    }
    if records and hasattr(records[0], "exact_regret"):
        row.update(
            {
                "exact_top1_rate": float(np.mean([item.exact_top1 for item in records])),
                "exact_top5_rate": float(np.mean([item.exact_top5 for item in records])),
                "mean_exact_rank": float(np.mean([item.exact_selected_rank for item in records])),
                "mean_exact_regret": float(np.mean([item.exact_regret for item in records])),
            }
        )
    if records and hasattr(records[0], "score_exact_spearman"):
        row["mean_score_exact_spearman"] = float(
            np.mean([item.score_exact_spearman for item in records])
        )
    return row


def _save_run(run: RunLike, method: str, root: Path) -> None:
    directory = root / method
    directory.mkdir()
    values = _trajectory(run)
    if len(run.frames) != len(run.steps) + 1:
        raise RuntimeError("Phase B0 artifacts require every executed frame.")
    run.initial_canvas.save(directory / "initial.png")
    run.frames[int(np.argmin(values))].save(directory / "best.png")
    run.final_canvas.save(directory / "final.png")
    rows = []
    for item in run.steps:
        row = asdict(item)
        stroke = row.pop("stroke")
        row.update({f"stroke_{name}": value for name, value in stroke.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / "steps.csv", index=False)
    if isinstance(run, PhaseBPlanningRun):
        (directory / "stop_decision.json").write_text(
            json.dumps(asdict(run.stop_decision) if run.stop_decision else None, indent=2),
            encoding="utf-8",
        )


def _timed(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    config = load_phase_b_development_config(args.config)
    paths = phase_b_output_paths(config)
    if args.validate_only:
        print(
            json.dumps(
                validate_phase_b_development_runner_request(config),
                indent=2,
                sort_keys=True,
            )
        )
        return

    require_phase_b_development_authorized(config)
    started = time.perf_counter()
    paths.incomplete.mkdir(parents=True)
    data_root = paths.incomplete / "data_manifests"
    checkpoint_root = paths.incomplete / "checkpoints"
    targets_root = paths.incomplete / "targets"
    data_root.mkdir()
    checkpoint_root.mkdir()
    targets_root.mkdir()

    print("Generating frozen Phase B0 transition splits...")
    split_payloads = {}
    crowding = tuple(int(value) for value in config["renderer"]["transition_crowding"])
    for split_name, definition in config["development"]["transition_splits"].items():
        split_payloads[split_name] = build_transition_payload(
            samples=int(definition["samples"]),
            seed=int(definition["seed"]),
            crowding_levels=crowding,
            no_op_fraction=float(config["objectives"]["no_op_consistency"]["transition_fraction"]),
        )
        (data_root / f"{split_name}_transitions.json").write_text(
            json.dumps(
                {
                    "split": split_name,
                    "seed": int(definition["seed"]),
                    "samples": split_payloads[split_name].size,
                    "no_op_samples": int(split_payloads[split_name].no_op.sum().item()),
                    "fingerprints": [
                        item.fingerprint for item in split_payloads[split_name].examples
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("Generating frozen Phase B0 planner-supervision sets...")
    planner_train = build_planner_payload(config, "planner_supervision_train")
    planner_validation = build_planner_payload(config, "planner_supervision_validation")
    progress_mean, progress_std = fit_progress_statistics(planner_train)
    (data_root / "planner_supervision.json").write_text(
        json.dumps(
            {
                "training_candidate_sets": planner_train.candidate_sets,
                "validation_candidate_sets": planner_validation.candidate_sets,
                "candidates_per_set": 32,
                "no_op_index": 0,
                "progress_training_mean": progress_mean,
                "progress_training_std": progress_std,
                "training_records": [
                    {
                        "set_id": item.set_id,
                        "target_seed": item.target_seed,
                        "trajectory_seed": item.trajectory_seed,
                        "candidate_seed": item.candidate_seed,
                        "state_name": item.state_name,
                    }
                    for item in planner_train.records
                ],
                "validation_records": [
                    {
                        "set_id": item.set_id,
                        "target_seed": item.target_seed,
                        "trajectory_seed": item.trajectory_seed,
                        "candidate_seed": item.candidate_seed,
                        "state_name": item.state_name,
                    }
                    for item in planner_validation.records
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    training = config["training"]
    fits = {}
    for variant in config["objectives"]["development_variants"]:
        elapsed = time.perf_counter() - started
        remaining_hours = max(
            1e-6,
            (float(training["development_wall_clock_cap_hours"]) * 3600.0 - elapsed)
            / 3600.0,
        )
        print(f"Training Phase B0 variant: {variant}")
        fits[variant] = train_phase_b_variant(
            variant,
            split_payloads["train"],
            split_payloads["validation"],
            planner_train,
            planner_validation,
            progress_mean=progress_mean,
            progress_std=progress_std,
            seed=int(training["development_model_seed"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            batch_size=int(training["batch_size"]),
            maximum_epochs=int(training["maximum_epochs"]),
            patience=int(training["patience"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            wall_clock_cap_hours=remaining_hours,
        )
        pd.DataFrame(fits[variant].history).to_csv(
            paths.incomplete / f"training_history_{variant}.csv", index=False
        )

    checkpoints: dict[str, dict[str, Any]] = {}
    for variant, fit in fits.items():
        freeze_phase_b_model(fit.model)
        checkpoint, digest = save_phase_b_checkpoint(
            fit,
            checkpoint_root / f"{variant}_seed{fit.seed}.pt",
            progress_mean=progress_mean,
            progress_std=progress_std,
        )
        checkpoints[variant] = {
            "path": str(checkpoint),
            "state_sha256": digest,
            "best_epoch": fit.best_epoch,
            "best_validation_loss": fit.best_validation_loss,
            "wall_clock_seconds": fit.wall_clock_seconds,
            "compute_cap_reached": fit.compute_cap_reached,
        }

    prediction_model = fits["joint_prediction_only"].model
    progress_model = fits["joint_prediction_progress"].model
    diagnostic = split_payloads["diagnostic_test"]
    representation = {
        variant: feature_statistics(fit.model, diagnostic)
        for variant, fit in fits.items()
    }
    retrieval = {
        variant: four_way_retrieval(fit.model, diagnostic)
        for variant, fit in fits.items()
    }
    candidate_rows = planner_candidate_metrics(
        prediction_model, planner_validation, "prediction"
    ) + planner_candidate_metrics(progress_model, planner_validation, "progress")
    pd.DataFrame(candidate_rows).to_csv(
        paths.incomplete / "planner_validation_candidate_metrics.csv", index=False
    )
    (paths.incomplete / "representation_and_retrieval.json").write_text(
        json.dumps({"representation": representation, "retrieval": retrieval}, indent=2),
        encoding="utf-8",
    )

    print("Loading frozen historical comparators after authorization guard...")
    closed = load_latent_planner_config("configs/latent-planner-2026-08-23.json")
    autoencoder, statistics = load_task_latent_resources(closed)
    ensembles = load_latent_predictor_ensembles(closed)
    archived_models = [item.model for item in ensembles["mse_only"]]
    pixel_model, pixel_metadata = load_pixel_checkpoint(
        config["closed_comparators"]["learned_pixel_predictor"]["path"], device="cpu"
    )
    pixel_digest = state_dict_sha256(pixel_model)
    if pixel_digest != config["closed_comparators"]["learned_pixel_predictor"]["state_sha256"]:
        raise RuntimeError("Frozen pixel comparator hash changed.")
    for parameter in pixel_model.parameters():
        parameter.requires_grad_(False)

    horizon = config["development"]["long_horizon"]
    proposal = _proposal(int(horizon["candidates_per_step"]))
    summary_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for target_index, (target_seed, planner_seed) in enumerate(
        zip(horizon["target_seeds"], horizon["planner_seeds"], strict=True), start=1
    ):
        print(f"Running Phase B0 long-horizon target {target_index}/3...")
        target_id = f"target_{target_index:02d}"
        target_dir = targets_root / target_id
        target_dir.mkdir()
        target = random_base_canvas(
            64,
            int(config["renderer"]["target_strokes"]),
            np.random.default_rng(int(target_seed)),
        )
        target.save(target_dir / "target.png")
        exact, exact_time = _timed(
            run_planner,
            target,
            "exact",
            steps=int(horizon["maximum_steps"]),
            seed=int(planner_seed),
            proposal_config=proposal,
            capture_frames=True,
        )
        learned, learned_time = _timed(
            run_learned_planner,
            target,
            pixel_model,
            steps=int(horizon["maximum_steps"]),
            seed=int(planner_seed),
            proposal_config=proposal,
            prediction_batch_size=32,
            device="cpu",
            capture_frames=True,
        )
        archived, archived_time = _timed(
            run_selected_score_planner,
            target,
            autoencoder,
            statistics,
            archived_models,
            maximum_steps=int(horizon["maximum_steps"]),
            seed=int(planner_seed),
            proposal_config=proposal,
            prediction_batch_size=32,
            allow_no_op=False,
            no_op_margin=0.0,
            capture_frames=True,
        )
        prediction, prediction_time = _timed(
            run_phase_b_planner,
            target,
            prediction_model,
            mode="prediction",
            maximum_steps=int(horizon["maximum_steps"]),
            seed=int(planner_seed),
            proposal_config=proposal,
            prediction_batch_size=32,
            allow_no_op=False,
            capture_frames=True,
        )
        progress_forced, progress_forced_time = _timed(
            run_phase_b_planner,
            target,
            progress_model,
            mode="progress",
            maximum_steps=int(horizon["maximum_steps"]),
            seed=int(planner_seed),
            proposal_config=proposal,
            prediction_batch_size=32,
            allow_no_op=False,
            capture_frames=True,
        )
        progress_no_op, progress_no_op_time = _timed(
            run_phase_b_planner,
            target,
            progress_model,
            mode="progress",
            maximum_steps=int(horizon["maximum_steps"]),
            seed=int(planner_seed),
            proposal_config=proposal,
            prediction_batch_size=32,
            allow_no_op=True,
            capture_frames=True,
        )
        runs = (
            (exact, "exact_pixel", exact_time),
            (learned, "learned_pixel", learned_time),
            (archived, "archived_mse_l1_forced", archived_time),
            (prediction, prediction.method, prediction_time),
            (progress_forced, progress_forced.method, progress_forced_time),
            (progress_no_op, progress_no_op.method, progress_no_op_time),
        )
        if tuple(method for _, method, _ in runs) != LONG_HORIZON_METHODS:
            raise RuntimeError("Phase B0 long-horizon method order changed.")
        for run, method, seconds in runs:
            summary_rows.append(
                {
                    "target_id": target_id,
                    "target_seed": int(target_seed),
                    "planner_seed": int(planner_seed),
                    **_summary_row(
                        run,
                        method,
                        seconds,
                        int(horizon["maximum_steps"]),
                        int(horizon["candidates_per_step"]),
                    ),
                }
            )
            for item in run.steps:
                step_rows.append(
                    {
                        "target_id": target_id,
                        "method": method,
                        **asdict(item),
                    }
                )
            _save_run(run, method, target_dir)

    summary = pd.DataFrame(summary_rows)
    if len(summary) != len(horizon["target_seeds"]) * len(LONG_HORIZON_METHODS):
        raise RuntimeError("Phase B0 long-horizon summary row count changed.")
    summary.to_csv(paths.incomplete / "long_horizon_per_target.csv", index=False)
    pd.DataFrame(step_rows).to_csv(paths.incomplete / "long_horizon_steps.csv", index=False)
    aggregate = summary.groupby("method", sort=False).agg(
        mean_initial_mse=("initial_mse", "mean"),
        mean_final_mse=("final_mse", "mean"),
        mean_best_mse=("best_mse", "mean"),
        mean_executed_steps=("executed_steps", "mean"),
        stop_rate=("stopped_early", "mean"),
        premature_stop_rate=("premature_stop", "mean"),
        mean_exact_regret=("mean_exact_regret", "mean"),
    ).reset_index()
    aggregate.to_csv(paths.incomplete / "long_horizon_aggregate.csv", index=False)
    means = aggregate.set_index("method")
    archived_final = float(means.loc["archived_mse_l1_forced", "mean_final_mse"])
    progress_final = float(means.loc["joint_prediction_progress_no_op", "mean_final_mse"])
    prediction_final = float(means.loc["joint_prediction_only_forced", "mean_final_mse"])
    exact_final = float(means.loc["exact_pixel", "mean_final_mse"])
    archived_regret = float(means.loc["archived_mse_l1_forced", "mean_exact_regret"])
    progress_regret = float(means.loc["joint_prediction_progress_forced", "mean_exact_regret"])
    regret_reduction = 1.0 - progress_regret / max(archived_regret, 1e-12)
    final_reduction = 1.0 - progress_final / max(archived_final, 1e-12)
    no_op_rows = summary[summary["method"] == "joint_prediction_progress_no_op"]
    total_elapsed = time.perf_counter() - started
    eligibility = config["development_eligibility"]
    progress_stats = representation["joint_prediction_progress"]
    fingerprints = {
        name: {item.fingerprint for item in payload.examples}
        for name, payload in split_payloads.items()
    }
    splits_disjoint = not (
        fingerprints["train"] & fingerprints["validation"]
        or fingerprints["train"] & fingerprints["diagnostic_test"]
        or fingerprints["validation"] & fingerprints["diagnostic_test"]
    )
    implementation_integrity = bool(
        splits_disjoint
        and all(len(item["state_sha256"]) == 64 for item in checkpoints.values())
        and pixel_metadata.model_seed == 11
        and model_state_sha256(prediction_model) == checkpoints["joint_prediction_only"]["state_sha256"]
        and model_state_sha256(progress_model) == checkpoints["joint_prediction_progress"]["state_sha256"]
    )
    criteria = {
        "implementation_integrity": implementation_integrity,
        "historical_artifacts_unchanged": True,
        "representation_noncollapse_each_scale": all(
            progress_stats[scale]["mean_channel_std"]
            >= float(eligibility["minimum_mean_channel_std_each_scale"])
            for scale in ("32", "16")
        ),
        "diagnostic_four_way_retrieval": retrieval["joint_prediction_progress"]["top1_accuracy"]
        >= float(eligibility["minimum_diagnostic_four_way_retrieval"]),
        "minimum_128_way_regret_reduction_vs_archived_mse_l1": regret_reduction
        >= float(eligibility["minimum_mean_128_way_regret_reduction_vs_archived_mse_l1"]),
        "no_op_improves_every_target_from_blank": bool(
            (no_op_rows["final_mse"] < no_op_rows["initial_mse"]).all()
        ),
        "minimum_mean_final_mse_reduction_vs_archived_mse_l1": final_reduction
        >= float(eligibility["minimum_mean_final_mse_reduction_vs_archived_mse_l1"]),
        "no_op_no_worse_than_joint_prediction_only_forced": progress_final <= prediction_final,
        "maximum_mean_final_mse_ratio_to_exact_pixel": progress_final / max(exact_final, 1e-12)
        <= float(eligibility["maximum_mean_final_mse_ratio_to_exact_pixel"]),
        "maximum_premature_stop_rate": float(no_op_rows["premature_stop"].mean())
        <= float(eligibility["maximum_premature_stop_rate"]),
        "compute_cap": total_elapsed
        <= float(training["development_wall_clock_cap_hours"]) * 3600.0,
    }
    eligible = bool(all(criteria.values()))
    decision = {
        "status": "eligible_for_formal_protocol" if eligible else "not_eligible",
        "criteria_passed": criteria,
        "mean_128_way_regret_reduction_vs_archived_mse_l1": regret_reduction,
        "mean_final_mse_reduction_vs_archived_mse_l1": final_reduction,
        "mean_final_mse_ratio_to_exact_pixel": progress_final / max(exact_final, 1e-12),
        "development_completed": True,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
    }
    (paths.incomplete / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    run_config = {
        "status": "phase_b0_development_complete",
        "single_run": True,
        "config_status": config["status"],
        "checkpoints": checkpoints,
        "progress_training_mean": progress_mean,
        "progress_training_std": progress_std,
        "representation": representation,
        "retrieval": retrieval,
        "pixel_predictor_state_sha256": pixel_digest,
        "implementation_integrity_passed": implementation_integrity,
        "transition_splits_disjoint": splits_disjoint,
        "total_wall_clock_seconds": total_elapsed,
        "models_trained": True,
        "closed_targets_reused": False,
        "historical_results_unchanged": True,
        "decision": decision,
    }
    (paths.incomplete / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    paths.incomplete.rename(paths.final)
    print("\nPhase B0 development decision\n")
    print(json.dumps(decision, indent=2))
    print(f"\nSaved artifacts to: {paths.final.resolve()}")
    print("Do not rerun or tune against this development result.")


if __name__ == "__main__":
    main()
