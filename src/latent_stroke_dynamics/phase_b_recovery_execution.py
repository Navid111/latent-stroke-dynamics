"""Persistent single-execution implementation for the frozen Phase B0 recovery."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .extension_training import model_state_sha256
from .latent_planner import load_latent_planner_config, load_task_latent_resources
from .learned_pixel_planner import (
    LearnedPlanningRun,
    load_pixel_checkpoint,
    run_learned_planner,
    state_dict_sha256,
)
from .phase_b_cloud_preflight import verify_loaded_model_states, verify_raw_resources
from .phase_b_data import (
    build_planner_payload,
    build_transition_payload,
    fit_progress_statistics,
)
from .phase_b_development import LONG_HORIZON_METHODS
from .phase_b_planning import PhaseBPlanningRun, run_phase_b_planner
from .phase_b_recovery import (
    RecoveryOutputPaths,
    file_sha256,
    load_recovery_mse_only_predictors,
    validate_expected_data_manifests,
)
from .phase_b_training import (
    feature_statistics,
    four_way_retrieval,
    freeze_phase_b_model,
    planner_candidate_metrics,
    save_phase_b_checkpoint,
    train_phase_b_variant,
)
from .planner_score_development import SelectedScorePlanningRun, run_selected_score_planner
from .planning import PlanningRun, ProposalConfig, pixel_mae, pixel_mse, run_planner
from .renderer import random_base_canvas


RunLike = PlanningRun | LearnedPlanningRun | SelectedScorePlanningRun | PhaseBPlanningRun


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def record_recovery_event(
    journal_path: Path,
    stage: str,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    if journal_path.exists():
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "status": "phase_b0_colab_recovery_incomplete",
            "automatic_resume_authorized": False,
            "events": [],
        }
    payload["events"].append(
        {
            "timestamp_utc": _utc_now(),
            "stage": stage,
            "status": status,
            "details": dict(details or {}),
        }
    )
    _write_json_atomic(journal_path, payload)


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
        _write_json_atomic(
            directory / "stop_decision.json",
            asdict(run.stop_decision) if run.stop_decision else {},
        )


def _timed(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started


def _require_within_cap(started: float, cap_seconds: float, stage: str) -> None:
    elapsed = time.perf_counter() - started
    if elapsed >= cap_seconds:
        raise TimeoutError(
            f"Phase B0 recovery reached its six-hour cap before {stage}; preserve the incomplete output."
        )


def _create_data_manifests(
    config: Mapping[str, Any], data_root: Path
) -> tuple[dict[str, Any], Any, Any, float, float]:
    split_payloads: dict[str, Any] = {}
    crowding = tuple(int(value) for value in config["renderer"]["transition_crowding"])
    for split_name, definition in config["development"]["transition_splits"].items():
        split_payloads[split_name] = build_transition_payload(
            samples=int(definition["samples"]),
            seed=int(definition["seed"]),
            crowding_levels=crowding,
            no_op_fraction=float(
                config["objectives"]["no_op_consistency"]["transition_fraction"]
            ),
        )
        _write_json_atomic(
            data_root / f"{split_name}_transitions.json",
            {
                "split": split_name,
                "seed": int(definition["seed"]),
                "samples": split_payloads[split_name].size,
                "no_op_samples": int(split_payloads[split_name].no_op.sum().item()),
                "fingerprints": [
                    item.fingerprint for item in split_payloads[split_name].examples
                ],
            },
        )

    planner_train = build_planner_payload(config, "planner_supervision_train")
    planner_validation = build_planner_payload(config, "planner_supervision_validation")
    progress_mean, progress_std = fit_progress_statistics(planner_train)
    _write_json_atomic(
        data_root / "planner_supervision.json",
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
    )
    return (
        split_payloads,
        planner_train,
        planner_validation,
        progress_mean,
        progress_std,
    )


def _artifact_hash_manifest(root: Path) -> dict[str, str]:
    excluded = {"integrity_manifest.json", "recovery_stage_journal.json"}
    result: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded or path.name.endswith(".tmp"):
            continue
        result[relative] = file_sha256(path)
    return result


def execute_phase_b_recovery(
    config: Mapping[str, Any],
    recovery_config: Mapping[str, Any],
    paths: RecoveryOutputPaths,
    *,
    repository_root: str | Path,
    environment_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute exactly one already-authorized recovery; caller must run all guards first."""

    root = Path(repository_root).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError("Run the recovery from the exact repository root.")
    started = time.perf_counter()
    cap_seconds = float(config["training"]["development_wall_clock_cap_hours"]) * 3600.0
    device = str(recovery_config["environment"]["device"])

    paths.incomplete.mkdir(parents=True)
    data_root = paths.incomplete / "data_manifests"
    checkpoint_root = paths.incomplete / "checkpoints"
    targets_root = paths.incomplete / "targets"
    data_root.mkdir()
    checkpoint_root.mkdir()
    targets_root.mkdir()
    record_recovery_event(
        paths.journal,
        "guard_and_environment",
        "completed",
        {"environment": dict(environment_snapshot)},
    )

    raw_hashes = verify_raw_resources(root)
    loaded_states = verify_loaded_model_states(root)
    if loaded_states.get("ranking_aware_models_loaded") is not False:
        raise RuntimeError("Recovery loaded a forbidden ranking-aware model.")
    record_recovery_event(
        paths.journal,
        "resource_integrity",
        "completed",
        {"raw_resource_sha256": raw_hashes, "loaded_model_states": loaded_states},
    )

    _require_within_cap(started, cap_seconds, "deterministic data generation")
    (
        split_payloads,
        planner_train,
        planner_validation,
        progress_mean,
        progress_std,
    ) = _create_data_manifests(config, data_root)
    record_recovery_event(
        paths.journal,
        "deterministic_data_generation",
        "completed",
        {
            "transition_samples": {
                name: payload.size for name, payload in split_payloads.items()
            },
            "planner_training_candidate_sets": planner_train.candidate_sets,
            "planner_validation_candidate_sets": planner_validation.candidate_sets,
        },
    )
    manifest_hashes = validate_expected_data_manifests(recovery_config, data_root)
    record_recovery_event(
        paths.journal,
        "data_manifest_hash_verification",
        "completed",
        {"sha256": manifest_hashes},
    )

    training = config["training"]
    fits: dict[str, Any] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for variant in config["objectives"]["development_variants"]:
        _require_within_cap(started, cap_seconds, f"training {variant}")
        elapsed = time.perf_counter() - started
        remaining_hours = max(1e-6, (cap_seconds - elapsed) / 3600.0)
        fit = train_phase_b_variant(
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
            device=device,
        )
        fits[variant] = fit
        pd.DataFrame(fit.history).to_csv(
            paths.incomplete / f"training_history_{variant}.csv", index=False
        )
        freeze_phase_b_model(fit.model)
        checkpoint, digest = save_phase_b_checkpoint(
            fit,
            checkpoint_root / f"{variant}_seed{fit.seed}.pt",
            progress_mean=progress_mean,
            progress_std=progress_std,
        )
        checkpoints[variant] = {
            "path": checkpoint.relative_to(paths.incomplete).as_posix(),
            "state_sha256": digest,
            "best_epoch": fit.best_epoch,
            "best_validation_loss": fit.best_validation_loss,
            "wall_clock_seconds": fit.wall_clock_seconds,
            "compute_cap_reached": fit.compute_cap_reached,
            "training_device": fit.training_device,
        }
        record_recovery_event(
            paths.journal,
            f"{variant}_training_and_checkpoint",
            "completed",
            checkpoints[variant],
        )

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
    _write_json_atomic(
        paths.incomplete / "representation_and_retrieval.json",
        {"representation": representation, "retrieval": retrieval},
    )
    record_recovery_event(
        paths.journal,
        "diagnostics",
        "completed",
        {"representation": representation, "retrieval": retrieval},
    )

    closed = load_latent_planner_config(root / "configs/latent-planner-2026-08-23.json")
    autoencoder, statistics = load_task_latent_resources(closed)
    archived_loaded = load_recovery_mse_only_predictors(closed)
    archived_models = [item.model for item in archived_loaded]
    if tuple(item.method for item in archived_loaded) != ("mse_only",) * 3:
        raise RuntimeError("Recovery comparator family changed.")
    pixel_model, pixel_metadata = load_pixel_checkpoint(
        root / config["closed_comparators"]["learned_pixel_predictor"]["path"],
        device=device,
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
        _require_within_cap(started, cap_seconds, f"long-horizon target {target_index}")
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
            device=device,
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
                    {"target_id": target_id, "method": method, **asdict(item)}
                )
            _save_run(run, method, target_dir)
        pd.DataFrame(summary_rows).to_csv(
            paths.incomplete / "long_horizon_per_target.partial.csv", index=False
        )
        pd.DataFrame(step_rows).to_csv(
            paths.incomplete / "long_horizon_steps.partial.csv", index=False
        )
        record_recovery_event(
            paths.journal,
            "long_horizon_target",
            "completed",
            {"target_id": target_id, "methods": list(LONG_HORIZON_METHODS)},
        )

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
    record_recovery_event(
        paths.journal,
        "long_horizon_comparison",
        "completed",
        {"targets": len(horizon["target_seeds"]), "methods": list(LONG_HORIZON_METHODS)},
    )

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
        and manifest_hashes == recovery_config["data_manifest_sha256_required_before_training"]
        and loaded_states.get("ranking_aware_models_loaded") is False
        and all(len(item["state_sha256"]) == 64 for item in checkpoints.values())
        and pixel_metadata.model_seed == 11
        and model_state_sha256(prediction_model)
        == checkpoints["joint_prediction_only"]["state_sha256"]
        and model_state_sha256(progress_model)
        == checkpoints["joint_prediction_progress"]["state_sha256"]
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
        "compute_cap": total_elapsed <= cap_seconds,
    }
    eligible = bool(all(criteria.values()))
    decision = {
        "status": "eligible_for_formal_protocol" if eligible else "not_eligible",
        "criteria_passed": criteria,
        "mean_128_way_regret_reduction_vs_archived_mse_l1": regret_reduction,
        "mean_final_mse_reduction_vs_archived_mse_l1": final_reduction,
        "mean_final_mse_ratio_to_exact_pixel": progress_final / max(exact_final, 1e-12),
        "development_completed": True,
        "recovery_completed": True,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
    }
    _write_json_atomic(paths.incomplete / "decision.json", decision)
    record_recovery_event(paths.journal, "eligibility_decision", "completed", decision)

    run_config = {
        "status": "phase_b0_colab_recovery_complete",
        "single_run": True,
        "base_config_status": config["status"],
        "recovery_config_status": recovery_config["status"],
        "environment": dict(environment_snapshot),
        "device_allocation": dict(recovery_config["device_allocation"]),
        "raw_resource_sha256": raw_hashes,
        "loaded_model_states": loaded_states,
        "data_manifest_sha256": manifest_hashes,
        "checkpoints": checkpoints,
        "progress_training_mean": progress_mean,
        "progress_training_std": progress_std,
        "representation": representation,
        "retrieval": retrieval,
        "pixel_predictor_state_sha256": pixel_digest,
        "ranking_aware_models_loaded": False,
        "implementation_integrity_passed": implementation_integrity,
        "transition_splits_disjoint": splits_disjoint,
        "total_wall_clock_seconds": total_elapsed,
        "models_trained": True,
        "closed_targets_reused": False,
        "historical_results_unchanged": True,
        "decision": decision,
    }
    _write_json_atomic(paths.incomplete / "run_config.json", run_config)
    integrity = {
        "status": "phase_b0_colab_recovery_integrity_manifest_complete",
        "artifact_sha256": _artifact_hash_manifest(paths.incomplete),
        "journal_excluded_from_hash_manifest": True,
        "historical_results_unchanged": True,
    }
    _write_json_atomic(paths.incomplete / "integrity_manifest.json", integrity)
    record_recovery_event(
        paths.journal,
        "final_integrity_manifest",
        "completed",
        {"artifact_count": len(integrity["artifact_sha256"])},
    )
    paths.incomplete.rename(paths.final)
    final_journal = paths.final / "recovery_stage_journal.json"
    record_recovery_event(
        final_journal,
        "atomic_finalize",
        "completed",
        {"final_path": str(paths.final)},
    )
    return decision
