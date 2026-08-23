"""Guards, validation, aggregation, and decisions for controlled latent planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .latent_smoke import SMOKE_METHODS


CONTROLLED_METHODS = SMOKE_METHODS
CONTROLLED_TARGET_SEEDS = (
    20261211,
    20261212,
    20261213,
    20261214,
    20261215,
    20261216,
)
CONTROLLED_PLANNER_SEEDS = (
    20261221,
    20261222,
    20261223,
    20261224,
    20261225,
    20261226,
)


@dataclass(frozen=True)
class ControlledOutputPaths:
    final: Path
    incomplete: Path


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def controlled_output_paths(config: Mapping[str, Any]) -> ControlledOutputPaths:
    controlled = _mapping(config.get("controlled"), "controlled")
    final = Path(str(controlled.get("output_dir", "")))
    if not final.name:
        raise ValueError("Controlled output directory is invalid.")
    return ControlledOutputPaths(
        final=final,
        incomplete=final.with_name(final.name + ".incomplete"),
    )


def require_controlled_outputs_absent(paths: ControlledOutputPaths) -> None:
    """Refuse completed output and preserve any partial controlled evidence."""

    if paths.final.exists():
        raise FileExistsError(f"Controlled output already exists: {paths.final}")
    if paths.incomplete.exists():
        raise FileExistsError(
            f"Incomplete controlled output exists: {paths.incomplete}. "
            "Preserve and review it before any retry."
        )


def require_controlled_authorized(config: Mapping[str, Any]) -> None:
    """Stop before model loading or reserved-target generation."""

    controlled = config.get("controlled")
    if not isinstance(controlled, Mapping) or controlled.get("authorized") is not True:
        raise PermissionError(
            "The latent-planner controlled comparison is not authorized. "
            "No models were loaded and no controlled target was generated."
        )


def validate_controlled_runner_request(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the guarded runner while every controlled seed stays untouched."""

    if config.get("status") != "smoke_complete_controlled_unauthorized":
        raise ValueError("The completed smoke must be closed before controlled validation.")
    planner = _mapping(config.get("planner"), "planner")
    smoke = _mapping(config.get("smoke"), "smoke")
    controlled = _mapping(config.get("controlled"), "controlled")
    criteria = _mapping(config.get("success_criteria"), "success_criteria")
    if tuple(planner.get("methods", ())) != CONTROLLED_METHODS:
        raise ValueError("Controlled methods changed from the frozen protocol.")
    if smoke.get("authorized") is not False:
        raise ValueError("The completed smoke must remain closed.")
    if controlled.get("authorized") is not False:
        raise ValueError("Validation-only mode requires controlled authorization false.")
    if tuple(controlled.get("target_seeds", ())) != CONTROLLED_TARGET_SEEDS:
        raise ValueError("Controlled target seeds changed.")
    if tuple(controlled.get("planner_seeds", ())) != CONTROLLED_PLANNER_SEEDS:
        raise ValueError("Controlled planner seeds changed.")
    if controlled.get("single_run") is not True:
        raise ValueError("Controlled comparison must remain single-run.")
    if planner.get("steps") != 100 or planner.get("candidates_per_step") != 128:
        raise ValueError("Controlled planner budget changed.")
    if planner.get("prediction_batch_size") != 32:
        raise ValueError("Controlled prediction batch size changed.")
    if dict(criteria) != {
        "latent_ranking_improves_every_target_from_initial": True,
        "minimum_mean_final_mse_reduction_vs_random": 0.2,
        "maximum_mean_final_mse_ratio_to_exact_pixel": 1.5,
        "implementation_integrity_required": True,
        "outperform_learned_pixel_required": False,
    }:
        raise ValueError("Controlled success criteria changed.")
    paths = controlled_output_paths(config)
    require_controlled_outputs_absent(paths)
    return {
        "status": "latent_planner_controlled_runner_valid_unauthorized",
        "config_status": config["status"],
        "smoke_completed_and_closed": True,
        "methods": list(CONTROLLED_METHODS),
        "target_seeds_reserved": list(CONTROLLED_TARGET_SEEDS),
        "planner_seeds_reserved": list(CONTROLLED_PLANNER_SEEDS),
        "target_count": len(CONTROLLED_TARGET_SEEDS),
        "steps_per_target": planner["steps"],
        "candidates_per_step": planner["candidates_per_step"],
        "smoke_authorized": False,
        "controlled_authorized": False,
        "controlled_output_dir_available": True,
        "controlled_incomplete_dir_available": True,
        "models_loaded": False,
        "controlled_targets_generated": False,
        "controlled_planner_data_generated": False,
        "models_trained_or_finetuned": False,
        "criteria_frozen_before_controlled_data": True,
        "historical_results_unchanged": True,
    }


def validate_controlled_summary(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """Validate complete method-aware metrics before aggregation or decision."""

    planner = _mapping(config.get("planner"), "planner")
    expected_rows = len(CONTROLLED_TARGET_SEEDS) * len(CONTROLLED_METHODS)
    if len(summary) != expected_rows:
        raise RuntimeError("Controlled summary row count is incorrect.")
    expected_ids = [f"target_{index:02d}" for index in range(1, 7)]
    for index, (target_id, target_seed, planner_seed) in enumerate(
        zip(expected_ids, CONTROLLED_TARGET_SEEDS, CONTROLLED_PLANNER_SEEDS, strict=True),
        start=1,
    ):
        subset = summary.loc[summary["target_id"] == target_id]
        if tuple(subset["method"]) != CONTROLLED_METHODS:
            raise RuntimeError(f"Method order changed for controlled target {index}.")
        if set(subset["target_seed"]) != {target_seed}:
            raise RuntimeError(f"Target seed changed for controlled target {index}.")
        if set(subset["planner_seed"]) != {planner_seed}:
            raise RuntimeError(f"Planner seed changed for controlled target {index}.")
        initial = subset["initial_mse"].to_numpy(dtype=np.float64)
        if float(np.ptp(initial)) > 1e-12:
            raise RuntimeError(f"Initial canvases differ for controlled target {index}.")
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
    if not bool(np.isfinite(summary[shared].to_numpy(dtype=np.float64)).all()):
        raise RuntimeError("Controlled shared summary contains a non-finite value.")
    if set(summary["steps"]) != {planner["steps"]}:
        raise RuntimeError("Controlled step count changed.")
    if set(summary["candidates_per_step"]) != {planner["candidates_per_step"]}:
        raise RuntimeError("Controlled candidate count changed.")
    if not bool(((summary["best_step"] >= 0) & (summary["best_step"] <= planner["steps"])).all()):
        raise RuntimeError("A controlled best-step value is out of range.")

    learned = summary["method"].isin(("learned_pixel", "latent_mse", "latent_ranking"))
    diagnostics = [
        "exact_top1_rate",
        "exact_top5_rate",
        "mean_exact_rank",
        "mean_exact_regret",
        "max_exact_regret",
    ]
    if not bool(
        np.isfinite(summary.loc[learned, diagnostics].to_numpy(dtype=np.float64)).all()
    ):
        raise RuntimeError("Controlled learned-method diagnostics are non-finite.")
    latent = summary["method"].isin(("latent_mse", "latent_ranking"))
    if not bool(
        np.isfinite(
            summary.loc[latent, ["mean_score_exact_spearman"]].to_numpy(
                dtype=np.float64
            )
        ).all()
    ):
        raise RuntimeError("Controlled latent correlations are non-finite.")


def aggregate_controlled_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the six frozen targets without selecting or tuning a model."""

    rows: list[dict[str, Any]] = []
    for method in CONTROLLED_METHODS:
        subset = summary.loc[summary["method"] == method]
        row: dict[str, Any] = {
            "method": method,
            "targets": int(len(subset)),
            "mean_initial_mse": float(subset["initial_mse"].mean()),
            "mean_final_mse": float(subset["final_mse"].mean()),
            "std_final_mse": float(subset["final_mse"].std(ddof=1)),
            "mean_best_mse": float(subset["best_mse"].mean()),
            "mean_best_step": float(subset["best_step"].mean()),
            "mean_final_mae": float(subset["final_mae"].mean()),
            "mean_relative_final_mse_improvement": float(
                subset["relative_final_mse_improvement"].mean()
            ),
            "mean_improved_steps": float(subset["improved_steps"].mean()),
            "mean_elapsed_seconds": float(subset["elapsed_seconds"].mean()),
            "exact_top1_rate": None,
            "exact_top5_rate": None,
            "mean_exact_rank": None,
            "mean_exact_regret": None,
            "max_exact_regret": None,
            "mean_score_exact_spearman": None,
        }
        if method in ("learned_pixel", "latent_mse", "latent_ranking"):
            row.update(
                {
                    "exact_top1_rate": float(subset["exact_top1_rate"].mean()),
                    "exact_top5_rate": float(subset["exact_top5_rate"].mean()),
                    "mean_exact_rank": float(subset["mean_exact_rank"].mean()),
                    "mean_exact_regret": float(subset["mean_exact_regret"].mean()),
                    "max_exact_regret": float(subset["max_exact_regret"].max()),
                }
            )
        if method in ("latent_mse", "latent_ranking"):
            row["mean_score_exact_spearman"] = float(
                subset["mean_score_exact_spearman"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_controlled_decision(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    implementation_integrity_passed: bool,
) -> dict[str, Any]:
    """Apply only the criteria frozen before controlled target generation."""

    validate_controlled_summary(summary, config)
    criteria = _mapping(config.get("success_criteria"), "success_criteria")
    means = summary.groupby("method")["final_mse"].mean()
    random_mean = float(means.loc["random"])
    exact_mean = float(means.loc["exact_pixel"])
    learned_mean = float(means.loc["learned_pixel"])
    mse_mean = float(means.loc["latent_mse"])
    ranking_mean = float(means.loc["latent_ranking"])
    ranking_rows = summary.loc[summary["method"] == "latent_ranking"]
    ranking_improves_all = bool(
        (ranking_rows["final_mse"] < ranking_rows["initial_mse"]).all()
    )
    reduction_vs_random = 1.0 - ranking_mean / max(random_mean, 1e-12)
    ratio_to_exact = ranking_mean / max(exact_mean, 1e-12)
    criteria_passed = {
        "latent_ranking_improves_every_target": ranking_improves_all,
        "minimum_mean_reduction_vs_random": bool(
            reduction_vs_random
            >= float(criteria["minimum_mean_final_mse_reduction_vs_random"])
        ),
        "maximum_mean_ratio_to_exact_pixel": bool(
            ratio_to_exact
            <= float(criteria["maximum_mean_final_mse_ratio_to_exact_pixel"])
        ),
        "implementation_integrity": bool(implementation_integrity_passed),
    }
    success = bool(all(criteria_passed.values()))
    return {
        "status": "success" if success else "fail",
        "controlled_comparison_completed": True,
        "implementation_integrity_passed": bool(implementation_integrity_passed),
        "criteria_frozen_before_controlled_data": True,
        "criteria_passed": criteria_passed,
        "latent_ranking_improved_every_target": ranking_improves_all,
        "mean_random_final_mse": random_mean,
        "mean_exact_pixel_final_mse": exact_mean,
        "mean_learned_pixel_final_mse": learned_mean,
        "mean_latent_mse_final_mse": mse_mean,
        "mean_latent_ranking_final_mse": ranking_mean,
        "latent_ranking_mean_final_mse_reduction_vs_random": reduction_vs_random,
        "latent_ranking_mean_final_mse_ratio_to_exact_pixel": ratio_to_exact,
        "latent_ranking_mean_final_mse_ratio_to_latent_mse": (
            ranking_mean / max(mse_mean, 1e-12)
        ),
        "latent_ranking_mean_final_mse_ratio_to_learned_pixel": (
            ranking_mean / max(learned_mean, 1e-12)
        ),
        "outperform_learned_pixel_required": False,
        "models_trained_or_finetuned": False,
        "smoke_result_unchanged": True,
        "formal_result_unchanged": True,
    }
