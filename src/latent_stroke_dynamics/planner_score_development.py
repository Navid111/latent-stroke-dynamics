"""Guarded long-horizon development for the frozen planner-score winner."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .latent_planner import encode_task_latents
from .latent_smoke import spearman_rank_correlation
from .planner_score_alignment import (
    exact_candidate_scores,
    predict_candidate_latents,
    validate_closed_resource_references,
)
from .planning import ProposalConfig, pixel_mae, pixel_mse, propose_strokes
from .representation_extension import LatentChannelStatistics, StrokeAutoencoder
from .renderer import Stroke, blank_canvas


DEFAULT_DEVELOPMENT_SELECTION = Path(
    "results/planner-score-alignment/development-selection.json"
)
PLANNER_DEVELOPMENT_METHODS = (
    "exact_pixel",
    "learned_pixel",
    "current_latent_mse_forced",
    "development_selected_score_forced",
    "development_selected_score_no_op",
)
PLANNER_DEVELOPMENT_TARGET_SEEDS = (20270201, 20270202, 20270203)
PLANNER_DEVELOPMENT_SEEDS = (20270211, 20270212, 20270213)
SELECTED_PREDICTOR_FAMILY = "mse_only"
SELECTED_SCORE_NAME = "normalized_latent_l1"
EXPECTED_PIXEL_PATH = "checkpoints/stage3-pixel-mlp-seed11.pt"
EXPECTED_PIXEL_SHA256 = (
    "e32f3612f7a184e4e9b58f95a987551bd25cdb17ff1bf2b6be40fcf5781ea472"
)


@dataclass(frozen=True)
class PlannerDevelopmentOutputPaths:
    final: Path
    incomplete: Path


@dataclass(frozen=True)
class NoOpDecision:
    proposal_round: int
    executed_steps: int
    current_score: float
    best_candidate_score: float
    margin: float


@dataclass(frozen=True)
class SelectedScorePlanningStep:
    step: int
    selected_index: int
    stroke: Stroke
    candidate_count: int
    mse_before: float
    mse_after: float
    mae_after: float
    current_state_score: float
    predicted_selected_score: float
    predicted_score_range: float
    per_model_selected_scores: tuple[float, ...]
    exact_best_candidate_mse: float
    exact_selected_rank: int
    exact_top1: bool
    exact_top5: bool
    exact_regret: float
    score_exact_spearman: float
    improved: bool


@dataclass(frozen=True)
class SelectedScorePlanningRun:
    method: str
    seed: int
    target: Image.Image
    initial_canvas: Image.Image
    final_canvas: Image.Image
    steps: tuple[SelectedScorePlanningStep, ...]
    frames: tuple[Image.Image, ...]
    target_encoding_count: int
    observed_canvas_encoding_count: int
    proposal_rounds_evaluated: int
    stop_decision: NoOpDecision | None


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def expected_development_selection() -> dict[str, Any]:
    return {
        "status": "development_score_audit_complete",
        "selection": {
            "predictor_family": SELECTED_PREDICTOR_FAMILY,
            "score_name": SELECTED_SCORE_NAME,
            "mean_exact_regret": 0.0010517144069821269,
            "exact_top5_rate": 0.5138888888888888,
            "mean_score_exact_spearman": 0.6192424342849114,
            "selection_order": [
                "lowest_mean_exact_regret",
                "highest_exact_top5_rate",
                "highest_mean_score_exact_spearman",
                "fixed_score_simplicity_order",
                "mse_only_before_ranking_aware",
            ],
        },
        "implementation_integrity_passed": True,
        "models_trained_or_finetuned": False,
        "closed_targets_reused": False,
        "historical_results_unchanged": True,
    }


def load_frozen_development_selection(
    path: str | Path = DEFAULT_DEVELOPMENT_SELECTION,
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload != expected_development_selection():
        raise ValueError("Archived development score selection changed.")
    return payload


def validate_planner_development_resources(
    config: Mapping[str, Any],
    closed_config: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, bool]:
    checks = validate_closed_resource_references(config, closed_config)
    if dict(selection) != expected_development_selection():
        raise ValueError("Planner development received the wrong frozen selection.")
    pixel = _mapping(closed_config.get("pixel_predictor"), "pixel_predictor")
    if pixel.get("path") != EXPECTED_PIXEL_PATH:
        raise ValueError("Frozen pixel predictor path changed.")
    if pixel.get("state_sha256") != EXPECTED_PIXEL_SHA256:
        raise ValueError("Frozen pixel predictor SHA-256 changed.")
    return {
        **checks,
        "development_selection_verified": True,
        "pixel_predictor_hash_reference_verified": True,
    }


def planner_development_output_paths(
    config: Mapping[str, Any],
) -> PlannerDevelopmentOutputPaths:
    phase = _mapping(config.get("planner_development"), "planner_development")
    final = Path(str(phase.get("output_dir", "")))
    if not final.name:
        raise ValueError("Planner-development output directory is invalid.")
    return PlannerDevelopmentOutputPaths(
        final=final,
        incomplete=final.with_name(final.name + ".incomplete"),
    )


def require_planner_development_outputs_absent(
    paths: PlannerDevelopmentOutputPaths,
) -> None:
    if paths.final.exists():
        raise FileExistsError(f"Planner-development output exists: {paths.final}")
    if paths.incomplete.exists():
        raise FileExistsError(
            f"Incomplete planner-development output exists: {paths.incomplete}. "
            "Preserve and review it before any retry."
        )


def require_planner_development_authorized(config: Mapping[str, Any]) -> None:
    phase = _mapping(config.get("planner_development"), "planner_development")
    audit = _mapping(config.get("development_score_audit"), "development_score_audit")
    confirmatory = _mapping(config.get("confirmatory_reserved"), "confirmatory_reserved")
    if (
        config.get("status") != "planner_development_authorized_once"
        or phase.get("authorized") is not True
        or audit.get("authorized") is not False
        or confirmatory.get("authorized") is not False
    ):
        raise PermissionError(
            "Planner development is not authorized. No models were loaded and no "
            "planner-development target was generated."
        )


def validate_planner_development_runner_request(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    if config.get("status") != "development_score_audit_complete_planner_unauthorized":
        raise ValueError("The score audit must be closed before planner validation.")
    audit = _mapping(config.get("development_score_audit"), "development_score_audit")
    phase = _mapping(config.get("planner_development"), "planner_development")
    confirmatory = _mapping(config.get("confirmatory_reserved"), "confirmatory_reserved")
    if audit.get("authorized") is not False:
        raise ValueError("The completed score audit must remain unauthorized.")
    if phase.get("authorized") is not False:
        raise ValueError("Validation-only mode requires planner authorization false.")
    if confirmatory.get("authorized") is not False:
        raise ValueError("Confirmatory evaluation must remain unauthorized.")
    if tuple(phase.get("methods", ())) != PLANNER_DEVELOPMENT_METHODS:
        raise ValueError("Planner-development methods changed.")
    if tuple(phase.get("target_seeds", ())) != PLANNER_DEVELOPMENT_TARGET_SEEDS:
        raise ValueError("Planner-development target seeds changed.")
    if tuple(phase.get("planner_seeds", ())) != PLANNER_DEVELOPMENT_SEEDS:
        raise ValueError("Planner-development planner seeds changed.")
    if dict(selection) != expected_development_selection():
        raise ValueError("Planner development did not load the archived winner.")
    if phase.get("maximum_steps") != 100 or phase.get("candidates_per_step") != 128:
        raise ValueError("Planner-development budget changed.")
    if phase.get("prediction_batch_size") != 32:
        raise ValueError("Planner-development prediction batch size changed.")
    if phase.get("no_op_margin") != 0.0:
        raise ValueError("The no-op margin must remain zero.")
    paths = planner_development_output_paths(config)
    require_planner_development_outputs_absent(paths)
    return {
        "status": "planner_score_planner_development_runner_valid_unauthorized",
        "config_status": config["status"],
        "score_audit_completed_and_closed": True,
        "selected_predictor_family": SELECTED_PREDICTOR_FAMILY,
        "selected_score_name": SELECTED_SCORE_NAME,
        "methods": list(PLANNER_DEVELOPMENT_METHODS),
        "target_seeds_reserved": list(PLANNER_DEVELOPMENT_TARGET_SEEDS),
        "planner_seeds_reserved": list(PLANNER_DEVELOPMENT_SEEDS),
        "target_count": len(PLANNER_DEVELOPMENT_TARGET_SEEDS),
        "maximum_steps": phase["maximum_steps"],
        "candidates_per_step": phase["candidates_per_step"],
        "no_op_margin": phase["no_op_margin"],
        "score_audit_authorized": False,
        "planner_development_authorized": False,
        "confirmatory_authorized": False,
        "output_dir_available": True,
        "incomplete_dir_available": True,
        "models_loaded": False,
        "targets_generated": False,
        "planner_data_generated": False,
        "models_trained_or_finetuned": False,
        "closed_targets_reused": False,
        "historical_results_unchanged": True,
    }


@torch.inference_mode()
def normalized_latent_l1_candidate_scores(
    predictors: Sequence[nn.Module],
    current_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    candidates: Sequence[Stroke],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = predict_candidate_latents(
        predictors,
        current_tokens,
        candidates,
        batch_size=batch_size,
    )
    predicted_normalized = F.normalize(predicted.float(), dim=-1)
    target_normalized = F.normalize(target_tokens.float(), dim=-1)
    per_model_tensor = (
        predicted_normalized - target_normalized[None, :, :, :]
    ).abs().mean(dim=(2, 3))
    per_model = per_model_tensor.cpu().numpy().astype(np.float64, copy=False)
    aggregate = per_model.mean(axis=0)
    if per_model.shape != (len(predictors), len(candidates)):
        raise RuntimeError("Selected-score per-model shape is invalid.")
    if not bool(np.isfinite(per_model).all() and np.isfinite(aggregate).all()):
        raise RuntimeError("Selected-score candidate values are non-finite.")
    return aggregate, per_model


def normalized_latent_l1_state_score(
    current_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> float:
    if current_tokens.shape != (1, 256, 32) or target_tokens.shape != (1, 256, 32):
        raise ValueError("State-score tokens must have shape [1, 256, 32].")
    value = (
        F.normalize(current_tokens.float(), dim=-1)
        - F.normalize(target_tokens.float(), dim=-1)
    ).abs().mean()
    result = float(value.item())
    if not np.isfinite(result):
        raise RuntimeError("Current-state score is non-finite.")
    return result


def should_take_no_op(
    current_score: float,
    candidate_scores: np.ndarray,
    *,
    margin: float = 0.0,
) -> bool:
    scores = np.asarray(candidate_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 1:
        raise ValueError("No-op comparison requires a non-empty 1D candidate vector.")
    if margin != 0.0:
        raise ValueError("The frozen no-op margin must remain zero.")
    if not bool(np.isfinite(scores).all() and np.isfinite(current_score)):
        raise ValueError("No-op scores must be finite.")
    return bool(float(current_score) <= float(scores.min()) + margin)


def _planner_step_rng(seed: int, step: int) -> np.random.Generator:
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    return np.random.default_rng(np.random.SeedSequence([seed, step, 0]))


@torch.inference_mode()
def run_selected_score_planner(
    target: Image.Image,
    autoencoder: StrokeAutoencoder,
    statistics: LatentChannelStatistics,
    predictors: Sequence[nn.Module],
    *,
    maximum_steps: int,
    seed: int,
    proposal_config: ProposalConfig,
    prediction_batch_size: int,
    allow_no_op: bool,
    no_op_margin: float = 0.0,
    capture_frames: bool = False,
) -> SelectedScorePlanningRun:
    if target.mode != "L" or target.size != (64, 64):
        raise ValueError("Selected-score planning requires a 64x64 grayscale target.")
    if maximum_steps < 1 or prediction_batch_size < 1:
        raise ValueError("Planner budgets must be positive.")
    if len(predictors) != 3:
        raise ValueError("Selected-score planning requires three frozen predictors.")
    if no_op_margin != 0.0:
        raise ValueError("The frozen no-op margin must remain zero.")
    if any(parameter.requires_grad for parameter in autoencoder.parameters()):
        raise ValueError("Selected-score planning requires a frozen autoencoder.")
    if any(parameter.requires_grad for model in predictors for parameter in model.parameters()):
        raise ValueError("Selected-score planning requires frozen predictors.")

    target_tokens = encode_task_latents(
        autoencoder,
        statistics,
        (target,),
        batch_size=1,
    )
    initial = blank_canvas(64)
    current = initial.copy()
    records: list[SelectedScorePlanningStep] = []
    frames: list[Image.Image] = [current.copy()] if capture_frames else []
    stop_decision: NoOpDecision | None = None
    proposal_rounds = 0

    for proposal_round in range(1, maximum_steps + 1):
        proposal_rounds += 1
        candidates = propose_strokes(
            current,
            target,
            rng=_planner_step_rng(seed, proposal_round),
            config=proposal_config,
        )
        current_tokens = encode_task_latents(
            autoencoder,
            statistics,
            (current,),
            batch_size=1,
        )
        scores, per_model_scores = normalized_latent_l1_candidate_scores(
            predictors,
            current_tokens,
            target_tokens,
            candidates,
            batch_size=prediction_batch_size,
        )
        current_score = normalized_latent_l1_state_score(
            current_tokens,
            target_tokens,
        )
        if allow_no_op and should_take_no_op(
            current_score,
            scores,
            margin=no_op_margin,
        ):
            stop_decision = NoOpDecision(
                proposal_round=proposal_round,
                executed_steps=len(records),
                current_score=current_score,
                best_candidate_score=float(scores.min()),
                margin=no_op_margin,
            )
            break

        selected_index = int(np.argmin(scores))
        exact_canvases, exact_scores = exact_candidate_scores(
            current,
            target,
            candidates,
        )
        next_canvas = exact_canvases[selected_index]
        exact_selected = float(exact_scores[selected_index])
        exact_best = float(exact_scores.min())
        exact_rank = 1 + int(np.sum(exact_scores < exact_selected - 1e-12))
        mse_before = pixel_mse(current, target)
        record = SelectedScorePlanningStep(
            step=len(records) + 1,
            selected_index=selected_index,
            stroke=candidates[selected_index],
            candidate_count=len(candidates),
            mse_before=mse_before,
            mse_after=exact_selected,
            mae_after=pixel_mae(next_canvas, target),
            current_state_score=current_score,
            predicted_selected_score=float(scores[selected_index]),
            predicted_score_range=float(np.ptp(scores)),
            per_model_selected_scores=tuple(
                float(value) for value in per_model_scores[:, selected_index]
            ),
            exact_best_candidate_mse=exact_best,
            exact_selected_rank=exact_rank,
            exact_top1=exact_rank == 1,
            exact_top5=exact_rank <= 5,
            exact_regret=max(0.0, exact_selected - exact_best),
            score_exact_spearman=spearman_rank_correlation(scores, exact_scores),
            improved=exact_selected < mse_before,
        )
        numeric = (
            record.mse_before,
            record.mse_after,
            record.mae_after,
            record.current_state_score,
            record.predicted_selected_score,
            record.predicted_score_range,
            record.exact_best_candidate_mse,
            record.exact_regret,
            record.score_exact_spearman,
            *record.per_model_selected_scores,
        )
        if not bool(np.isfinite(np.asarray(numeric, dtype=np.float64)).all()):
            raise RuntimeError("Selected-score planning produced a non-finite diagnostic.")
        records.append(record)
        current = next_canvas
        if capture_frames:
            frames.append(current.copy())

    method = (
        "development_selected_score_no_op"
        if allow_no_op
        else "development_selected_score_forced"
    )
    return SelectedScorePlanningRun(
        method=method,
        seed=seed,
        target=target.copy(),
        initial_canvas=initial,
        final_canvas=current,
        steps=tuple(records),
        frames=tuple(frames),
        target_encoding_count=1,
        observed_canvas_encoding_count=proposal_rounds,
        proposal_rounds_evaluated=proposal_rounds,
        stop_decision=stop_decision,
    )


def validate_planner_development_summary(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    phase = _mapping(config.get("planner_development"), "planner_development")
    expected_rows = len(PLANNER_DEVELOPMENT_TARGET_SEEDS) * len(
        PLANNER_DEVELOPMENT_METHODS
    )
    if len(summary) != expected_rows:
        raise RuntimeError("Planner-development summary row count is incorrect.")
    for index, (target_seed, planner_seed) in enumerate(
        zip(
            PLANNER_DEVELOPMENT_TARGET_SEEDS,
            PLANNER_DEVELOPMENT_SEEDS,
            strict=True,
        ),
        start=1,
    ):
        target_id = f"target_{index:02d}"
        subset = summary.loc[summary["target_id"] == target_id]
        if tuple(subset["method"]) != PLANNER_DEVELOPMENT_METHODS:
            raise RuntimeError(f"Planner-development method order changed for {target_id}.")
        if set(subset["target_seed"]) != {target_seed}:
            raise RuntimeError(f"Target seed changed for {target_id}.")
        if set(subset["planner_seed"]) != {planner_seed}:
            raise RuntimeError(f"Planner seed changed for {target_id}.")
        if float(np.ptp(subset["initial_mse"].to_numpy(dtype=np.float64))) > 1e-12:
            raise RuntimeError(f"Initial canvases differ for {target_id}.")

    numeric = [
        "maximum_steps",
        "executed_steps",
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
    if not bool(np.isfinite(summary[numeric].to_numpy(dtype=np.float64)).all()):
        raise RuntimeError("Planner-development shared metrics are non-finite.")
    if set(summary["maximum_steps"]) != {phase["maximum_steps"]}:
        raise RuntimeError("Planner-development maximum steps changed.")
    if set(summary["candidates_per_step"]) != {phase["candidates_per_step"]}:
        raise RuntimeError("Planner-development candidate count changed.")

    forced = summary["method"] != "development_selected_score_no_op"
    if not bool((summary.loc[forced, "executed_steps"] == phase["maximum_steps"]).all()):
        raise RuntimeError("A forced-horizon method stopped early.")
    no_op = summary.loc[
        summary["method"] == "development_selected_score_no_op"
    ]
    if not bool(
        (
            (no_op["executed_steps"] >= 0)
            & (no_op["executed_steps"] <= phase["maximum_steps"])
        ).all()
    ):
        raise RuntimeError("No-op executed-step count is invalid.")
    expected_stopped = no_op["executed_steps"] < phase["maximum_steps"]
    if not bool((no_op["stopped_early"] == expected_stopped).all()):
        raise RuntimeError("No-op stop flags are inconsistent.")
    stopped = no_op.loc[no_op["stopped_early"]]
    if len(stopped):
        stop_numeric = stopped[
            ["stop_round", "current_score_at_stop", "best_candidate_score_at_stop"]
        ].to_numpy(dtype=np.float64)
        if not bool(np.isfinite(stop_numeric).all()):
            raise RuntimeError("A no-op stop diagnostic is non-finite.")
        if not bool(
            (stopped["stop_round"] == stopped["executed_steps"] + 1).all()
        ):
            raise RuntimeError("No-op stop round is inconsistent.")
        if not bool(
            (
                stopped["current_score_at_stop"]
                <= stopped["best_candidate_score_at_stop"]
            ).all()
        ):
            raise RuntimeError("A no-op stop violated the frozen zero-margin rule.")

    learned = summary["method"].isin(
        (
            "learned_pixel",
            "current_latent_mse_forced",
            "development_selected_score_forced",
            "development_selected_score_no_op",
        )
    ) & (summary["executed_steps"] > 0)
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
        raise RuntimeError("Planner-development learned diagnostics are non-finite.")
    latent = summary["method"].isin(
        (
            "current_latent_mse_forced",
            "development_selected_score_forced",
            "development_selected_score_no_op",
        )
    ) & (summary["executed_steps"] > 0)
    if not bool(
        np.isfinite(
            summary.loc[latent, ["mean_score_exact_spearman"]].to_numpy(
                dtype=np.float64
            )
        ).all()
    ):
        raise RuntimeError("Planner-development latent correlations are non-finite.")


def aggregate_planner_development_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in PLANNER_DEVELOPMENT_METHODS:
        subset = summary.loc[summary["method"] == method]
        row: dict[str, Any] = {
            "method": method,
            "targets": int(len(subset)),
            "mean_initial_mse": float(subset["initial_mse"].mean()),
            "mean_final_mse": float(subset["final_mse"].mean()),
            "mean_best_mse": float(subset["best_mse"].mean()),
            "mean_best_step": float(subset["best_step"].mean()),
            "mean_executed_steps": float(subset["executed_steps"].mean()),
            "stop_rate": float(subset["stopped_early"].mean()),
            "mean_final_mae": float(subset["final_mae"].mean()),
            "mean_relative_final_mse_improvement": float(
                subset["relative_final_mse_improvement"].mean()
            ),
            "exact_top1_rate": None,
            "exact_top5_rate": None,
            "mean_exact_rank": None,
            "mean_exact_regret": None,
            "max_exact_regret": None,
            "mean_score_exact_spearman": None,
        }
        available = subset.loc[subset["executed_steps"] > 0]
        if method != "exact_pixel" and len(available):
            row.update(
                {
                    "exact_top1_rate": float(available["exact_top1_rate"].mean()),
                    "exact_top5_rate": float(available["exact_top5_rate"].mean()),
                    "mean_exact_rank": float(available["mean_exact_rank"].mean()),
                    "mean_exact_regret": float(available["mean_exact_regret"].mean()),
                    "max_exact_regret": float(available["max_exact_regret"].max()),
                }
            )
        if method in (
            "current_latent_mse_forced",
            "development_selected_score_forced",
            "development_selected_score_no_op",
        ) and len(available):
            row["mean_score_exact_spearman"] = float(
                available["mean_score_exact_spearman"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_planner_development_decision(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    implementation_integrity_passed: bool,
    selected_pair_matches: bool,
) -> dict[str, Any]:
    validate_planner_development_summary(summary, config)
    phase = _mapping(config.get("planner_development"), "planner_development")
    eligibility = _mapping(
        phase.get("eligibility_for_confirmatory"),
        "eligibility_for_confirmatory",
    )
    means = summary.groupby("method")["final_mse"].mean()
    baseline = float(means.loc["current_latent_mse_forced"])
    selected_forced = float(means.loc["development_selected_score_forced"])
    selected_no_op = float(means.loc["development_selected_score_no_op"])
    no_op_rows = summary.loc[
        summary["method"] == "development_selected_score_no_op"
    ]
    improves_all = bool((no_op_rows["final_mse"] < no_op_rows["initial_mse"]).all())
    reduction = 1.0 - selected_no_op / max(baseline, 1e-12)
    criteria = {
        "implementation_integrity": bool(implementation_integrity_passed),
        "selected_pair_matches_score_audit": bool(selected_pair_matches),
        "selected_no_op_improves_every_target_from_blank": improves_all,
        "minimum_mean_reduction_vs_current_latent_mse_forced": bool(
            reduction
            >= float(
                eligibility[
                    "minimum_mean_final_mse_reduction_vs_current_latent_mse_forced"
                ]
            )
        ),
    }
    eligible = bool(all(criteria.values()))
    return {
        "status": "eligible_for_confirmatory" if eligible else "not_eligible",
        "planner_development_completed": True,
        "criteria_frozen_before_planner_development": True,
        "criteria_passed": criteria,
        "implementation_integrity_passed": bool(implementation_integrity_passed),
        "selected_pair_matches_score_audit": bool(selected_pair_matches),
        "selected_predictor_family": SELECTED_PREDICTOR_FAMILY,
        "selected_score_name": SELECTED_SCORE_NAME,
        "mean_current_latent_mse_forced_final_mse": baseline,
        "mean_selected_score_forced_final_mse": selected_forced,
        "mean_selected_score_no_op_final_mse": selected_no_op,
        "selected_no_op_mean_final_mse_reduction_vs_current_latent_mse_forced": reduction,
        "selected_no_op_improved_every_target_from_blank": improves_all,
        "confirmatory_authorized": False,
        "models_trained_or_finetuned": False,
        "score_audit_result_unchanged": True,
        "controlled_result_unchanged": True,
        "historical_results_unchanged": True,
    }
