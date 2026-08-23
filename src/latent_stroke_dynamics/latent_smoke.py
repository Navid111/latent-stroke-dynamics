"""Guarded smoke utilities and latent observe-predict-execute planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch.nn as nn
from PIL import Image

from .latent_planner import encode_task_latents, latent_candidate_scores
from .planning import (
    ProposalConfig,
    pixel_mae,
    pixel_mse,
    propose_strokes,
    render_candidate_canvases,
)
from .representation_extension import LatentChannelStatistics, StrokeAutoencoder
from .renderer import Stroke, blank_canvas


SMOKE_METHODS = (
    "random",
    "exact_pixel",
    "learned_pixel",
    "latent_mse",
    "latent_ranking",
)


@dataclass(frozen=True)
class SmokeOutputPaths:
    final: Path
    incomplete: Path


@dataclass(frozen=True)
class LatentPlanningStep:
    """One latent-ranked decision followed by exact stroke execution."""

    step: int
    selected_index: int
    stroke: Stroke
    candidate_count: int
    mse_before: float
    mse_after: float
    mae_after: float
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
class LatentPlanningRun:
    """A complete observe-predict-execute-re-encode latent trajectory."""

    method: str
    seed: int
    target: Image.Image
    initial_canvas: Image.Image
    final_canvas: Image.Image
    steps: tuple[LatentPlanningStep, ...]
    frames: tuple[Image.Image, ...]
    target_encoding_count: int
    observed_canvas_encoding_count: int


def smoke_output_paths(config: Mapping[str, Any]) -> SmokeOutputPaths:
    smoke = config.get("smoke")
    if not isinstance(smoke, Mapping):
        raise ValueError("smoke must be a mapping.")
    final = Path(str(smoke.get("output_dir", "")))
    if not final.name:
        raise ValueError("Smoke output directory is invalid.")
    return SmokeOutputPaths(
        final=final,
        incomplete=final.with_name(final.name + ".incomplete"),
    )


def require_smoke_outputs_absent(paths: SmokeOutputPaths) -> None:
    """Refuse both completed output and preserved incomplete output."""

    if paths.final.exists():
        raise FileExistsError(f"Smoke output already exists: {paths.final}")
    if paths.incomplete.exists():
        raise FileExistsError(
            f"Incomplete smoke output exists: {paths.incomplete}. "
            "Preserve and review it before any retry."
        )


def require_smoke_authorized(config: Mapping[str, Any]) -> None:
    """Stop before loading models or generating the reserved target."""

    smoke = config.get("smoke")
    if not isinstance(smoke, Mapping) or smoke.get("authorized") is not True:
        raise PermissionError(
            "The latent-planner smoke is not authorized. No models were loaded "
            "and the reserved target was not generated."
        )


def validate_smoke_runner_request(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the current unauthorized smoke command without side effects."""

    if config.get("status") != "hashes_frozen_before_smoke":
        raise ValueError("Latent predictor hashes must be frozen before smoke validation.")
    planner = config.get("planner")
    smoke = config.get("smoke")
    controlled = config.get("controlled")
    predictors = config.get("latent_predictors")
    if not all(isinstance(value, Mapping) for value in (planner, smoke, controlled, predictors)):
        raise ValueError("Planner smoke configuration is malformed.")
    assert isinstance(planner, Mapping)
    assert isinstance(smoke, Mapping)
    assert isinstance(controlled, Mapping)
    assert isinstance(predictors, Mapping)
    if tuple(planner.get("methods", ())) != SMOKE_METHODS:
        raise ValueError("The smoke runner must contain the five frozen methods.")
    if smoke.get("authorized") is not False:
        raise ValueError("Validation-only mode requires smoke authorization false.")
    if controlled.get("authorized") is not False:
        raise ValueError("Controlled planning must remain unauthorized.")
    hashes = [
        entry.get("state_sha256")
        for method in ("mse_only", "ranking_aware")
        for entry in predictors.get(method, ())
        if isinstance(entry, Mapping)
    ]
    if len(hashes) != 6 or any(
        not isinstance(value, str) or len(value) != 64 for value in hashes
    ):
        raise ValueError("All six latent predictor hashes must be frozen.")
    paths = smoke_output_paths(config)
    require_smoke_outputs_absent(paths)
    return {
        "status": "latent_planner_smoke_runner_valid_unauthorized",
        "config_status": config["status"],
        "methods": list(SMOKE_METHODS),
        "target_seed_reserved": smoke["target_seed"],
        "planner_seed_reserved": smoke["planner_seed"],
        "steps": smoke["steps"],
        "candidates_per_step": smoke["candidates_per_step"],
        "latent_predictor_hashes_frozen": True,
        "smoke_authorized": False,
        "controlled_authorized": False,
        "smoke_output_dir_available": True,
        "smoke_incomplete_dir_available": True,
        "models_loaded": False,
        "target_generated": False,
        "planner_data_generated": False,
        "models_trained_or_finetuned": False,
        "historical_results_unchanged": True,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not bool(np.isfinite(values).all()):
        raise ValueError("Rank inputs must be one-dimensional and finite.")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        mean_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = mean_rank
        start = stop
    return ranks


def spearman_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return finite tie-aware Spearman rank correlation without SciPy."""

    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if left_ranks.shape != right_ranks.shape or len(left_ranks) < 2:
        raise ValueError("Spearman inputs must share a length of at least two.")
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(
        np.sqrt(
            np.sum(left_centered * left_centered)
            * np.sum(right_centered * right_centered)
        )
    )
    if denominator == 0.0:
        return 0.0
    value = float(np.sum(left_centered * right_centered) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _planner_step_rng(seed: int, step: int) -> np.random.Generator:
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    return np.random.default_rng(np.random.SeedSequence([seed, step, 0]))


def run_latent_planner(
    target: Image.Image,
    method: str,
    autoencoder: StrokeAutoencoder,
    statistics: LatentChannelStatistics,
    predictors: Sequence[nn.Module],
    *,
    steps: int,
    seed: int,
    proposal_config: ProposalConfig,
    prediction_batch_size: int,
    capture_frames: bool,
) -> LatentPlanningRun:
    """Plan from predicted next latents while executing and observing exactly."""

    if method not in ("latent_mse", "latent_ranking"):
        raise ValueError("Unknown latent planner method.")
    if target.mode != "L" or target.size != (64, 64):
        raise ValueError("Latent planning requires a 64x64 grayscale target.")
    if steps < 1 or prediction_batch_size < 1:
        raise ValueError("Steps and prediction batch size must be positive.")
    if len(predictors) != 3:
        raise ValueError("Latent planning requires exactly three frozen predictors.")
    if any(parameter.requires_grad for parameter in autoencoder.parameters()):
        raise ValueError("Latent planning requires a frozen autoencoder.")
    if any(parameter.requires_grad for model in predictors for parameter in model.parameters()):
        raise ValueError("Latent planning requires frozen predictors.")

    target_tokens = encode_task_latents(
        autoencoder,
        statistics,
        (target,),
        batch_size=1,
    )
    initial = blank_canvas(64)
    current = initial.copy()
    records: list[LatentPlanningStep] = []
    frames: list[Image.Image] = [current.copy()] if capture_frames else []

    for step in range(1, steps + 1):
        candidates = propose_strokes(
            current,
            target,
            rng=_planner_step_rng(seed, step),
            config=proposal_config,
        )
        current_tokens = encode_task_latents(
            autoencoder,
            statistics,
            (current,),
            batch_size=1,
        )
        predicted_scores, per_model_scores = latent_candidate_scores(
            predictors,
            current_tokens,
            target_tokens,
            candidates,
            batch_size=prediction_batch_size,
        )
        selected_index = int(np.argmin(predicted_scores))
        exact_canvases = render_candidate_canvases(current, candidates)
        exact_scores = np.asarray(
            [pixel_mse(canvas, target) for canvas in exact_canvases],
            dtype=np.float64,
        )
        next_canvas = exact_canvases[selected_index]
        exact_selected = float(exact_scores[selected_index])
        exact_best = float(exact_scores.min())
        tolerance = 1e-12
        exact_rank = 1 + int(np.sum(exact_scores < exact_selected - tolerance))
        mse_before = pixel_mse(current, target)
        record = LatentPlanningStep(
            step=step,
            selected_index=selected_index,
            stroke=candidates[selected_index],
            candidate_count=len(candidates),
            mse_before=mse_before,
            mse_after=exact_selected,
            mae_after=pixel_mae(next_canvas, target),
            predicted_selected_score=float(predicted_scores[selected_index]),
            predicted_score_range=float(np.ptp(predicted_scores)),
            per_model_selected_scores=tuple(
                float(value) for value in per_model_scores[:, selected_index]
            ),
            exact_best_candidate_mse=exact_best,
            exact_selected_rank=exact_rank,
            exact_top1=exact_rank == 1,
            exact_top5=exact_rank <= 5,
            exact_regret=max(0.0, exact_selected - exact_best),
            score_exact_spearman=spearman_rank_correlation(
                predicted_scores,
                exact_scores,
            ),
            improved=exact_selected < mse_before,
        )
        numeric = (
            record.mse_before,
            record.mse_after,
            record.mae_after,
            record.predicted_selected_score,
            record.predicted_score_range,
            record.exact_best_candidate_mse,
            record.exact_regret,
            record.score_exact_spearman,
            *record.per_model_selected_scores,
        )
        if not bool(np.isfinite(np.asarray(numeric, dtype=np.float64)).all()):
            raise RuntimeError("Latent planning produced a non-finite diagnostic.")
        records.append(record)
        current = next_canvas
        if capture_frames:
            frames.append(current.copy())

    return LatentPlanningRun(
        method=method,
        seed=seed,
        target=target.copy(),
        initial_canvas=initial,
        final_canvas=current,
        steps=tuple(records),
        frames=tuple(frames),
        target_encoding_count=1,
        observed_canvas_encoding_count=steps,
    )
