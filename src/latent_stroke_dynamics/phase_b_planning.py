"""Observe-predict-execute planning with frozen Phase B0 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .latent_smoke import spearman_rank_correlation
from .phase_b_data import stroke_action_raster
from .phase_b_joint_embedding import MultiScaleActionJointEmbeddingModel
from .planning import ProposalConfig, pixel_mae, pixel_mse, propose_strokes, render_candidate_canvases
from .representation_extension import images_to_grayscale_tensor
from .renderer import Stroke, blank_canvas


@dataclass(frozen=True)
class PhaseBStopDecision:
    proposal_round: int
    executed_steps: int
    no_op_score: float
    best_stroke_score: float
    current_mse: float
    exact_best_candidate_mse: float
    premature: bool


@dataclass(frozen=True)
class PhaseBPlanningStep:
    step: int
    selected_index: int
    stroke: Stroke
    candidate_count: int
    mse_before: float
    mse_after: float
    mae_after: float
    predicted_selected_score: float
    predicted_score_range: float
    exact_best_candidate_mse: float
    exact_selected_rank: int
    exact_top1: bool
    exact_top5: bool
    exact_regret: float
    score_exact_spearman: float
    improved: bool


@dataclass(frozen=True)
class PhaseBPlanningRun:
    method: str
    seed: int
    target: Image.Image
    initial_canvas: Image.Image
    final_canvas: Image.Image
    steps: tuple[PhaseBPlanningStep, ...]
    frames: tuple[Image.Image, ...]
    proposal_rounds: int
    stop_decision: PhaseBStopDecision | None


def phase_b_candidate_scores(
    model: MultiScaleActionJointEmbeddingModel,
    current: Image.Image,
    target: Image.Image,
    candidates: Sequence[Stroke],
    *,
    mode: str,
    batch_size: int = 32,
) -> np.ndarray:
    if mode not in {"prediction", "progress"} or not candidates:
        raise ValueError("Invalid Phase B0 candidate scoring request.")
    current_tensor = images_to_grayscale_tensor((current,))
    target_tensor = images_to_grayscale_tensor((target,))
    goal = model.encode_target(target_tensor)
    parts: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            actions = torch.stack([stroke_action_raster(stroke) for stroke in batch])
            repeated = current_tensor.expand(len(batch), -1, -1, -1)
            output = model(
                repeated,
                actions,
                target_tensor.expand(len(batch), -1, -1, -1) if mode == "progress" else None,
            )
            if mode == "progress":
                parts.append(output["predicted_progress"].cpu())
            else:
                distance = torch.zeros(len(batch))
                for scale, weight in (("32", 0.5), ("16", 0.5)):
                    distance += weight * F.smooth_l1_loss(
                        output["predicted_next"][scale],
                        goal[scale].expand_as(output["predicted_next"][scale]),
                        reduction="none",
                    ).mean(dim=(1, 2, 3)).cpu()
                parts.append(distance)
    result = torch.cat(parts).numpy().astype(np.float64, copy=False)
    if result.shape != (len(candidates),) or not np.isfinite(result).all():
        raise RuntimeError("Phase B0 candidate scores are invalid.")
    return result


def _no_op_progress_score(
    model: MultiScaleActionJointEmbeddingModel,
    current: Image.Image,
    target: Image.Image,
) -> float:
    current_tensor = images_to_grayscale_tensor((current,))
    target_tensor = images_to_grayscale_tensor((target,))
    with torch.inference_mode():
        output = model(
            current_tensor,
            torch.zeros(1, 2, 64, 64),
            target_tensor,
        )
    return float(output["predicted_progress"].item())


def run_phase_b_planner(
    target: Image.Image,
    model: MultiScaleActionJointEmbeddingModel,
    *,
    mode: str,
    maximum_steps: int,
    seed: int,
    proposal_config: ProposalConfig,
    prediction_batch_size: int,
    allow_no_op: bool,
    capture_frames: bool,
) -> PhaseBPlanningRun:
    if mode not in {"prediction", "progress"}:
        raise ValueError("Unknown Phase B0 planner mode.")
    if allow_no_op and mode != "progress":
        raise ValueError("Only the progress-aligned model may take no-op.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Phase B0 planning requires a frozen model.")
    initial = blank_canvas(64)
    current = initial.copy()
    frames = [current.copy()] if capture_frames else []
    steps: list[PhaseBPlanningStep] = []
    stop: PhaseBStopDecision | None = None
    rounds = 0
    for proposal_round in range(1, maximum_steps + 1):
        rounds += 1
        rng = np.random.default_rng(np.random.SeedSequence([seed, proposal_round, 0]))
        candidates = propose_strokes(current, target, rng, proposal_config)
        predicted = phase_b_candidate_scores(
            model,
            current,
            target,
            candidates,
            mode=mode,
            batch_size=prediction_batch_size,
        )
        canvases = render_candidate_canvases(current, candidates)
        exact = np.asarray([pixel_mse(image, target) for image in canvases], dtype=np.float64)
        current_error = pixel_mse(current, target)
        if allow_no_op:
            no_op_score = _no_op_progress_score(model, current, target)
            if no_op_score >= float(predicted.max()):
                exact_best = float(exact.min())
                stop = PhaseBStopDecision(
                    proposal_round=proposal_round,
                    executed_steps=len(steps),
                    no_op_score=no_op_score,
                    best_stroke_score=float(predicted.max()),
                    current_mse=current_error,
                    exact_best_candidate_mse=exact_best,
                    premature=exact_best < current_error - 1e-12,
                )
                break
        selected = int(np.argmin(predicted) if mode == "prediction" else np.argmax(predicted))
        exact_selected = float(exact[selected])
        exact_best = float(exact.min())
        rank = 1 + int(np.sum(exact < exact_selected - 1e-12))
        correlation_scores = predicted if mode == "prediction" else -predicted
        record = PhaseBPlanningStep(
            step=len(steps) + 1,
            selected_index=selected,
            stroke=candidates[selected],
            candidate_count=len(candidates),
            mse_before=current_error,
            mse_after=exact_selected,
            mae_after=pixel_mae(canvases[selected], target),
            predicted_selected_score=float(predicted[selected]),
            predicted_score_range=float(np.ptp(predicted)),
            exact_best_candidate_mse=exact_best,
            exact_selected_rank=rank,
            exact_top1=rank == 1,
            exact_top5=rank <= 5,
            exact_regret=max(0.0, exact_selected - exact_best),
            score_exact_spearman=spearman_rank_correlation(correlation_scores, exact),
            improved=exact_selected < current_error,
        )
        steps.append(record)
        current = canvases[selected]
        if capture_frames:
            frames.append(current.copy())
    if mode == "prediction":
        method = "joint_prediction_only_forced"
    elif allow_no_op:
        method = "joint_prediction_progress_no_op"
    else:
        method = "joint_prediction_progress_forced"
    return PhaseBPlanningRun(
        method=method,
        seed=seed,
        target=target.copy(),
        initial_canvas=initial,
        final_canvas=current,
        steps=tuple(steps),
        frames=tuple(frames),
        proposal_rounds=rounds,
        stop_decision=stop,
    )
