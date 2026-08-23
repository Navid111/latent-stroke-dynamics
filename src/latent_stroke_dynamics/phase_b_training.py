"""Training and diagnostic evaluation for the frozen Phase B0 models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .extension_training import model_state_sha256, seed_everything
from .gate2 import TransitionExample, build_counterfactual_set
from .latent_smoke import spearman_rank_correlation
from .phase_b_data import (
    PlannerTensorPayload,
    TransitionTensorPayload,
    stroke_action_raster,
)
from .phase_b_joint_embedding import (
    MultiScaleActionJointEmbeddingModel,
    phase_b_objective,
    trainable_parameter_count,
)
from .representation_extension import images_to_grayscale_tensor


@dataclass
class PhaseBFitResult:
    model: MultiScaleActionJointEmbeddingModel
    variant: str
    seed: int
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, Any]]
    wall_clock_seconds: float
    compute_cap_reached: bool


def _transition_mean(
    model: MultiScaleActionJointEmbeddingModel,
    payload: TransitionTensorPayload,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, payload.size, batch_size):
            stop = min(start + batch_size, payload.size)
            output = model(payload.current[start:stop], payload.actions[start:stop])
            target = model.encode_target(payload.next_canvas[start:stop])
            loss = phase_b_objective(
                variant="joint_prediction_only",
                online_features=output["current"],
                predicted_next=output["predicted_next"],
                target_next=target,
                residuals=output["residual"],
                action_rasters=payload.actions[start:stop],
                no_op_examples=payload.no_op[start:stop],
            )["total"]
            total += float(loss.item()) * (stop - start)
            count += stop - start
    return total / count


def _planner_mean(
    model: MultiScaleActionJointEmbeddingModel,
    payload: PlannerTensorPayload,
    progress_mean: float,
    progress_std: float,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for set_id in range(payload.candidate_sets):
            indices = torch.nonzero(payload.set_index == set_id, as_tuple=False).squeeze(1)
            output = model(
                payload.current[indices],
                payload.actions[indices],
                payload.target[indices],
            )
            target = model.encode_target(payload.next_canvas[indices])
            loss = phase_b_objective(
                variant="joint_prediction_progress",
                online_features=output["current"],
                predicted_next=output["predicted_next"],
                target_next=target,
                residuals=output["residual"],
                action_rasters=payload.actions[indices],
                no_op_examples=payload.candidate_index[indices].eq(0),
                predicted_progress=output["predicted_progress"].reshape(1, -1),
                exact_progress=payload.exact_progress[indices].reshape(1, -1),
                progress_training_mean=progress_mean,
                progress_training_std=progress_std,
            )["total"]
            values.append(float(loss.item()))
    return float(np.mean(values))


def train_phase_b_variant(
    variant: str,
    train: TransitionTensorPayload,
    validation: TransitionTensorPayload,
    planner_train: PlannerTensorPayload,
    planner_validation: PlannerTensorPayload,
    *,
    progress_mean: float,
    progress_std: float,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    gradient_clip_norm: float,
    wall_clock_cap_hours: float,
) -> PhaseBFitResult:
    """Fit one preregistered variant with paired initialization and early stopping."""

    if variant not in {"joint_prediction_only", "joint_prediction_progress"}:
        raise ValueError("Unexpected Phase B0 training variant.")
    seed_everything(seed)
    model = MultiScaleActionJointEmbeddingModel().cpu().train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    cap_seconds = wall_clock_cap_hours * 3600.0
    cap_reached = False
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        permutation = torch.randperm(train.size, generator=generator)
        transition_total = 0.0
        transition_count = 0
        for start in range(0, train.size, batch_size):
            indices = permutation[start : start + batch_size]
            output = model(train.current[indices], train.actions[indices])
            target = model.encode_target(train.next_canvas[indices])
            loss = phase_b_objective(
                variant="joint_prediction_only",
                online_features=output["current"],
                predicted_next=output["predicted_next"],
                target_next=target,
                residuals=output["residual"],
                action_rasters=train.actions[indices],
                no_op_examples=train.no_op[indices],
            )["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                gradient_clip_norm,
            )
            optimizer.step()
            model.update_target_encoder(0.99)
            transition_total += float(loss.detach().item()) * len(indices)
            transition_count += len(indices)

        planner_train_mean: float | None = None
        if variant == "joint_prediction_progress":
            set_order = torch.randperm(planner_train.candidate_sets, generator=generator)
            planner_values: list[float] = []
            for set_id_tensor in set_order:
                set_id = int(set_id_tensor)
                indices = torch.nonzero(
                    planner_train.set_index == set_id, as_tuple=False
                ).squeeze(1)
                output = model(
                    planner_train.current[indices],
                    planner_train.actions[indices],
                    planner_train.target[indices],
                )
                target = model.encode_target(planner_train.next_canvas[indices])
                loss = phase_b_objective(
                    variant="joint_prediction_progress",
                    online_features=output["current"],
                    predicted_next=output["predicted_next"],
                    target_next=target,
                    residuals=output["residual"],
                    action_rasters=planner_train.actions[indices],
                    no_op_examples=planner_train.candidate_index[indices].eq(0),
                    predicted_progress=output["predicted_progress"].reshape(1, -1),
                    exact_progress=planner_train.exact_progress[indices].reshape(1, -1),
                    progress_training_mean=progress_mean,
                    progress_training_std=progress_std,
                )["total"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    gradient_clip_norm,
                )
                optimizer.step()
                model.update_target_encoder(0.99)
                planner_values.append(float(loss.detach().item()))
            planner_train_mean = float(np.mean(planner_values))

        train_transition = transition_total / transition_count
        validation_transition = _transition_mean(model, validation, batch_size)
        validation_planner: float | None = None
        selection = validation_transition
        if variant == "joint_prediction_progress":
            validation_planner = _planner_mean(
                model, planner_validation, progress_mean, progress_std
            )
            selection = 0.5 * (validation_transition + validation_planner)
        if not np.isfinite(selection):
            raise RuntimeError("Phase B0 validation loss is non-finite.")
        history.append(
            {
                "variant": variant,
                "epoch": epoch,
                "train_transition_total": train_transition,
                "train_planner_total": planner_train_mean,
                "validation_transition_total": validation_transition,
                "validation_planner_total": validation_planner,
                "selection_loss": selection,
            }
        )
        if selection < best_loss - 1e-12:
            best_loss = selection
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
        if time.perf_counter() - started >= cap_seconds:
            cap_reached = True
            break
    model.load_state_dict(best_state, strict=True)
    return PhaseBFitResult(
        model=model.eval(),
        variant=variant,
        seed=seed,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        history=history,
        wall_clock_seconds=time.perf_counter() - started,
        compute_cap_reached=cap_reached,
    )


def freeze_phase_b_model(
    model: MultiScaleActionJointEmbeddingModel,
) -> MultiScaleActionJointEmbeddingModel:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def save_phase_b_checkpoint(
    fit: PhaseBFitResult,
    path: str | Path,
    *,
    progress_mean: float,
    progress_std: float,
) -> tuple[Path, str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    freeze_phase_b_model(fit.model)
    digest = model_state_sha256(fit.model)
    torch.save(
        {
            "format_version": 1,
            "architecture": "MultiScaleActionJointEmbeddingModel",
            "variant": fit.variant,
            "seed": fit.seed,
            "best_epoch": fit.best_epoch,
            "best_validation_loss": fit.best_validation_loss,
            "trainable_parameter_count": trainable_parameter_count(
                MultiScaleActionJointEmbeddingModel()
            ),
            "progress_training_mean": progress_mean,
            "progress_training_std": progress_std,
            "state_sha256": digest,
            "state_dict": {name: value.detach().cpu() for name, value in fit.model.state_dict().items()},
        },
        output,
    )
    return output, digest


def feature_statistics(
    model: MultiScaleActionJointEmbeddingModel,
    payload: TransitionTensorPayload,
    batch_size: int = 32,
) -> dict[str, dict[str, float]]:
    features: dict[str, list[torch.Tensor]] = {"32": [], "16": []}
    model.eval()
    with torch.inference_mode():
        for start in range(0, payload.size, batch_size):
            encoded = model.online_encoder(payload.current[start : start + batch_size])
            for scale in features:
                features[scale].append(encoded[scale].cpu())
    result: dict[str, dict[str, float]] = {}
    for scale, parts in features.items():
        value = torch.cat(parts).permute(0, 2, 3, 1).reshape(-1, parts[0].shape[1])
        std = value.std(dim=0, unbiased=False)
        centered = value - value.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (len(value) - 1)
        off = covariance - torch.diag(torch.diagonal(covariance))
        result[scale] = {
            "mean_channel_std": float(std.mean().item()),
            "mean_squared_off_diagonal_covariance": float(
                (off.square().sum() / value.shape[1]).item()
            ),
        }
    return result


def four_way_retrieval(
    model: MultiScaleActionJointEmbeddingModel,
    payload: TransitionTensorPayload,
    batch_size: int = 16,
) -> dict[str, float | int]:
    examples = [item for item in payload.examples if not item.no_op]
    correct = 0
    total = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            current = images_to_grayscale_tensor([item.current for item in batch])
            actions = torch.stack([stroke_action_raster(item.stroke) for item in batch])
            output = model(current, actions)
            candidate_images = []
            for item in batch:
                assert item.stroke is not None
                counterfactual = build_counterfactual_set(
                    TransitionExample(
                        current=item.current,
                        next_canvas=item.next_canvas,
                        stroke=item.stroke,
                        crowding=item.crowding,
                        sample_id=item.sample_id,
                    )
                )
                candidate_images.extend(counterfactual.canvases)
            encoded = model.encode_target(images_to_grayscale_tensor(candidate_images))
            scores = torch.zeros(len(batch), 4)
            for scale, weight in (("32", 0.5), ("16", 0.5)):
                candidates = encoded[scale].reshape(len(batch), 4, *encoded[scale].shape[1:])
                predicted = output["predicted_next"][scale][:, None]
                scores += weight * F.smooth_l1_loss(
                    predicted.expand_as(candidates), candidates, reduction="none"
                ).mean(dim=(2, 3, 4)).cpu()
            correct += int(scores.argmin(dim=1).eq(0).sum().item())
            total += len(batch)
    return {"examples": total, "top1_accuracy": correct / total}


def planner_candidate_metrics(
    model: MultiScaleActionJointEmbeddingModel,
    payload: PlannerTensorPayload,
    mode: str,
) -> list[dict[str, Any]]:
    if mode not in {"prediction", "progress"}:
        raise ValueError("Unexpected Phase B0 candidate score mode.")
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for set_id in range(payload.candidate_sets):
            indices = torch.nonzero(payload.set_index == set_id, as_tuple=False).squeeze(1)
            output = model(
                payload.current[indices],
                payload.actions[indices],
                payload.target[indices] if mode == "progress" else None,
            )
            if mode == "progress":
                score = output["predicted_progress"].cpu().numpy().astype(np.float64)
            else:
                goal = model.encode_target(payload.target[indices[:1]])
                distance = torch.zeros(len(indices))
                for scale, weight in (("32", 0.5), ("16", 0.5)):
                    distance += weight * F.smooth_l1_loss(
                        output["predicted_next"][scale],
                        goal[scale].expand_as(output["predicted_next"][scale]),
                        reduction="none",
                    ).mean(dim=(1, 2, 3)).cpu()
                score = -distance.numpy().astype(np.float64)
            exact = payload.exact_progress[indices].numpy().astype(np.float64)
            selected = int(np.argmax(score))
            selected_progress = float(exact[selected])
            best = float(exact.max())
            rank = 1 + int(np.sum(exact > selected_progress + 1e-12))
            rows.append(
                {
                    "set_id": set_id,
                    "mode": mode,
                    "selected_index": selected,
                    "exact_selected_rank": rank,
                    "exact_top1": rank == 1,
                    "exact_top5": rank <= 5,
                    "exact_regret": max(0.0, best - selected_progress),
                    "score_exact_spearman": spearman_rank_correlation(score, exact),
                }
            )
    return rows
