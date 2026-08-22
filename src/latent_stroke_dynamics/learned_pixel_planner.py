"""Saved pixel-predictor checkpoints and learned one-step stroke planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .gate2 import ACTION_DIM, parameter_count, stroke_action_vector
from .pixel_control import (
    PIXEL_INPUT_DIM,
    MLPPixelDeltaPredictor,
    image_to_normalized_tensor,
    stroke_pixel_mask,
)
from .planning import (
    ProposalConfig,
    pixel_mae,
    pixel_mse,
    propose_strokes,
    render_candidate_canvases,
)
from .renderer import Stroke, blank_canvas


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_TYPE = "stage3_pixel_mlp_demo"


@dataclass(frozen=True)
class PixelCheckpointMetadata:
    """Reproducibility and scope metadata stored beside demonstration weights."""

    checkpoint_type: str
    format_version: int
    canvas_size: int
    pixel_input_dim: int
    action_dim: int
    architecture: str
    hidden_dim: int
    parameter_count: int
    model_seed: int
    train_seed: int
    validation_seed: int
    train_samples: int
    validation_samples: int
    crowding: tuple[int, ...]
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    best_epoch: int
    best_validation_loss: float
    test_rows_used_for_training_or_selection: bool

    def __post_init__(self) -> None:
        if self.checkpoint_type != CHECKPOINT_TYPE:
            raise ValueError("Unexpected checkpoint_type.")
        if self.format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint format version.")
        if self.canvas_size < 8 or self.hidden_dim < 1:
            raise ValueError("Invalid canvas or hidden dimension.")
        if self.pixel_input_dim != PIXEL_INPUT_DIM or self.action_dim != ACTION_DIM:
            raise ValueError("Checkpoint input dimensions do not match the code.")
        if self.architecture != "MLPPixelDeltaPredictor":
            raise ValueError("Only MLPPixelDeltaPredictor checkpoints are supported.")
        if self.parameter_count < 1 or self.train_samples < 1 or self.validation_samples < 1:
            raise ValueError("Checkpoint counts must be positive.")
        if not self.crowding or min(self.crowding) < 0:
            raise ValueError("crowding must be non-empty and non-negative.")
        if self.best_epoch < 1 or not np.isfinite(self.best_validation_loss):
            raise ValueError("Checkpoint must record a finite selected epoch.")
        if self.test_rows_used_for_training_or_selection:
            raise ValueError("Stage 3 deployment checkpoints must not use test rows.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["crowding"] = list(self.crowding)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PixelCheckpointMetadata":
        values = dict(payload)
        values["crowding"] = tuple(int(value) for value in values["crowding"])
        return cls(**values)


@dataclass(frozen=True)
class LearnedPlanningStep:
    """One learned ranking decision followed by exact stroke execution."""

    step: int
    selected_index: int
    stroke: Stroke
    candidate_count: int
    mse_before: float
    mse_after: float
    mae_after: float
    predicted_selected_mse: float
    exact_best_candidate_mse: float
    exact_selected_rank: int
    exact_top1: bool
    exact_top5: bool
    exact_regret: float
    improved: bool


@dataclass(frozen=True)
class LearnedPlanningRun:
    """A complete learned one-step planning trajectory."""

    seed: int
    target: Image.Image
    initial_canvas: Image.Image
    final_canvas: Image.Image
    steps: tuple[LearnedPlanningStep, ...]
    frames: tuple[Image.Image, ...]


def state_dict_sha256(model: nn.Module) -> str:
    """Return a deterministic digest of model parameter names, shapes, and bytes."""

    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(values.shape)).encode("ascii"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def save_pixel_checkpoint(
    model: MLPPixelDeltaPredictor,
    metadata: PixelCheckpointMetadata,
    path: str | Path,
) -> Path:
    """Atomically save demonstration weights with strict scope metadata."""

    if parameter_count(model) != metadata.parameter_count:
        raise ValueError("Model parameter count does not match checkpoint metadata.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    payload = {
        "metadata": metadata.to_dict(),
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "state_dict_sha256": state_dict_sha256(model),
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)
    return output_path


def load_pixel_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[MLPPixelDeltaPredictor, PixelCheckpointMetadata]:
    """Load and integrity-check a Stage 3 demonstration checkpoint."""

    resolved_device = torch.device(device)
    payload = torch.load(
        Path(path),
        map_location=resolved_device,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("Checkpoint metadata is missing or invalid.")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Checkpoint state_dict is missing or invalid.")
    metadata = PixelCheckpointMetadata.from_dict(payload["metadata"])
    model = MLPPixelDeltaPredictor(hidden_dim=metadata.hidden_dim)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(resolved_device).eval()
    if parameter_count(model) != metadata.parameter_count:
        raise ValueError("Loaded model parameter count does not match metadata.")
    expected_digest = str(payload.get("state_dict_sha256", ""))
    if not expected_digest or state_dict_sha256(model) != expected_digest:
        raise ValueError("Checkpoint state-dict integrity check failed.")
    return model, metadata


def learned_candidate_scores(
    model: nn.Module,
    current: Image.Image,
    target: Image.Image,
    candidates: Sequence[Stroke],
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Predict every candidate next canvas and score full target pixel MSE."""

    if not candidates:
        raise ValueError("At least one candidate is required.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if current.mode != "L" or target.mode != "L" or current.size != target.size:
        raise ValueError("current and target must be same-sized grayscale images.")
    if current.width != current.height:
        raise ValueError("Pixel planning expects square canvases.")

    resolved_device = torch.device(device)
    model = model.to(resolved_device).eval()
    current_tensor = image_to_normalized_tensor(current).to(resolved_device)
    target_tensor = image_to_normalized_tensor(target).to(resolved_device)
    score_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            candidate_batch = candidates[start : start + batch_size]
            actions = torch.stack(
                [stroke_action_vector(stroke) for stroke in candidate_batch]
            ).to(resolved_device)
            masks = torch.stack(
                [stroke_pixel_mask(stroke, current.width) for stroke in candidate_batch]
            ).to(resolved_device)
            repeated_current = current_tensor[None, :, :].expand(
                len(candidate_batch), -1, -1
            )
            predicted_delta = model(repeated_current, actions, masks)
            predicted_next = (repeated_current + predicted_delta).clamp(0.0, 1.0)
            scores = (predicted_next - target_tensor[None, :, :]).square()
            score_batches.append(scores.flatten(start_dim=1).mean(dim=1).cpu())
    result = torch.cat(score_batches).numpy().astype(np.float64, copy=False)
    if result.shape != (len(candidates),) or not bool(np.isfinite(result).all()):
        raise RuntimeError("Learned candidate scoring produced invalid metrics.")
    return result


def _planner_step_rng(seed: int, step: int) -> np.random.Generator:
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    return np.random.default_rng(np.random.SeedSequence([seed, step, 0]))


def run_learned_planner(
    target: Image.Image,
    model: nn.Module,
    steps: int = 100,
    seed: int = 0,
    proposal_config: ProposalConfig | None = None,
    prediction_batch_size: int = 32,
    device: str | torch.device = "cpu",
    capture_frames: bool = False,
) -> LearnedPlanningRun:
    """Rank with a pixel model, execute exactly, observe, and replan."""

    if target.mode != "L" or target.width != target.height:
        raise ValueError("target must be a square grayscale ('L') image.")
    if steps < 1:
        raise ValueError("steps must be positive.")
    config = proposal_config or ProposalConfig()
    initial_canvas = blank_canvas(target.width)
    current = initial_canvas.copy()
    records: list[LearnedPlanningStep] = []
    frames: list[Image.Image] = [current.copy()] if capture_frames else []

    for step in range(1, steps + 1):
        candidates = propose_strokes(
            current,
            target,
            rng=_planner_step_rng(seed, step),
            config=config,
        )
        predicted_scores = learned_candidate_scores(
            model,
            current,
            target,
            candidates,
            batch_size=prediction_batch_size,
            device=device,
        )
        selected_index = int(np.argmin(predicted_scores))
        exact_canvases = render_candidate_canvases(current, candidates)
        exact_scores = np.asarray(
            [pixel_mse(canvas, target) for canvas in exact_canvases],
            dtype=np.float64,
        )
        exact_best = float(exact_scores.min())
        exact_selected = float(exact_scores[selected_index])
        tolerance = 1e-12
        exact_rank = 1 + int(np.sum(exact_scores < exact_selected - tolerance))
        exact_regret = max(0.0, exact_selected - exact_best)
        next_canvas = exact_canvases[selected_index]
        mse_before = pixel_mse(current, target)
        records.append(
            LearnedPlanningStep(
                step=step,
                selected_index=selected_index,
                stroke=candidates[selected_index],
                candidate_count=len(candidates),
                mse_before=mse_before,
                mse_after=exact_selected,
                mae_after=pixel_mae(next_canvas, target),
                predicted_selected_mse=float(predicted_scores[selected_index]),
                exact_best_candidate_mse=exact_best,
                exact_selected_rank=exact_rank,
                exact_top1=exact_rank == 1,
                exact_top5=exact_rank <= 5,
                exact_regret=exact_regret,
                improved=exact_selected < mse_before,
            )
        )
        current = next_canvas
        if capture_frames:
            frames.append(current.copy())

    return LearnedPlanningRun(
        seed=seed,
        target=target.copy(),
        initial_canvas=initial_canvas,
        final_canvas=current,
        steps=tuple(records),
        frames=tuple(frames),
    )
