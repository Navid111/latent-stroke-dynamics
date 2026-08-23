"""Frozen renderer-data construction for the authorized Phase B0 run."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from .planning import ProposalConfig, pixel_mse, propose_strokes, run_planner
from .representation_extension import images_to_grayscale_tensor
from .renderer import Stroke, blank_canvas, random_base_canvas, render_stroke, sample_stroke


@dataclass(frozen=True)
class PhaseBTransitionExample:
    current: Image.Image
    next_canvas: Image.Image
    stroke: Stroke | None
    crowding: int
    sample_id: int
    no_op: bool
    fingerprint: str


@dataclass(frozen=True)
class TransitionTensorPayload:
    examples: tuple[PhaseBTransitionExample, ...]
    current: torch.Tensor
    next_canvas: torch.Tensor
    actions: torch.Tensor
    no_op: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.current.shape[0])


@dataclass(frozen=True)
class PlannerCandidateSet:
    set_id: int
    target_seed: int
    trajectory_seed: int
    candidate_seed: int
    state_name: str
    current: Image.Image
    target: Image.Image
    candidates: tuple[Stroke | None, ...]


@dataclass(frozen=True)
class PlannerTensorPayload:
    records: tuple[PlannerCandidateSet, ...]
    current: torch.Tensor
    next_canvas: torch.Tensor
    target: torch.Tensor
    actions: torch.Tensor
    exact_progress: torch.Tensor
    set_index: torch.Tensor
    candidate_index: torch.Tensor

    @property
    def candidate_sets(self) -> int:
        return len(self.records)

    @property
    def size(self) -> int:
        return int(self.current.shape[0])


def stroke_action_raster(stroke: Stroke | None, canvas_size: int = 64) -> torch.Tensor:
    """Return coverage and coverage×raw-value channels; no-op is exactly zero."""

    if canvas_size != 64:
        raise ValueError("Phase B0 action rasters require a 64x64 canvas.")
    if stroke is None:
        return torch.zeros(2, canvas_size, canvas_size, dtype=torch.float32)
    geometry = Stroke(stroke.x0, stroke.y0, stroke.x1, stroke.y1, stroke.width, 0)
    rendered = render_stroke(blank_canvas(canvas_size), geometry)
    coverage = torch.from_numpy((np.asarray(rendered) != 255).astype(np.float32))
    return torch.stack((coverage, coverage * (float(stroke.value) / 255.0)))


def _fingerprint(
    current: Image.Image,
    next_canvas: Image.Image,
    action: torch.Tensor,
    no_op: bool,
) -> str:
    digest = sha256()
    digest.update(np.asarray(current, dtype=np.uint8).tobytes())
    digest.update(np.asarray(next_canvas, dtype=np.uint8).tobytes())
    digest.update(action.numpy().tobytes())
    digest.update(bytes([int(no_op)]))
    return digest.hexdigest()


def build_transition_payload(
    *,
    samples: int,
    seed: int,
    crowding_levels: tuple[int, ...],
    no_op_fraction: float = 0.1,
) -> TransitionTensorPayload:
    """Generate one independent transition split with an exact no-op fraction."""

    if samples < 10 or not 0.0 < no_op_fraction < 1.0:
        raise ValueError("Invalid transition sample count or no-op fraction.")
    rng = np.random.default_rng(seed)
    no_op_count = int(round(samples * no_op_fraction))
    no_op_indices = set(int(value) for value in rng.choice(samples, no_op_count, replace=False))
    examples: list[PhaseBTransitionExample] = []
    actions: list[torch.Tensor] = []
    for sample_id in range(samples):
        crowding = int(rng.choice(crowding_levels))
        current = random_base_canvas(64, crowding, rng)
        if sample_id in no_op_indices:
            stroke = None
            next_canvas = current.copy()
            no_op = True
        else:
            no_op = False
            for _ in range(100):
                stroke = sample_stroke(
                    rng,
                    width_choices=(1, 2, 3, 4),
                    value_choices=(0, 32, 64, 96, 128),
                    min_length=0.2,
                )
                next_canvas = render_stroke(current, stroke)
                if not np.array_equal(np.asarray(current), np.asarray(next_canvas)):
                    break
            else:
                raise RuntimeError("Could not construct a changing Phase B0 transition.")
        action = stroke_action_raster(stroke)
        examples.append(
            PhaseBTransitionExample(
                current=current,
                next_canvas=next_canvas,
                stroke=stroke,
                crowding=crowding,
                sample_id=sample_id,
                no_op=no_op,
                fingerprint=_fingerprint(current, next_canvas, action, no_op),
            )
        )
        actions.append(action)
    if sum(int(item.no_op) for item in examples) != no_op_count:
        raise RuntimeError("Phase B0 no-op count changed.")
    return TransitionTensorPayload(
        examples=tuple(examples),
        current=images_to_grayscale_tensor([item.current for item in examples]),
        next_canvas=images_to_grayscale_tensor([item.next_canvas for item in examples]),
        actions=torch.stack(actions),
        no_op=torch.tensor([item.no_op for item in examples], dtype=torch.bool),
    )


def _proposal_config(count: int) -> ProposalConfig:
    return ProposalConfig(
        count=count,
        error_guided_fraction=0.8,
        min_length=0.1,
        max_length=0.6,
        width_choices=(1, 2, 3, 4),
        value_choices=(0, 32, 64, 96, 128),
    )


def planner_state_bank(
    target: Image.Image,
    trajectory_seed: int,
    state_names: tuple[str, ...],
) -> dict[str, Image.Image]:
    """Construct the eight frozen blank/exact/random supervision states."""

    exact = run_planner(
        target,
        "exact",
        steps=80,
        seed=trajectory_seed,
        proposal_config=_proposal_config(128),
        capture_frames=True,
    )
    random = run_planner(
        target,
        "random",
        steps=80,
        seed=trajectory_seed,
        proposal_config=_proposal_config(128),
        capture_frames=True,
    )
    available = {
        "blank": blank_canvas(64),
        "exact_20": exact.frames[20],
        "exact_40": exact.frames[40],
        "exact_60": exact.frames[60],
        "random_20": random.frames[20],
        "random_40": random.frames[40],
        "random_60": random.frames[60],
        "random_80": random.frames[80],
    }
    if set(state_names) != set(available):
        raise ValueError("Frozen Phase B0 planner-state names changed.")
    return {name: available[name] for name in state_names}


def build_planner_payload(
    config: Mapping[str, Any],
    section_name: str,
) -> PlannerTensorPayload:
    """Build fixed 32-way sets: no-op index zero plus 31 guided strokes."""

    section = config["development"][section_name]
    target_seeds = tuple(int(value) for value in section["target_seeds"])
    trajectory_seeds = tuple(int(value) for value in section["trajectory_seeds"])
    candidate_seeds = tuple(int(value) for value in section["candidate_seeds"])
    state_names = tuple(str(value) for value in section["states"])
    candidate_count = int(section["candidates_per_state"])
    if candidate_count != 32:
        raise ValueError("Phase B0 planner supervision requires 32 candidates.")
    records: list[PlannerCandidateSet] = []
    current_tensors: list[torch.Tensor] = []
    next_tensors: list[torch.Tensor] = []
    target_tensors: list[torch.Tensor] = []
    action_tensors: list[torch.Tensor] = []
    progress_tensors: list[torch.Tensor] = []
    set_indices: list[torch.Tensor] = []
    candidate_indices: list[torch.Tensor] = []
    for target_seed, trajectory_seed, candidate_seed in zip(
        target_seeds, trajectory_seeds, candidate_seeds, strict=True
    ):
        target = random_base_canvas(
            64,
            int(config["renderer"]["target_strokes"]),
            np.random.default_rng(target_seed),
        )
        states = planner_state_bank(target, trajectory_seed, state_names)
        for state_index, state_name in enumerate(state_names):
            current = states[state_name]
            rng = np.random.default_rng(
                np.random.SeedSequence([candidate_seed, state_index, 0])
            )
            strokes = propose_strokes(current, target, rng, _proposal_config(31))
            candidates: tuple[Stroke | None, ...] = (None, *strokes)
            next_images = (current.copy(),) + tuple(
                render_stroke(current, stroke) for stroke in strokes
            )
            signatures = {np.asarray(image).tobytes() for image in next_images}
            if len(signatures) != candidate_count:
                raise RuntimeError("A Phase B0 planner candidate set is not unique.")
            current_error = pixel_mse(current, target)
            progress = torch.tensor(
                [current_error - pixel_mse(image, target) for image in next_images],
                dtype=torch.float32,
            )
            if float(progress[0]) != 0.0:
                raise RuntimeError("No-op exact progress must be zero.")
            set_id = len(records)
            records.append(
                PlannerCandidateSet(
                    set_id=set_id,
                    target_seed=target_seed,
                    trajectory_seed=trajectory_seed,
                    candidate_seed=candidate_seed,
                    state_name=state_name,
                    current=current,
                    target=target,
                    candidates=candidates,
                )
            )
            current_tensor = images_to_grayscale_tensor((current,)).expand(
                candidate_count, -1, -1, -1
            ).clone()
            target_tensor = images_to_grayscale_tensor((target,)).expand(
                candidate_count, -1, -1, -1
            ).clone()
            current_tensors.append(current_tensor)
            target_tensors.append(target_tensor)
            next_tensors.append(images_to_grayscale_tensor(next_images))
            action_tensors.append(torch.stack([stroke_action_raster(item) for item in candidates]))
            progress_tensors.append(progress)
            set_indices.append(torch.full((candidate_count,), set_id, dtype=torch.int64))
            candidate_indices.append(torch.arange(candidate_count, dtype=torch.int64))
    return PlannerTensorPayload(
        records=tuple(records),
        current=torch.cat(current_tensors),
        next_canvas=torch.cat(next_tensors),
        target=torch.cat(target_tensors),
        actions=torch.cat(action_tensors),
        exact_progress=torch.cat(progress_tensors),
        set_index=torch.cat(set_indices),
        candidate_index=torch.cat(candidate_indices),
    )


def fit_progress_statistics(payload: PlannerTensorPayload) -> tuple[float, float]:
    mean = float(payload.exact_progress.mean().item())
    std = float(payload.exact_progress.std(unbiased=False).item())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-12:
        raise RuntimeError("Train-only progress statistics are invalid.")
    return mean, std
